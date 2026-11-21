#!/usr/bin/env python3
"""Train and run the original CASCADE baseline from local ground-truth .mat files.

Run from the project root:
    python baseline_model.py train --help
    python baseline_model.py predict --help

Sources: https://github.com/HelmchenLabSoftware/Cascade
Pinned revision: 5c5ccef3e45e1bf8b9bf0ae62a23a0f5d1550299
CASCADE contributors include Peter Rupprecht and Adrian Hoffmann (utils.py).
Use and redistribution are governed by the GNU GPL v3.

Modified on 2026-11-20. The command-line interface supplies file paths and
training settings. The upstream CNN definition, calibration, sample weighting,
and training schedule are retained. The integration changes remove package
imports, size the window buffers from the recordings, update SciPy/NumPy
syntax, and make HDF5 inference work with modern Keras. The CLI explicitly
checks for empty training data and loads/resamples a MATLAB trial for inference.
"""

import argparse
import gc
import glob
import os
from pathlib import Path
import time

import numpy as np
from ruamel.yaml import YAML
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve, resample
from scipy.stats import invgauss
import tensorflow as tf


# CASCADE defaults from cascade2p/config.py.
config_template = """\

## Main parameter of this model:

model_name: YOUR_MODEL_NAME                   # Name of the model
sampling_rate: YOUR_SAMPLING_RATE             # Sampling rate in Hz

# Dataset of ground truth data (in folder 'Ground_truth')   Example: DS14-GCaMP6s-m-V1
training_datasets:
- placeholder_1
- placeholder_2

placeholder_1: 0       # protect formatting


# Noise levels for training (integers, normally 1-9)
noise_levels:
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

placeholder_2: 0       # protect formatting


# Standard deviation of Gaussian smoothing in time (sec)
smoothing: 0.2

# Smoothing kernel is symmetric in time (0) or is causal (1)
causal_kernel: 0

## Additional parameters for model specification:


windowsize: 64                   # Windowsize in timepoints
before_frac: 0.5                 # Fraction of timepoints before prediction point (0-1)

# Filter sizes for each convolutional layer
filter_sizes:
- 31
- 19
- 5

# Filter numbers for each convolutional layer
filter_numbers:
- 30
- 40
- 50

dense_expansion: 10              # For dense layer


loss_function: mean_squared_error     # gradient-descent loss function
optimizer: Adagrad                    #                  optimizer

nr_of_epochs: 20                 # Number of training epochs per model
ensemble_size: 5                 # Number of models trained for one noise level
batch_size: 1024                 # Batch size

## Information about status of fitting

training_finished: No            # Yes / No / Running
verbose : 1                      # level of status messages (0: minimal, 1: standard, 2: most, 3: all)


## Additional parameter not specified in template

"""


def read_config(config_yaml_file):
    """Read the same configuration fields as CASCADE's config.py."""
    with open(config_yaml_file, encoding="utf-8") as file:
        return YAML(typ="rt").load(file)


def write_config(config_dict, save_file):
    """Apply CASCADE's upstream defaults and write its config.yaml."""
    yaml = YAML(typ="rt")
    merged = yaml.load(config_template)
    merged.update(config_dict)
    with open(save_file, "w", encoding="utf-8") as file:
        yaml.dump(merged, file)


# Dataset list from CASCADE's Demo_train.py.
DEMO_DATASETS = (
    'DS03-Cal520-m-S1',
    'DS04-OGB1-zf-pDp',
    'DS05-Cal520-zf-pDp',
    'DS06-GCaMP6f-zf-aDp',
    'DS07-GCaMP6f-zf-dD',
    'DS08-GCaMP6f-zf-OB',
    'DS09-GCaMP6f-m-V1',
    'DS10-GCaMP6f-m-V1-neuropil-corrected',
    'DS11-GCaMP6f-m-V1-neuropil-corrected',
    'DS12-GCaMP6s-m-V1-neuropil-corrected',
    'DS13-GCaMP6s-m-V1-neuropil-corrected',
    'DS14-GCaMP6s-m-V1',
    'DS15-GCaMP6s-m-V1',
    'DS16-GCaMP6s-m-V1',
    'DS17-GCaMP5k-m-V1',
    'DS18-R-CaMP-m-CA3',
    'DS19-R-CaMP-m-S1',
    'DS20-jRCaMP1a-m-V1',
)


# Upstream model definition and data preparation from cascade2p/utils.py.
def define_model(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer):

  """"
  Defines the model using the API of Keras.

  The model consists of 3 convolutional layers ('conv_filter'), 2 downsampling layers
  ('MaxPooling1D') and 1 dense layer ('Dense').

  To modify the architecture of the network, only the define_model() function needs to be modified.

  Example: model = define_model(filter_sizes,filter_numbers,dense_expansion,windowsize,loss_function,optimizer)

  """

  from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Conv1D, Input
  from tensorflow.keras import Model
  from tensorflow.keras.optimizers import Adagrad

  inputs = Input(shape=(windowsize,1))

  conv_filter = Conv1D

  outX = conv_filter(filter_numbers[0], filter_sizes[0], strides=1, activation='relu')(inputs)
  outX = conv_filter(filter_numbers[1], filter_sizes[1], activation='relu')(outX)
  outX = MaxPooling1D(2)(outX)
  outX = conv_filter(filter_numbers[2], filter_sizes[2], activation='relu')(outX)
  outX = MaxPooling1D(2)(outX)

  outX = Dense(dense_expansion, activation='relu')(outX) # 'linear' units work here as well!
  outX = Flatten()(outX)
  predictions = Dense(1,activation='linear')(outX)
  model = Model(inputs=[inputs],outputs=predictions)
  optimizer = Adagrad(learning_rate=0.05)
  model.compile(loss=loss_function, optimizer=optimizer)

  return model


def calculate_noise_levels(neurons_x_time, frame_rate):

    """"
    Computes the noise levels for each neuron of the input matrix 'dF_traces'.

    The noise level is computed as the median absolute dF/F difference
    between two subsequent time points. This is a outlier-robust measurement
    that converges to the simple standard deviation of the dF/F trace for
    uncorrelated and outlier-free dF/F traces.

    Afterwards, the value is divided by the square root of the frame rate
    in order to make it comparable across recordings with different frame rates.


    input: dF_traces (matrix with nb_neurons x time_points)
    output: vector of noise levels for all neurons

    """
    dF_traces = neurons_x_time
    noise_levels = np.nanmedian(np.abs(np.diff(dF_traces, axis=-1)), axis=-1) / np.sqrt(frame_rate)
    return noise_levels * 100


def preprocess_traces(neurons_x_time, before_frac, window_size):

    """
    Transform dF/F data into a format that can be used by the deep network.

    For each time point, a window of the size 'window_size' of the dF/F is extracted.

    input:  dF/F traces (matrix with nb_neurons x time_points)
            window_size (size of the receptive window of the deep network)
            before_frac (positioning of the window around the current time point; 0.5 means center position)
    output: X, a matrix with nb_neurons x time_points x window_size

    """
    dF_traces = neurons_x_time
    start = int(before_frac * window_size)
    end = dF_traces.shape[1] - window_size + start + 1
    # extract a moving window from the calcium trace
    window_indexes = (np.expand_dims(np.arange(window_size), 0) + np.expand_dims(np.arange(dF_traces.shape[1] - window_size + 1), 0).T)
    X = np.full(shape=(dF_traces.shape[0], dF_traces.shape[1], window_size), fill_value=np.nan)
    X[:, start:end, :] = dF_traces[:, window_indexes]
    return X


def calibrated_ground_truth_artificial_noise(ground_truth_folder,noise_level,sampling_rate,replicas,omission_list=[], verbose=3):

    """
    sub_traces_all, sub_traces_events_all, frame_rate = calibrated_ground_truth(ground_truth_folder,noise_level,sampling_rate)

    Inputs:

      Folder with ground truth pixel traces (dF/F) in *.mat files
      Noise level at which the ground truth should be resampled
      >> The noise level is defined as the median different between subsequent samples (i.e., high-frequency noise),
        normalized by the square root of the frame rate
      Temporal sampling rate at which the ground truth should be resampled

    Outputs:

      The extracted subtraces as a matrix ('sub' refers to the subset of ROI pixels used for the respective subtrace)
      The simultaneously recorded spikes, with the same time bins
      The frame rate, usually identical to the input target sampling rate
      >> If the input sampling rate does not differ >5% from the original sampling rate of the ground truth, there will be no resampling

    """

    # Iterate through all ground truth files of the selected dataset
    fileList = sorted(list(set(glob.glob( os.path.join(ground_truth_folder,'*_mini.mat')))))

    # Omit neurons from the training data, if indicated in the omission_list
    # This is useful if the network is tested with a leave-one-out strategy
    for index in sorted(omission_list, reverse=True):
        del fileList[index]

    # Initialize lists which will later contain the resampled ground truth
    sub_traces_all = [None] * len(fileList)
    sub_traces_events_all = [None] * len(fileList)
    events_all = [None] * len(fileList)
    framerate_all = [None] * len(fileList)

    # for loop over all mat files / neurons in this dataset
    for file_index,neuron_file in enumerate(fileList):

      if verbose > 2: print('Resampling neuron '+str(file_index+1)+' from a total of '+str(len(fileList))+' neurons.')

      # Load mat file; the file contains a Matlab struct
      dataset_neuron_all = sio.loadmat(neuron_file)['CAttached'][0]

      # Initialize arrays that will contain the ground truth extracted from this neuron
      sub_traces = None
      sub_traces_events = None
      events_all[file_index] = [None] * 100000
      counter = 0

      # For loop over all trials of this neuron; trial is an uninterrupted
      # ground truth recording of a single neuron
      for i,trial in enumerate(dataset_neuron_all):

          try:
            # Find the relevant elements in the data structure
            # (dF/F traces; spike events; time stamps of fluorescence recording)
            keys = trial[0][0].dtype.descr
            keys_unfolded = list(sum(keys, ()))
    
            traces_index = int(keys_unfolded.index("fluo_mean")/2)
            fluo_time_index = int(keys_unfolded.index("fluo_time")/2)
            events_index = int(keys_unfolded.index("events_AP")/2)
    
            events = trial[0][0][events_index]
            events = events[~np.isnan(events)] # exclude NaN entries for the Theis et al. data sets
            ephys_sampling_rate = 1e4
    
            fluo_times = np.squeeze(trial[0][0][fluo_time_index])
            frame_rate = 1/np.nanmean(np.diff(fluo_times))

            traces_mean = np.squeeze(trial[0][0][traces_index])
            traces_mean = traces_mean[:fluo_times.shape[0]]
    
            traces_mean = traces_mean[~np.isnan(fluo_times)]
            fluo_times = fluo_times[~np.isnan(fluo_times)]
    
            # Compute the baseline noise level for this recording
            base_noise = np.nanmedian(np.abs(np.diff(traces_mean)))*100/np.sqrt(frame_rate)
            # Test how much artificial noise must be added to reach the target noise level
            # THe output of this procedure is 'noise_std'
            test_noise = np.zeros((60,))
            for test_i in np.arange(60):
              noise_trace = np.random.normal(0,test_i/100*np.sqrt(frame_rate), traces_mean.shape)
              test_noise[test_i] = np.nanmedian(np.abs(np.diff(noise_trace+traces_mean)))*100/np.sqrt(frame_rate)
    
            interpolating_function = interp1d(test_noise,np.arange(60), kind='linear')
    
            if noise_level >= test_noise[0]:
    
              noise_std = interpolating_function(noise_level)/100*np.sqrt(frame_rate)
              # Get as many artificial noisified replica traces such that natural noise (which
              # is correlated across replicas is not dominant; this is a heuristic procedure.
              # Limit the maximum number of replicas per ground truth trace to 500.
              nb_subROIs = np.minimum(500,np.ceil( 1.2*(noise_level/base_noise)**2 ))
    
              if not replicas:
    
                nb_subROIs = 1
    
            else:
    
              nb_subROIs = 0
    
            if nb_subROIs >= 1:
    
              # Resampling is not necessary if sampling rates of ground truth and
              # target sampling rate are similar (<5% relative difference)
              if np.abs(sampling_rate - frame_rate)/frame_rate > 0.05:
    
                num_samples = int(round(traces_mean.shape[0]*sampling_rate/frame_rate))
                (traces_mean,temp) = resample(traces_mean,num_samples,np.squeeze(fluo_times),axis=0)
 
                noise_std = noise_std*np.sqrt(sampling_rate/frame_rate)
    
              # Bin the ground truth (spike times) into time bins determined by the resampled calcium recording
              fluo_times_bin_centers = np.linspace(fluo_times[0],fluo_times[-1],len(traces_mean))
              fluo_times_bin_edges = np.append(fluo_times_bin_centers,fluo_times_bin_centers[-1]+1/sampling_rate) - 1/sampling_rate/2
    
              [events_binned,event_bins] = np.histogram(events/ephys_sampling_rate, bins=fluo_times_bin_edges)
    
              # Limit number of replicas for longer recordings
              nb_subROIs = int(np.minimum(8e5,len(dataset_neuron_all)*len(traces_mean)*nb_subROIs)/len(traces_mean))/len(dataset_neuron_all)
    
              # Generate a noisified trace in each iteration of the for-loop
              # Noise is scaled with the square root of the mean fluorescence (fluo_level),
              # corresponding to POisson noise
              for iii in range(int(nb_subROIs)):
    
                fluo_level = np.sqrt(np.abs(traces_mean + 1))
                fluo_level /= np.median(fluo_level)
    
                noise_additional = np.random.normal(0,noise_std*fluo_level, traces_mean.shape)
                sub_traces_single = traces_mean + noise_additional
    
                # 'sub_traces' are sub-sampled replica traces from the same mean trace 'traces_mean';
                # 'sub_traces_events' are the corresponding ground truth action potentials
    
                # If 'sub_traces' exists already, append the subROI-trace; else, generate it
                # The nested if-clause covers edge cases in some ground truth data sets where
                # different trials of the same neuron have variable numbers of time points
                if np.any(sub_traces):
    
                  if sub_traces.shape[0]-len(sub_traces_single) >= 0:
    
                    sub_traces_single = np.append(sub_traces_single, np.zeros(sub_traces.shape[0]-len(sub_traces_single)) + np.nan )
                    events_binned = np.append(events_binned, np.zeros(sub_traces_events.shape[0]-len(events_binned)) + np.nan )
    
                  else:
                    sub_traces = np.append(sub_traces,np.zeros((len(sub_traces_single)-sub_traces.shape[0],sub_traces.shape[1])) + np.nan, axis=0)
                    sub_traces_events = np.append(sub_traces_events,np.zeros((len(events_binned)-sub_traces_events.shape[0],sub_traces_events.shape[1])) + np.nan, axis=0)
    
                  sub_traces = np.append(sub_traces,sub_traces_single.reshape(-1, 1),axis=1)
                  sub_traces_events = np.append(sub_traces_events,events_binned.reshape(-1, 1),axis=1)
    
                else:
    
                  sub_traces = sub_traces_single.reshape(-1, 1)
                  sub_traces_events = events_binned.reshape(-1, 1)
    
                events_all[file_index][counter] = events/ephys_sampling_rate
                counter += 1
            
          except:
              pass

          # Write the subROI-traces for each neuron into a list item of 'sub_traces_all'
          # (calcium) and 'sub_traces_events_all' (spikes)
          sub_traces_all[file_index] = sub_traces
          sub_traces_events_all[file_index] = sub_traces_events

      # Optional output: ground truth spike times; not needed to generate a training data set
      try:
        events_all[file_index] = events_all[file_index][0:sub_traces.shape[1]]
      except:
        pass

      framerate_all[file_index] = frame_rate

    return sub_traces_all, sub_traces_events_all, framerate_all, events_all


def preprocess_groundtruth_artificial_noise_balanced(ground_truth_folders,before_frac,windowsize,after_frac,noise_level,sampling_rate,smoothing,omission_list=[],permute=1,maximum_traces=5000000,verbose=3,replicas=1,causal_kernel=0):

    """
    The calcium traces are extracted, brought to a desired 'noise_level' and
    resampled at the 'sampling_rate' in the function 'calibrated_ground_truth_artificial_noise()',

    The function 'preprocess_groundtruth_artificial_noise()' goes through all
    'ground_truth_folders' and extracts the ground truth in a way that can be
    used to train the deep network.

    As output, this function creates a large matrix 'X' that contains for each
    timepoint of each calcium trace a vector of length 'windowsize' around the timepoint.

    The function also creates a vector Y that contains the corresponding spikes/non-spikes.
    Random permutations ('permute = 1') un-do the original sequence of the timepoints.

    The number of samples is limited to 5 million.

    """

    sub_traces_all = [None]*500
    sub_traces_events_all = [None]*500
    events_all = [None]*500

    neuron_counter = 0
    nbx_datapoints = [None]*500
    dataset_sizes = np.zeros(len(ground_truth_folders),)
    dataset_indices = [None]*500

    # Go through all ground truth data sets and extract re-sampled ground truth
    for dataset_index,training_dataset in enumerate(ground_truth_folders):

      base_folder = os.getcwd()

      # Exception handling ('try') is used here to catch errors that arise if for example
      # some of the datasets contribute zero samples because they do not contain
      # recordings with sufficiently low noise levels (must be lower than 'noise_level')
      # or sufficiently long trials (must be significantly longer than 'window_size').
      try:
          if verbose > 1: print('Preprocessing dataset number', dataset_index)

          sub_traces_allX, sub_traces_events_allX, frame_rate, events_allX = calibrated_ground_truth_artificial_noise(ground_truth_folders[dataset_index],noise_level,sampling_rate,replicas,omission_list, verbose)

          datapoint_counter = 0
          for k in range(len(sub_traces_allX)):
            try:
               datapoint_counter += sub_traces_allX[k].shape[1]*sub_traces_allX[k].shape[0]
            except:
              if verbose > 2: print('No things for k={}'.format(k))

          dataset_sizes[dataset_index] = datapoint_counter

          nbx_datapoints[neuron_counter:neuron_counter+len(sub_traces_allX)] = datapoint_counter*np.ones(len(sub_traces_allX),)
          sub_traces_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = sub_traces_allX
          sub_traces_events_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = sub_traces_events_allX
          events_all[neuron_counter:neuron_counter+len(sub_traces_allX)] = events_allX
          dataset_indices[neuron_counter:neuron_counter+len(sub_traces_allX)] = dataset_index*np.ones(len(sub_traces_allX),)

          neuron_counter += len(sub_traces_allX)

      except:
           sub_traces_allX = None
           dataset_sizes[dataset_index] = np.nan
      os.chdir(base_folder)

    mininum_traces = 15e6/len(ground_truth_folders)

    # Reduce the number of data points for relatively large data sets to avoid bias
    reduction_factors = dataset_sizes/mininum_traces

    if np.nanmax(reduction_factors) > 1:
      oversampling = 1
    else:
      oversampling = 0

    if verbose>1: print('Reducing ground truth by a factor of approximately '+str(int(3*np.nanmean(reduction_factors)))+'.')

    nbx_datapoints = nbx_datapoints[:neuron_counter]
    sub_traces_all = sub_traces_all[:neuron_counter]
    sub_traces_events_all = sub_traces_events_all[:neuron_counter]
    events_all = events_all[:neuron_counter]
    dataset_indices = dataset_indices[:neuron_counter]

    if verbose>1: print('Number of neurons in the ground truth: '+str(len(sub_traces_events_all)))

    before = int(np.round(before_frac*windowsize))
    after = int(np.round(after_frac*windowsize))
    

    if causal_kernel:

      xx = np.arange(0,199)/sampling_rate
      yy = invgauss.pdf(xx,smoothing/sampling_rate*2.0,101/sampling_rate,1)
      ix = np.argmax(yy)
      yy = np.roll(yy,int((99-ix)/1.5))
      causal_smoothing_kernel = yy/np.nansum(yy)



    # Only reserve enough space for the recordings in this run.
    capacity = 0
    for neuron_ix, traces in enumerate(sub_traces_all):
        if traces is None:
            continue
        windows_per_trace = np.maximum(
            0, np.count_nonzero(~np.isnan(traces), axis=0) - windowsize
        )
        if oversampling:
            reduction = max(reduction_factors[int(dataset_indices[neuron_ix])], 1)
            windows_per_trace = np.floor(windows_per_trace / reduction)
        capacity += int(windows_per_trace.sum())
    X = np.zeros((capacity, windowsize))
    Y = np.zeros(capacity)

    # For-loop to generate the outputs 'X' and 'Y'
    counter = 0
    for neuron_ix,(sub_traces,sub_traces_events) in enumerate(zip(sub_traces_all,sub_traces_events_all)):

      if sub_traces is not None:

        for trace_index in range(sub_traces.shape[1]):

          single_trace = sub_traces[:,trace_index]
          single_spikes = sub_traces_events[:,trace_index]

          # Optional: Generates ground truth with causally smoothed kernel (see paper for details)
          if causal_kernel:

            single_spikes = convolve(single_spikes.astype(float),causal_smoothing_kernel,mode='same')

          else:

            single_spikes = gaussian_filter(single_spikes.astype(float), sigma=smoothing)

          recording_length = np.sum(~np.isnan(single_trace))

          datapoints_used = np.minimum(len(single_spikes)-windowsize,recording_length-windowsize)

          if datapoints_used <= 0:
            continue

          # Discarding (randomly chosen) samples to reduce ground truth dataset size
          if oversampling:

            datapoints_used_rand = np.random.permutation(datapoints_used)
            reduce_samples = reduction_factors[int(dataset_indices[neuron_ix])]
            datapoints_used_rand = datapoints_used_rand[0:int(len(datapoints_used_rand)/( max(reduce_samples,1)  ))]

          else:

            datapoints_used_rand = np.arange(datapoints_used)

          for time_points in datapoints_used_rand:

            Y[counter,] = single_spikes[time_points+before]
            X[counter,:,] = single_trace[time_points:(time_points+before+after)]
            counter += 1

    Y = np.expand_dims(Y[:counter],axis=1)
    X = np.expand_dims(X[:counter,:],axis=2)

    # Permute the ordering of the output for improved gradient descent during learning
    if permute == 1:

      # Select the same samples without making a full shuffled copy.
      p = np.random.permutation(len(X))[:5000000]
      X = X[p,:,:]
      Y = Y[p,:]

    X = X[np.where(~np.isnan(Y))[0],:,:]
    Y = Y[np.where(~np.isnan(Y))[0],:]


    os.chdir(base_folder)

    if verbose > 1: print('Shape of training dataset X: {}    Y: {}'.format(X.shape, Y.shape))
    return X,Y




# Upstream training and inference from cascade2p/cascade.py.
def train_model(
    model_name, model_folder="Pretrained_models", ground_truth_folder="Ground_truth"
):

    """Train neural network with parameters specified in the config.yaml file in the model folder

    In this function, a model is configured (defined in the input 'model_name': frame rate, noise levels, ground truth datasets, etc.).
    The ground truth is resampled (function 'preprocess_groundtruth_artificial_noise_balanced', defined in "utils.py").
    The network architecture is defined (function 'define_model', defined in "utils.py").
    The thereby defined model is trained with the resampled ground truth data.
    The trained model with its weight and configuration details is saved to disk.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
        This name has to correspond to the folder with the config.yaml file which defines the model parameters

    model_folder: str
        Absolute or relative path, which defines the location of the specified model_name folder
        Default value 'Pretrained_models' assumes a current working directory in the Cascade folder

    ground_truth_folder : str
        Absolute or relative path, which defines the location of the ground truth datasets
        Default value 'Ground_truth'  assumes a current working directory in the Cascade folder

    Returns
    --------
    None
        All results are saved in the folder model_name as .h5 files containing the trained model

    """
    import tensorflow.keras
    from tensorflow.keras.optimizers import Adagrad

    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join(model_path, "config.yaml")

    # check if configuration file can be found
    if not os.path.isfile(cfg_file):
        m = (
            'The configuration file "config.yaml" can not be found at the location "{}".\n'.format(
                os.path.abspath(cfg_file)
            )
            + 'You have provided the model "{}" at the absolute or relative path "{}".\n'.format(
                model_name, model_folder
            )
            + 'Please check if there is a folder for model "{}" at the location "{}".'.format(
                model_name, os.path.abspath(model_folder)
            )
        )
        print(m)
        raise Exception(m)

    # load cfg dictionary from config.yaml file
    cfg = read_config(cfg_file)
    verbose = cfg["verbose"]

    if verbose:
        print(
            "Used configuration for model fitting (file {}):\n".format(
                os.path.abspath(cfg_file)
            )
        )
        for key in cfg:
            print("{}:\t{}".format(key, cfg[key]))

        print("\n\nModels will be saved into this folder:", os.path.abspath(model_path))

    # add base folder to selected training datasets
    training_folders = [
        os.path.join(ground_truth_folder, ds) for ds in cfg["training_datasets"]
    ]

    # check if the training datasets can be found
    missing = False
    for folder in training_folders:
        if not os.path.isdir(folder):
            print(
                'The folder "{}" could not be found at the specified location "{}"'.format(
                    folder, os.path.abspath(folder)
                )
            )
            missing = True
    if missing:
        m = (
            'At least one training dataset could not be located.\nThis could mean that the given path "{}" '.format(
                ground_truth_folder
            )
            + "does not specify the correct location or that e.g. a training dataset referenced in the config.yaml file "
            + "contained a typo."
        )
        print(m)
        raise Exception(m)

    start = time.time()
    # Update model fitting status
    cfg["training_finished"] = "Running"
    write_config(cfg, os.path.join(model_path, "config.yaml"))

    nr_model_fits = len(cfg["noise_levels"]) * cfg["ensemble_size"]
    print("Fitting a total of {} models:".format(nr_model_fits))
  
    curr_model_nr = 0

    print(training_folders[0])

    for noise_level in cfg["noise_levels"]:
        for ensemble in range(cfg["ensemble_size"]):
            # train 'ensemble_size' (e.g. 5) models for each noise level

            curr_model_nr += 1
            print(
                "\nFitting model {} with noise level {} (total {} out of {}).".format(
                    ensemble + 1, noise_level, curr_model_nr, nr_model_fits
                )
            )

            if cfg["sampling_rate"] > 30:

                windowsize_suggestion = int(np.power(cfg["sampling_rate"] / 30, 0.25) * 64)

                print(
                    "Window size should be enlarged to "
                    + str(windowsize_suggestion)
                    + " time points (if not done already) due to the high calcium imaging sampling rate ("
                    + str(cfg["sampling_rate"])
                    + ")."
                )

            # preprocess dataset to get uniform dataset for training
            X, Y = preprocess_groundtruth_artificial_noise_balanced(
                ground_truth_folders=training_folders,
                before_frac=cfg["before_frac"],
                windowsize=cfg["windowsize"],
                after_frac=1 - cfg["before_frac"],
                noise_level=noise_level,
                sampling_rate=cfg["sampling_rate"],
                smoothing=cfg["smoothing"] * cfg["sampling_rate"],
                omission_list=[],
                permute=1,
                verbose=cfg["verbose"],
                replicas=1,
                causal_kernel=cfg["causal_kernel"],
            )

            if len(X) == 0:
                raise ValueError(
                    f"No training windows for noise level {noise_level}. "
                    "Check the recordings or choose a noise level at or above "
                    "their native noise."
                )

            model = define_model(
                filter_sizes=cfg["filter_sizes"],
                filter_numbers=cfg["filter_numbers"],
                dense_expansion=cfg["dense_expansion"],
                windowsize=cfg["windowsize"],
                loss_function=cfg["loss_function"],
                optimizer=cfg["optimizer"],
            )

            optimizer = Adagrad(learning_rate=0.05)
            model.compile(loss=cfg["loss_function"], optimizer=optimizer)

            cfg["nr_of_epochs"] = np.minimum(
                cfg["nr_of_epochs"], int(10 * np.floor(5e6 / len(X)))
            )

            model.fit(
                X,
                Y,
                batch_size=cfg["batch_size"],
                epochs=cfg["nr_of_epochs"],
                verbose=cfg["verbose"],
            )

            # save model
            file_name = "Model_NoiseLevel_{}_Ensemble_{}.h5".format(
                int(noise_level), ensemble
            )
            model.save(os.path.join(model_path, file_name))
            print("Saved model:", file_name)
            del model, X, Y
            tf.keras.backend.clear_session()
            gc.collect()

    # Update model fitting status
    # cfg['training_finished'] = 'Yes'
    # write_config(cfg, os.path.join( model_path, 'config.yaml' ))

    print("\n\nDone!")
    print("Runtime: {:.0f} min".format((time.time() - start) / 60))


def predict(
    model_name, traces, model_folder="Pretrained_models", threshold=0, padding=np.nan, trace_noise_levels=None, verbosity=1
):

    """Use a specific trained neural network ('model_name') to predict spiking activity for calcium traces ('traces')

    In this function, a already trained model (generated by 'train_model' or downloaded) is loaded.
    The model (frame rate, noise levels, ground truth datasets) should be chosen
      to match the properties of the calcium recordings in 'traces'.
    An ensemble of 5 models is loaded for each noise level.
    These models are used to predict spiking activity of neurons from 'traces' with the same noise levels.
    The predictions are made in the line with 'model.predict()'.
    The predictions are returned as a matrix 'Y_predict'.


    Parameters
    ------------
    model_name : str
        Name of the model, e.g. 'Universal_30Hz_smoothing100ms'
        This name has to correspond to the folder in which the config.yaml and .h5 files are stored which define
        the trained model

    traces : 2d numpy array (neurons x nr_timepoints)
        Df/f traces with recorded fluorescence (as fractions, not in percents) on which the spiking activity will
        be predicted. Required shape: (neurons x nr_timepoints)

    model_folder: str
        Absolute or relative path, which defines the location of the specified model_name folder
        Default value 'Pretrained_models' assumes a current working directory in the Cascade folder

    threshold : int or boolean
        Allowed values: 0, 1 or False
            0: All negative values are set to 0
            1 or True: Threshold signal to set every signal which is smaller than the expected signal size
                       of an action potential to zero (with dilated mask)
            False: No thresholding. The result can contain negative values as well

    padding : 0 or np.nan
        Value which is inserted for datapoints, where no prediction can be made (because of window around timepoint of prediction)
        Default value: np.nan, another recommended value would be 0 which circumvents some problems with following analysis.

    trace_noise_levels: float (neurons x 1)
        Noise levels of the traces. If not provided, the noise levels are calculated from the traces.

    vebosity : 0 or 1
        If set to 0, the output of predict() during inference in the console is suppressed
        Default value: 1

    Returns
    --------
    predicted_activity: 2d numpy array (neurons x nr_timepoints)
        Spiking activity as predicted by the model. The shape is the same as 'traces'
        This array can contain NaNs if the value 'padding' was np.nan as input argument

    """
    import tensorflow.keras
    from tensorflow.keras.models import load_model

    # reshape matrix of traces if only a single neuron's activity is provided as input to the inference
    if len(traces.shape) == 1:
        traces = np.expand_dims(traces,0)

  
    model_path = os.path.join(model_folder, model_name)
    cfg_file = os.path.join(model_path, "config.yaml")

    # check if configuration file can be found
    if not os.path.isfile(cfg_file):
        m = (
            'The configuration file "config.yaml" can not be found at the location "{}".\n'.format(
                os.path.abspath(cfg_file)
            )
            + 'You have provided the model "{}" at the absolute or relative path "{}".\n'.format(
                model_name, model_folder
            )
            + 'Please check if there is a folder for model "{}" at the location "{}".'.format(
                model_name, os.path.abspath(model_folder)
            )
        )
        print(m)
        raise Exception(m)

    # Load config file
    cfg = read_config(cfg_file)

    # extract values from config file into variables
    verbose = cfg["verbose"]
    if verbosity == 0:
      verbose = 0
    training_data = cfg["training_datasets"]
    ensemble_size = cfg["ensemble_size"]
    batch_size = cfg["batch_size"]
    sampling_rate = cfg["sampling_rate"]
    before_frac = cfg["before_frac"]
    window_size = cfg["windowsize"]
    noise_levels_model = cfg["noise_levels"]
    smoothing = cfg["smoothing"]
    causal_kernel = cfg["causal_kernel"]

    model_description = (
        "\n \nThe selected model was trained on "
        + str(len(training_data))
        + " datasets, with "
        + str(ensemble_size)
        + " ensembles for each noise level, at a sampling rate of "
        + str(sampling_rate)
        + "Hz,"
    )
    if causal_kernel:
        model_description += (
            " with a resampled ground truth that was smoothed with a causal kernel"
        )
    else:
        model_description += (
            " with a resampled ground truth that was smoothed with a Gaussian kernel"
        )
    model_description += (
        " of a standard deviation of "
        + str(int(1000 * smoothing))
        + " milliseconds. \n \n"
    )
    if verbose:
        print(model_description)

    if verbose:
        print("Loaded model was trained at frame rate {} Hz".format(sampling_rate))
    if verbose:
        print(
            "Given argument traces contains {} neurons and {} frames.".format(
                traces.shape[0], traces.shape[1]
            )
        )

    if trace_noise_levels is None:
        # calculate noise levels for each trace
        trace_noise_levels = calculate_noise_levels(traces, sampling_rate)

    if verbose:
        print(
            "Noise levels (mean, std; in standard units): "
            + str(int(np.nanmean(trace_noise_levels * 100)) / 100)
            + ", "
            + str(int(np.nanstd(trace_noise_levels * 100)) / 100)
        )

    # Get model paths as dictionary (key: noise_level) with lists of model
    # paths for the different ensembles
    model_dict = get_model_paths(model_path)  # function defined below
    if verbose > 2:
        print("Loaded models:", str(model_dict))

    # XX has shape: (neurons, timepoints, windowsize)
    XX = preprocess_traces(
        traces, before_frac=before_frac, window_size=window_size
    )
    Y_predict = np.zeros((XX.shape[0], XX.shape[1]))

    # Compute difference of noise levels between each neuron and each model; find the best fit
    differences = np.array(trace_noise_levels)[:,None] - np.array(noise_levels_model)[None,:]
    relative_differences = np.min(differences,axis=1)
    if np.mean(relative_differences) > 2:
        print("WARNING: The available models cannot match the experimentally obtained noise levels (difference: ", str(np.mean(relative_differences)),"). Please check that the computation of dF/F is performed correctly. Otherwise, please reach out and ask for pretrained models with higher noise level models (see: https://github.com/HelmchenLabSoftware/Cascade/issues/61).")
    best_model_for_each_neuron = np.argmin(np.abs(differences),axis=1)
    
    # Use for each noise level the matching model
    for i, model_noise in enumerate(noise_levels_model):

        if verbose:
            print("\nPredictions for noise level {}:".format(model_noise))

        # select neurons which have this noise level:
        neuron_idx = np.where(best_model_for_each_neuron == i)[0]
        
        if len(neuron_idx) == 0:  # no neurons were selected
            if verbose:
                print("\tNo neurons for this noise level")
            continue  # jump to next noise level

        # load keras models for the given noise level
        models = list()
        for model_path in model_dict[model_noise]:
            models.append(load_model(model_path, compile=False))

        # select neurons and merge neurons and timepoints into one dimension
        XX_sel = XX[neuron_idx, :, :]

        XX_sel = np.reshape(
            XX_sel, (XX_sel.shape[0] * XX_sel.shape[1], XX_sel.shape[2])
        )
        XX_sel = np.expand_dims(
            XX_sel, axis=2
        )  # add empty third dimension to match training shape

        for j, model in enumerate(models):
            if verbose:
                print("\t... ensemble", j)

            prediction_flat = model.predict(XX_sel, batch_size, verbose=verbose)
            prediction = np.reshape(prediction_flat, (len(neuron_idx), XX.shape[1]))

            Y_predict[neuron_idx, :] += prediction / len(models)  # average predictions

        # remove models from memory
        tensorflow.keras.backend.clear_session()

    if threshold is False:  # only if 'False' is passed as argument
        if verbose:
            print(
                "Skipping the thresholding. There can be negative values in the result."
            )

    elif threshold == 1:  # (1 or True)
        # Cut off noise floor (lower than 1/e of a single action potential)

        from scipy.ndimage import binary_dilation

        # find out empirically  how large a single AP is (depends on frame rate and smoothing)
        single_spike = np.zeros(
            1001,
        )
        single_spike[501] = 1
        single_spike_smoothed = gaussian_filter(
            single_spike.astype(float), sigma=smoothing * sampling_rate
        )
        threshold_value = np.max(single_spike_smoothed) / np.exp(1)

        # Set everything below threshold to zero.
        # Use binary dilation to avoid clipping of true events.
        for neuron in range(Y_predict.shape[0]):
            # ignore warning because of nan's in Y_predict in comparison with value
            with np.errstate(invalid="ignore"):
                activity_mask = Y_predict[neuron, :] > threshold_value
            activity_mask = binary_dilation(
                activity_mask, iterations=int(smoothing * sampling_rate)
            )

            Y_predict[neuron, ~activity_mask] = 0

            Y_predict[
                Y_predict < 0
            ] = 0  # set possible negative values in dilated mask to 0

    elif threshold == 0:
        # ignore warning because of nan's in Y_predict in comparison with value
        with np.errstate(invalid="ignore"):
            Y_predict[Y_predict < 0] = 0

    else:
        raise Exception(
            'Invalid value of threshold "{}". Only 0, 1 (or True) or False allowed'.format(
                threshold
            )
        )

    # NaN or 0 for first and last datapoints, for which no predictions can be made
    Y_predict[:, 0 : int(before_frac * window_size)] = padding
    Y_predict[:, -int((1 - before_frac) * window_size) :] = padding

    print("Spike rate inference done.")

    return Y_predict


def get_model_paths(model_path):

    """Find all models in the model folder and return as dictionary
    ( Helper function called by predict() )

    Returns
    -------
    model_dict : dict
        Dictionary with noise_level (int) as keys and entries are lists of model paths

    """
    import glob, re

    all_models = glob.glob(os.path.join(model_path, "*.h5"))
    all_models = sorted(all_models)  # sort

    # Exception in case no model was found to catch this mistake where it happened
    if len(all_models) == 0:
        m = 'No models (*.h5 files) were found in the specified folder "{}".'.format(
            os.path.abspath(model_path)
        )
        raise Exception(m)

    # dictionary with key for noise level, entries are lists of models
    model_dict = dict()

    for model_path in all_models:
        try:
            noise_level = int(re.findall(r"_NoiseLevel_(\d+)", model_path)[0])
        except:
            print("Error while processing the file with name: ", model_path)
            raise

        # add model path to the model dictionary
        if noise_level not in model_dict:
            model_dict[noise_level] = list()
        model_dict[noise_level].append(model_path)

    return model_dict


# Standalone command-line interface.
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train CASCADE directly from ground-truth files")
    train.add_argument("--data-root", type=Path, default=Path("data/raw"))
    train.add_argument("--datasets", nargs="+", default=DEMO_DATASETS,
                       help="dataset folders inside --data-root (default: Demo_train.py list)")
    train.add_argument("--model-root", type=Path, default=Path("models"))
    train.add_argument("--model-name", default="Cascade_baseline")
    train.add_argument("--sampling-rate", type=float, default=30)
    train.add_argument("--smoothing-ms", type=float, default=25)
    train.add_argument("--noise-levels", nargs="+", type=int, default=list(range(2, 15)))
    train.add_argument("--ensemble-size", type=int, default=5)
    train.add_argument("--epochs", type=int, default=20)

    inference = commands.add_parser("predict", help="predict spike activity for one trial in a .mat file")
    inference.add_argument("model_dir", type=Path)
    inference.add_argument("mat_file", type=Path)
    inference.add_argument("--trial", type=int, default=0, help="trial index, starting at 0")
    inference.add_argument("--out", type=Path, default=Path("cascade_predictions.npz"))
    args = parser.parse_args()

    if args.command == "train":
        if args.ensemble_size < 1 or args.epochs < 1:
            train.error("--ensemble-size and --epochs must be positive")
        if not np.isfinite(args.sampling_rate) or args.sampling_rate <= 0:
            train.error("--sampling-rate must be positive")
        if not np.isfinite(args.smoothing_ms) or args.smoothing_ms < 0:
            train.error("--smoothing-ms must be nonnegative")
        if min(args.noise_levels) < 1 or len(set(args.noise_levels)) != len(args.noise_levels):
            train.error("--noise-levels must be distinct positive integers")
        if (
            not args.model_name
            or args.model_name in {".", ".."}
            or Path(args.model_name).name != args.model_name
        ):
            train.error("--model-name must be a single folder name")

        for dataset in args.datasets:
            folder = args.data_root / dataset
            if not any(folder.glob("*_mini.mat")):
                train.error(f"No *_mini.mat files in {folder}")

        model_dir = args.model_root / args.model_name
        try:
            model_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            train.error(f"{model_dir} already exists. Choose a new --model-name.")

        cfg = dict(
            model_name=args.model_name,
            sampling_rate=args.sampling_rate,
            training_datasets=list(args.datasets),
            noise_levels=args.noise_levels,
            smoothing=args.smoothing_ms / 1000,
            causal_kernel=0,
            ensemble_size=args.ensemble_size,
            nr_of_epochs=args.epochs,
        )
        write_config(cfg, model_dir / "config.yaml")
        train_model(
            args.model_name,
            model_folder=args.model_root,
            ground_truth_folder=args.data_root,
        )
        return

    model_dir = args.model_dir.resolve()
    cfg = read_config(model_dir / "config.yaml")
    import loader
    recordings = loader.load_recordings_from_file(args.mat_file)
    if not 0 <= args.trial < len(recordings):
        inference.error(f"Choose a --trial from 0 to {len(recordings) - 1}.")

    recording = recordings[args.trial]
    trace = recording["dff"]
    times = recording["t"]
    recorded_rate = recording["frame_rate"]
    model_rate = cfg["sampling_rate"]
    if not np.isfinite(recorded_rate) or recorded_rate <= 0:
        inference.error("The recording needs a valid frame rate.")
    if abs(model_rate - recorded_rate) / recorded_rate > 0.05:
        n_frames = round(len(trace) * model_rate / recorded_rate)
        if n_frames <= cfg["windowsize"]:
            inference.error("The recording is too short at the model frame rate.")
        trace, times = resample(trace, n_frames, t=times)
    elif len(trace) <= cfg["windowsize"]:
        inference.error("The recording must be longer than one model window.")

    output = args.out.with_suffix(".npz")
    if output.exists():
        inference.error(f"{output} already exists. Choose another --out.")

    prediction = predict(model_dir.name, trace, model_folder=model_dir.parent)[0]
    with output.open("xb") as file:
        np.savez(
            file,
            time=times,
            dff=trace,
            predicted_spike_activity=prediction,
            true_spike_times=recording["spikes"],
        )
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()