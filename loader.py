import numpy as np
import scipy.io as sio

# Adapted from HelmchenLabSoftware/Cascade, cascade2p/utils.py
# https://github.com/HelmchenLabSoftware/Cascade
# Licensed under GPL-3.0

def load_recordings_from_file(file_path):
    """ Load all recordings from a given file into list of dictionaries

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the recorded ground truth file (*.mat)

    Returns
    --------
    recording_list: list of dictionaries
        List of recordings for given file
        Each recording is a dictionary with 't', 'dff', 'spikes', 'frame_rate' as keys
        'spikes' are the spike times in seconds, aligned to the time 't' and fluorescence 'dff'
    """

    data = sio.loadmat(file_path)['CAttached'][0]

    recording_list = list()

    for i,trial in enumerate(data):

        keys = trial[0][0].dtype.descr
        keys_unfolded = list(sum(keys, ()))

        traces_index = int(keys_unfolded.index("fluo_mean")/2)
        fluo_time_index = int(keys_unfolded.index("fluo_time")/2)
        events_index = int(keys_unfolded.index("events_AP")/2)

        # spikes
        events = trial[0][0][events_index]
        events = events[~np.isnan(events)] # exclude NaN entries for the Theis et al. data sets
        ephys_sampling_rate = 1e4
        event_time = events/ephys_sampling_rate

        # fluorescence
        fluo_times = np.squeeze(trial[0][0][fluo_time_index])
        fluo_times = fluo_times[~np.isnan(fluo_times)]
        frame_rate = 1/np.mean(np.diff(fluo_times))

        traces_mean = np.squeeze(trial[0][0][traces_index])
        traces_mean = traces_mean[:fluo_times.shape[0]]
        traces_mean = traces_mean[~np.isnan(fluo_times)]


        recording_list.append( dict(dff=traces_mean, t=fluo_times, spikes=event_time, frame_rate=frame_rate))

    return recording_list