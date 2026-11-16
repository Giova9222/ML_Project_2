"""Loading and plotting of two-photon calcium imaging sessions.

Each session is a MATLAB file holding a top-level struct (e.g. ``CAttached``)
with at least:

    fluo_time : time axis of the fluorescence trace [s]
    fluo_mean : dF/F trace
    events_AP : action potential times, in samples of the ephys recording

Optional fields seen across the Cascade ground-truth datasets include
``cell_type``, ``fluo_raw``, ``neuropil``, ``stim``, ``fluo_ideal``, ``ROI`` and
various movie/template arrays.  Only the three required fields are relied upon;
everything else is exposed through :attr:`Session.fields` and read on demand.

A few files store an *array* of structs rather than a single one — either
several cells recorded together or several trials of one cell.  Those are
exposed as one :class:`Session` per member, named ``<file>#<index>``.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# events_AP is stored in samples of the (10 kHz) ephys recording.
AP_SAMPLE_RATE = 10_000.0

# Default duration [s] of the zoomed-in panel.
ZOOM_DURATION = 20.0

# Fields every Cascade ground-truth file is expected to provide.
REQUIRED_FIELDS = ("fluo_time", "fluo_mean", "events_AP")


def read_mat_structs(path):
    """Return the top-level struct(s) stored in a Cascade ``.mat`` file.

    Most files hold a single struct; some hold an array of them.  Either way a
    flat list of structs is returned.
    """
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    values = [v for k, v in mat.items() if not k.startswith("__")]
    if not values:
        raise ValueError(f"{path}: no top-level struct found")

    value = values[0]
    if hasattr(value, "_fieldnames"):
        return [value]
    return list(np.asarray(value, dtype=object).ravel())


def read_mat_struct(path):
    """Return the first top-level struct in a Cascade ``.mat`` file."""
    return read_mat_structs(path)[0]


@dataclass
class Session:
    """A single recorded cell: dF/F trace and its action potentials."""

    name: str
    time: np.ndarray      # [s]
    dff: np.ndarray       # dF/F
    ap_times: np.ndarray  # [s]
    cell_type: str = ""
    dataset: str = ""     # parent folder, e.g. "DS28-XCaMPgf-m-V1"
    source: str = ""      # path the session was loaded from
    member: int = 0       # index within a multi-struct file
    fields: list[str] = field(default_factory=list)  # fields present in the file


def _session_from_struct(struct, path, member=0, ap_sample_rate=AP_SAMPLE_RATE):
    missing = [f for f in REQUIRED_FIELDS if not hasattr(struct, f)]
    if missing:
        raise ValueError(f"{path.name}: missing required field(s) {missing}")

    time = np.asarray(struct.fluo_time, dtype=float).ravel()
    dff = np.asarray(struct.fluo_mean, dtype=float).ravel()

    # A few traces are padded with NaN at the edges; trim those so the time
    # axis stays finite.  Interior NaNs are left alone (matplotlib breaks the
    # line there, which is the honest rendering).
    finite = np.isfinite(time)
    if finite.sum() < 2:
        raise ValueError(f"{path.name}: fewer than 2 finite fluo_time samples")
    lo, hi = np.argmax(finite), len(finite) - np.argmax(finite[::-1])
    time, dff = time[lo:hi], dff[lo:hi]

    name = path.stem if member == 0 else f"{path.stem}#{member}"
    return Session(
        name=name,
        time=time,
        dff=dff,
        ap_times=np.asarray(struct.events_AP, dtype=float).ravel() / ap_sample_rate,
        cell_type=str(getattr(struct, "cell_type", "")),
        dataset=path.parent.name,
        source=str(path),
        member=member,
        fields=list(struct._fieldnames),
    )


def load_sessions(path, ap_sample_rate=AP_SAMPLE_RATE):
    """Load every session stored in one Cascade ground-truth file.

    Returns a list with one :class:`Session` per struct in the file.
    """
    path = Path(path)
    sessions = []
    for i, struct in enumerate(read_mat_structs(path)):
        try:
            sessions.append(_session_from_struct(struct, path, i, ap_sample_rate))
        except ValueError:
            if i == 0:
                raise  # a single-struct file that is genuinely unusable
    if not sessions:
        raise ValueError(f"{path.name}: no usable session found")
    return sessions


def load_session(path, ap_sample_rate=AP_SAMPLE_RATE):
    """Load the first session from a Cascade ground-truth file."""
    return load_sessions(path, ap_sample_rate)[0]


def plot_session(session, save_path, zoom_duration=ZOOM_DURATION):
    """Plot dF/F with the action potentials overlaid and save as PNG.

    The top panel shows the whole session; the bottom panel zooms into the
    first ``zoom_duration`` seconds so individual APs and transients are
    resolvable.  Works for any Cascade session, including ones whose time axis
    starts below zero or that carry no action potentials.
    """
    t = session.time
    t0, t1 = float(t[0]), float(t[-1])
    zoom_end = min(t0 + zoom_duration, t1)

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 6), sharey=True)

    for ax, title in ((ax_full, "full session"),
                      (ax_zoom, f"zoom: {t0:g}–{zoom_end:g} s")):
        ax.plot(t, session.dff, lw=0.8, color="tab:blue", zorder=2)
        if session.ap_times.size:
            ax.eventplot(
                session.ap_times, colors="tab:red", lineoffsets=0.0,
                linelengths=1.0, linewidths=0.8, zorder=1,
            )
        ax.set_ylabel("dF/F")
        ax.set_title(f"{session.name}  ({session.cell_type or 'unknown'})  —  {title}")

    ax_full.set_xlim(t0, t1)
    ax_zoom.set_xlim(t0, zoom_end)
    ax_zoom.set_xlabel("time (s)")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
