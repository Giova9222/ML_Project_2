"""Download, index and load the Cascade ground-truth dataset.

Small, single-purpose modules:

``repo``
    :class:`~cascade_data.repo.CascadeRepository` — cached remote listing + download.
``index``
    :class:`~cascade_data.index.DatasetIndex` — local catalogue of downloaded files.
``two_photon``
    :class:`~cascade_data.two_photon.Session` — loader/plotter for ``.mat`` sessions.
``noise``
    :func:`~cascade_data.noise.noise_level` — Cascade's noise-level measure.
``cli``
    Command-line front-end tying the download and index helpers together.

Typical use::

    from cascade_data import CascadeRepository, DatasetIndex

    repo = CascadeRepository(cache_dir="data/.cache")
    repo.download("DS28-XCaMPgf-m-V1", "data/raw", limit=1)   # smallest file

    index = DatasetIndex("data")
    index.build()
    for entry in index.entries():
        print(entry.dataset, entry.name, entry.n_aps)
"""

from cascade_data.index import DatasetIndex, IndexedFile, probe_mat
from cascade_data.noise import (
    frame_rate_from_time,
    noise_level,
    noise_levels,
    plot_noise_distribution,
    plot_noise_levels,
)
from cascade_data.pretrained import DEFAULT_MODELS, PretrainedModels
from cascade_data.repo import CascadeRepository, RemoteFile, human_bytes
from cascade_data.two_photon import (
    AP_SAMPLE_RATE,
    Session,
    load_session,
    load_sessions,
    plot_session,
    read_mat_struct,
    read_mat_structs,
)

__all__ = [
    "CascadeRepository",
    "RemoteFile",
    "human_bytes",
    "PretrainedModels",
    "DEFAULT_MODELS",
    "DatasetIndex",
    "IndexedFile",
    "probe_mat",
    "Session",
    "AP_SAMPLE_RATE",
    "load_session",
    "load_sessions",
    "plot_session",
    "read_mat_struct",
    "read_mat_structs",
    "noise_level",
    "noise_levels",
    "frame_rate_from_time",
    "plot_noise_distribution",
    "plot_noise_levels",
]
