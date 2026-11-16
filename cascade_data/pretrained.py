"""Download the pretrained Cascade models from the Cascade repository.

The Cascade repository ships a ``Pretrained_models`` folder with one
sub-folder per trained network (``Global_EXC_30Hz_smoothing50ms``,
``GCaMP6f_mouse_30Hz_smoothing200ms``, ...).  Each folder holds the network
weights as Keras ``*.h5`` files (one per noise level and ensemble member) plus
a ``config.yaml`` describing the training data, sampling rate and architecture.

The weights themselves are **not** stored in the git repository: the folder
only carries a ``config.yaml`` and an ``available_models.yaml`` index that maps
each model name to a download link (a ``.zip`` hosted on Switch Drive).  This
module reads that index and downloads/extracts the chosen models.

    from cascade_data import PretrainedModels

    repo = PretrainedModels()
    repo.models()                       # -> ["GC8_EXC_...", ...]
    repo.download("OGB_zf_pDp_7.5Hz_smoothing200ms", "models/pretrained")

Downloads are idempotent: a model whose folder already holds ``*.h5`` files is
skipped unless ``overwrite`` is set.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import yaml

from cascade_data.repo import BRANCH, REPO, _USER_AGENT

MODELS_PATH = "Pretrained_models"
INFO_FILE = f"{MODELS_PATH}/available_models.yaml"

# Curated subset whose training data overlaps the datasets used here (mouse
# cortical GCaMP6f/8f/8m/8s, zebrafish OGB, spinal cord, plus a universal
# model).  ``download`` accepts any name from ``available_models.yaml``, so
# this is only the default for the CLI / benchmark.
DEFAULT_MODELS = (
    "Global_EXC_30Hz_smoothing50ms",
    "GCaMP6f_mouse_30Hz_smoothing200ms",
    "GC8f_EXC_30Hz_smoothing50ms_high_noise",
    "GC8m_EXC_30Hz_smoothing50ms_high_noise",
    "GC8s_EXC_30Hz_smoothing50ms_high_noise",
    "OGB_zf_pDp_7.5Hz_smoothing200ms",
    "Spinal_cord_excitatory_30Hz_smoothing50ms",
    "Spinal_cord_inhibitory_30Hz_smoothing50ms",
)


def _read_url(url, timeout=120):
    """Fetch ``url`` and return the raw bytes (with a browser-like UA)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class PretrainedModels:
    """Read-only view of the Cascade ``Pretrained_models`` index + downloader."""

    def __init__(self, repo=REPO, branch=BRANCH, info_file=INFO_FILE,
                 timeout=120):
        self.repo = repo
        self.branch = branch
        self.info_file = info_file.strip("/")
        self.timeout = timeout
        self._available: dict[str, str] | None = None

    @property
    def info_url(self):
        """Raw URL of the ``available_models.yaml`` index."""
        return (f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/"
                f"{self.info_file}")

    # -- remote index -----------------------------------------------------

    def available(self) -> dict[str, str]:
        """``{model_name: download_url}`` from ``available_models.yaml``."""
        if self._available is None:
            text = _read_url(self.info_url, self.timeout).decode()
            doc = yaml.safe_load(text) or {}
            self._available = {
                name: entry["Link"]
                for name, entry in doc.items()
                if isinstance(entry, dict) and entry.get("Link")
            }
        return self._available

    def models(self) -> list[str]:
        """Names of all downloadable pretrained models, sorted."""
        return sorted(self.available())

    # -- downloading ------------------------------------------------------

    def download(self, model, dest_root, overwrite=False, dry_run=False,
                 verbose=True) -> Path:
        """Download and extract one model into ``dest_root/<model>/``.

        The model is fetched as a ``.zip`` and extracted in place.  A model
        whose folder already contains ``*.h5`` files is skipped unless
        ``overwrite`` is set.  Returns the destination directory.
        """
        links = self.available()
        if model not in links:
            raise KeyError(
                f"unknown pretrained model {model!r}; "
                f"see `python src/cascade_data/cli.py models`")

        dest_dir = Path(dest_root) / model
        if dry_run:
            if verbose:
                print(f"{model}: would download -> {dest_dir}")
            return dest_dir

        if not overwrite and any(dest_dir.glob("*.h5")):
            if verbose:
                print(f"{model}: already downloaded ({dest_dir})")
            return dest_dir

        if verbose:
            print(f"{model}: downloading...")
        data = _read_url(links[model], self.timeout)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest_dir)
        _flatten_single_subdir(dest_dir)

        n = len(list(dest_dir.glob("*.h5")))
        if verbose:
            print(f"{model}: extracted {n} model file(s) -> {dest_dir}")
        return dest_dir

    def download_many(self, models, dest_root, **kwargs) -> dict[str, Path]:
        """Download several models; returns ``{model: dest_dir}``."""
        return {m: self.download(m, dest_root, **kwargs) for m in models}

    # -- local discovery --------------------------------------------------

    @staticmethod
    def local_models(dest_root) -> dict[str, Path]:
        """``{model_name: folder}`` for downloaded models under ``dest_root``.

        A folder counts as a model when it directly holds ``*.h5`` files.
        """
        root = Path(dest_root)
        if not root.is_dir():
            return {}
        return {
            d.name: d
            for d in sorted(root.iterdir())
            if d.is_dir() and any(d.glob("*.h5"))
        }


def _flatten_single_subdir(dest_dir):
    """Move the contents of a lone sub-folder up into ``dest_dir``.

    Some model archives wrap everything in a top-level folder; if the extracted
    tree has no ``*.h5`` at the top but exactly one sub-directory, its contents
    are moved up so the layout is always ``<model>/*.h5``.
    """
    if any(dest_dir.glob("*.h5")):
        return
    subdirs = [d for d in dest_dir.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        return
    inner = subdirs[0]
    for item in inner.iterdir():
        item.rename(dest_dir / item.name)
    inner.rmdir()
