"""Remote access to the Cascade ground-truth dataset.

The Cascade repository (https://github.com/HelmchenLabSoftware/Cascade) ships a
``Ground_truth`` folder with one sub-folder per dataset (``DS01-...``,
``X-NAOMi-...``), each holding many small ``*_mini.mat`` files, one per cell.

:class:`CascadeRepository` is a read-only, cached view of that folder.  It lists
datasets and their files and downloads a chosen subset locally.  Downloads are
idempotent (existing files are skipped) and can be limited per dataset, which
keeps the local footprint small.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = "HelmchenLabSoftware/Cascade"
GT_PATH = "Ground_truth"
BRANCH = "master"

_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
_USER_AGENT = "cascade-data/1.0"


def human_bytes(n: float) -> str:
    """Format a byte count for humans."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@dataclass(frozen=True)
class RemoteFile:
    """A single ``.mat`` file available in the remote repository."""

    dataset: str
    name: str
    size: int
    download_url: str


class CascadeRepository:
    """Read-only, cached view of the Cascade ``Ground_truth`` folder."""

    def __init__(self, repo=REPO, gt_path=GT_PATH, branch=BRANCH,
                 cache_dir=None, timeout=30):
        self.repo = repo
        self.gt_path = gt_path.strip("/")
        self.branch = branch
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self._listing_cache: dict[str, list] = {}

    # -- remote listing ---------------------------------------------------

    def _get_json(self, path):
        """Fetch (and memoise) the GitHub contents listing for ``path``."""
        if path in self._listing_cache:
            return self._listing_cache[path]

        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / (path.replace("/", "__") + ".json")

        if cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text())
        else:
            url = _API.format(repo=self.repo, path=path, ref=self.branch)
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data))

        self._listing_cache[path] = data
        return data

    def datasets(self) -> list[str]:
        """Names of all dataset folders, sorted."""
        entries = self._get_json(self.gt_path)
        return sorted(e["name"] for e in entries if e["type"] == "dir")

    def files(self, dataset: str) -> list[RemoteFile]:
        """All ``.mat`` files in one dataset folder, sorted by name."""
        entries = self._get_json(f"{self.gt_path}/{dataset}")
        out = [
            RemoteFile(dataset, e["name"], e["size"], e["download_url"])
            for e in entries
            if e["type"] == "file" and e["name"].endswith(".mat")
        ]
        return sorted(out, key=lambda f: f.name)

    # -- downloading ------------------------------------------------------

    def download(self, dataset, dest_root, limit=None, names=None,
                 overwrite=False, dry_run=False, verbose=True) -> list[Path]:
        """Download a subset of one dataset into ``dest_root/<dataset>/``.

        When ``names`` is not given the files are ordered smallest-first, so
        ``limit=1`` grabs the cheapest file — handy on a storage-tight machine.
        Existing files are skipped unless ``overwrite`` is set.
        """
        dest_root = Path(dest_root)
        files = self.files(dataset)

        if names is not None:
            wanted = set(names)
            files = [f for f in files if f.name in wanted]
        else:
            files = sorted(files, key=lambda f: f.size)
        if limit is not None:
            files = files[:limit]

        dest_dir = dest_root / dataset
        paths = [dest_dir / f.name for f in files]
        pending = [
            (f, dest_dir / f.name)
            for f in files
            if overwrite or not (dest_dir / f.name).exists()
        ]

        if verbose:
            total = sum(f.size for f, _ in pending)
            print(f"{dataset}: {len(files)} selected, {len(pending)} to fetch "
                  f"({human_bytes(total)})")
        if dry_run:
            return paths

        dest_dir.mkdir(parents=True, exist_ok=True)
        for f, dest in pending:
            if verbose:
                print(f"  -> {f.name} ({human_bytes(f.size)})")
            urllib.request.urlretrieve(f.download_url, dest)
        return paths

    def download_many(self, datasets, dest_root, **kwargs) -> dict[str, list[Path]]:
        """Download a subset of several datasets; returns ``{dataset: paths}``."""
        return {d: self.download(d, dest_root, **kwargs) for d in datasets}
