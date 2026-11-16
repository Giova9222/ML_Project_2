"""Local catalogue of downloaded Cascade ``.mat`` files.

:class:`DatasetIndex` scans a data root, records a few cheap-to-compute facts
about every ``.mat`` file it finds, and persists the result as JSON.  Sessions
can then be discovered and filtered without loading the traces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cascade_data.two_photon import AP_SAMPLE_RATE, read_mat_structs


@dataclass
class IndexedFile:
    """Metadata for one locally downloaded ``.mat`` file."""

    dataset: str
    name: str
    path: str          # relative to the index root
    size: int
    n_members: int     # structs in the file (cells or trials)
    n_samples: int
    t_start: float
    t_end: float
    duration_s: float
    n_aps: int
    cell_type: str
    mat_fields: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "IndexedFile":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def probe_mat(path, ap_sample_rate=AP_SAMPLE_RATE) -> dict:
    """Read the cheap-to-compute facts about a Cascade ``.mat`` file.

    Files holding several structs are summarised by their first usable member;
    ``n_members`` records how many there are.
    """
    structs = read_mat_structs(path)
    usable = [
        s for s in structs
        if hasattr(s, "fluo_time") and np.isfinite(
            np.asarray(s.fluo_time, dtype=float)).any()
    ]
    if not usable:
        raise ValueError("no struct with a finite fluo_time")

    struct = usable[0]
    t = np.asarray(struct.fluo_time, dtype=float).ravel()
    t = t[np.isfinite(t)]
    ap = np.asarray(struct.events_AP, dtype=float).ravel()
    return {
        "n_members": len(structs),
        "n_samples": int(t.size),
        "t_start": float(t.min()),
        "t_end": float(t.max()),
        "duration_s": float(t.max() - t.min()),
        "n_aps": int(ap.size),
        "cell_type": str(getattr(struct, "cell_type", "")),
        "mat_fields": list(struct._fieldnames),
    }


class DatasetIndex:
    """Local catalogue of downloaded ``.mat`` files, persisted as JSON."""

    def __init__(self, root, index_path=None):
        self.root = Path(root)
        self.index_path = Path(index_path) if index_path else self.root / "index.json"

    def build(self, force=False, verbose=True) -> dict:
        """Scan ``root`` recursively and (re)write the index."""
        if self.index_path.exists() and not force:
            return self.load()

        entries = []
        for path in sorted(self.root.rglob("*.mat")):
            rel = path.relative_to(self.root)
            # The dataset is the folder that directly contains the file, so
            # ``data/raw/DS28-.../x.mat`` is filed under ``DS28-...`` rather
            # than the ``raw`` wrapper.  Loose files in the root are "unfiled".
            dataset = rel.parts[-2] if len(rel.parts) > 1 else "unfiled"
            try:
                meta = probe_mat(path)
            except Exception as exc:  # unreadable / not a Cascade file
                if verbose:
                    print(f"  ! skipping {rel}: {exc}")
                continue
            entries.append(IndexedFile(
                dataset=dataset, name=path.name, path=str(rel),
                size=path.stat().st_size, **meta,
            ))

        doc = {
            "root": str(self.root),
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_files": len(entries),
            "datasets": sorted({e.dataset for e in entries}),
            "files": [asdict(e) for e in entries],
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(doc, indent=2))
        if verbose:
            print(f"indexed {len(entries)} file(s) across "
                  f"{len(doc['datasets'])} dataset(s) -> {self.index_path}")
        return doc

    def load(self) -> dict:
        return json.loads(self.index_path.read_text())

    def entries(self, dataset=None) -> list[IndexedFile]:
        out = [IndexedFile.from_dict(d) for d in self.load()["files"]]
        if dataset is not None:
            out = [e for e in out if e.dataset == dataset]
        return out

    def datasets(self) -> list[str]:
        return self.load()["datasets"]

    def paths(self, dataset=None) -> list[Path]:
        return [self.root / e.path for e in self.entries(dataset)]
