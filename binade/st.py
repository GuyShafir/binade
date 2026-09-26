"""Streaming safetensors access: parse headers, read one tensor (or row range) at a time.

safetensors 0.8.0 `safe_open` raises on BF16 for both the numpy and mlx frameworks
("data type 'bfloat16' not understood"), so tensors are read as raw bytes at their
header offsets. BF16 tensors come back as uint16 bit patterns.
"""

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NP_DTYPE = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
    "I16": np.int16,
    "F16": np.float16,
    "BF16": np.uint16,  # bit patterns
    "U32": np.uint32,
    "I32": np.int32,
    "F32": np.float32,
    "U64": np.uint64,
    "I64": np.int64,
    "F64": np.float64,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: Path
    offset: int  # absolute byte offset of the data in `file`
    nbytes: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def rows(self) -> int:
        """Rows of the 2-D view [prod(shape[:-1]), shape[-1]]."""
        return math.prod(self.shape[:-1]) if self.shape else 1


def read_header(path) -> tuple[dict, int]:
    """Return (header dict, byte offset where the data section starts)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def shard_files(model_dir) -> list[Path]:
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if not names:
        raise FileNotFoundError(f"no safetensors files in {model_dir}")
    return [model_dir / n for n in names]


def list_tensors(model_dir) -> list[TensorInfo]:
    out = []
    for path in shard_files(model_dir):
        header, start = read_header(path)
        for name, v in header.items():
            if name == "__metadata__":
                continue
            b, e = v["data_offsets"]
            out.append(TensorInfo(name, v["dtype"], tuple(v["shape"]), path, start + b, e - b))
    return sorted(out, key=lambda t: (str(t.file), t.offset))


def read(info: TensorInfo, rows: tuple[int, int] | None = None) -> np.ndarray:
    """Read a whole tensor, or rows [r0, r1) of its 2-D view, from disk (no mmap)."""
    dt = np.dtype(NP_DTYPE[info.dtype])
    if rows is None:
        a = np.fromfile(info.file, dtype=dt, count=info.numel, offset=info.offset)
        return a.reshape(info.shape)
    r0, r1 = rows
    k = info.shape[-1]
    a = np.fromfile(info.file, dtype=dt, count=(r1 - r0) * k, offset=info.offset + r0 * k * dt.itemsize)
    return a.reshape(r1 - r0, k)


_SNAPSHOT = re.compile(r"models--(?P<org>[^/]+?)--(?P<name>[^/]+)/snapshots/(?P<rev>[0-9a-f]{40})")


def resolve(model: str, revision: str | None = None) -> Path:
    """Local directory, or an HF repo id already present in the local cache (never downloads).

    Reads the cache index directly: downloads pinned to a commit have no refs/main, and
    downloads filtered with --include are partial snapshots, which
    snapshot_download(local_files_only=True) rejects in huggingface_hub 1.x.
    """
    p = Path(model).expanduser()
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import scan_cache_dir

    snaps = [
        s
        for r in scan_cache_dir().repos
        if r.repo_id == model and r.repo_type == "model"
        for s in r.revisions
        if revision is None or s.commit_hash.startswith(revision) or revision in s.refs
    ]
    if len(snaps) != 1:
        raise FileNotFoundError(f"{model}: {len(snaps)} matching cached snapshots; download it or pass --revision")
    return Path(snaps[0].snapshot_path)


_REPO = Path(__file__).resolve().parents[1]


def display_path(p) -> str:
    """A path for result files: relative to the repository when inside it, ~/... under the
    home directory, so results record no machine- or user-specific prefix."""
    s = str(p)
    try:
        q = Path(s).expanduser().resolve()
    except (OSError, RuntimeError):
        return s
    for base, prefix in ((_REPO, ""), (Path.home().resolve(), "~/")):
        try:
            return prefix + str(q.relative_to(base))
        except ValueError:
            continue
    return s


def display_args(args: dict) -> dict:
    """argparse values for result files, paths through display_path."""
    return {k: display_path(v) if isinstance(v, Path) else v for k, v in args.items()}


def provenance(model_dir: Path) -> dict:
    """Repo id and revision when the directory is an HF cache snapshot."""
    m = _SNAPSHOT.search(str(model_dir))
    if not m:
        return {"path": display_path(model_dir)}
    return {
        "path": display_path(model_dir),
        "repo_id": f"{m['org']}/{m['name']}",
        "revision": m["rev"],
    }


ST_DTYPE = {np.dtype(np.uint8): "U8", np.dtype(np.uint32): "U32", np.dtype(np.uint16): "U16", np.dtype(np.float32): "F32"}


class Writer:
    """Streaming safetensors writer.

    Tensors are appended back to back; the header goes into space reserved at the start
    of the file and is padded with spaces, which the format allows. Nothing is held in
    memory beyond the chunk being written.
    """

    def __init__(self, path, reserve: int, metadata: dict | None = None):
        self.path = Path(path)
        self.reserve = (reserve + 7) // 8 * 8
        self.metadata = metadata or {}
        self.entries = {}
        self.f = open(self.path, "wb")
        self.f.write(b"\0" * (8 + self.reserve))
        self.pos = 0
        self.open = None

    def begin(self, name: str, dtype: str, shape) -> None:
        assert self.open is None and name not in self.entries, name
        self.open = (name, dtype, [int(s) for s in shape], self.pos)

    def write(self, data: np.ndarray) -> None:
        b = np.ascontiguousarray(data).tobytes()
        self.f.write(b)
        self.pos += len(b)

    def end(self) -> None:
        name, dtype, shape, start = self.open
        itemsize = np.dtype(NP_DTYPE[dtype]).itemsize
        if self.pos - start != math.prod(shape) * itemsize:
            raise ValueError(f"{name}: wrote {self.pos - start} bytes, shape {shape} {dtype} needs {math.prod(shape) * itemsize}")
        self.entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, self.pos]}
        self.open = None

    def add(self, name: str, a: np.ndarray, dtype: str | None = None) -> None:
        self.begin(name, dtype or ST_DTYPE[a.dtype], a.shape)
        self.write(a)
        self.end()

    def close(self) -> None:
        header = dict(self.entries)
        if self.metadata:
            header["__metadata__"] = self.metadata
        h = json.dumps(header, separators=(",", ":")).encode()
        if len(h) > self.reserve:
            self.f.close()
            raise ValueError(f"{self.path}: header needs {len(h)} bytes, reserved {self.reserve}")
        self.f.seek(0)
        self.f.write(struct.pack("<Q", self.reserve) + h + b" " * (self.reserve - len(h)))
        self.f.close()
