"""Read individual tensors from a safetensors checkpoint without loading the model.

ROTATE only needs one neuron's weights and the (un)embedding matrix, so there is no reason to
download or instantiate a full model. :class:`Checkpoint` parses the safetensors headers and reads
just the bytes of the requested tensors (or rows of a tensor). A checkpoint is either a local
directory or a Hugging Face repo id; for a repo, the bytes are fetched with HTTP range requests,
so one Gemma-2-2B neuron plus its unembedding is ~1.2 GB of traffic instead of ~5 GB.
"""

import json
import struct
import warnings
from pathlib import Path
from typing import Optional, Sequence

import torch

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}

# Read rows one by one (one request each) only for a handful of rows; otherwise read it all.
_MAX_ROW_READS = 64


class Checkpoint:
    """Lazy, read-only access to the tensors of a (possibly sharded) safetensors checkpoint.

    Args:
        source: a local directory, or a Hugging Face repo id such as ``"google/gemma-2-2b-it"``.
        revision: optional branch, tag or commit of a Hugging Face repo.
    """

    def __init__(self, source: str, revision: Optional[str] = None) -> None:
        self.source = str(source)
        self.revision = revision
        self.is_local = Path(self.source).is_dir()
        self._local_files: dict[str, Path] = {}  # remote shards downloaded after a failed range read
        self._headers: dict[str, tuple[int, dict]] = {}
        if self.is_local:
            import fsspec

            self._fs = fsspec.filesystem("file")
        else:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem()
        self.weight_map = self._load_weight_map()

    # -- files ---------------------------------------------------------------------------------

    def _path(self, filename: str) -> str:
        if self.is_local:
            return str(Path(self.source) / filename)
        rev = f"@{self.revision}" if self.revision else ""
        return f"{self.source}{rev}/{filename}"

    def _read(self, filename: str, offset: int, length: int) -> bytes:
        """``length`` bytes of ``filename`` starting at ``offset``."""
        if filename not in self._local_files:
            try:
                with self._fs.open(self._path(filename), "rb", block_size=0) as f:
                    f.seek(offset)
                    data = f.read(length)
                if len(data) == length:
                    return data
                raise OSError(f"short read: got {len(data)} of {length} bytes")
            except Exception as err:  # range reads unsupported: fall back to the whole file
                if self.is_local:
                    raise
                from huggingface_hub import hf_hub_download

                warnings.warn(f"Range read of {filename} failed ({err}); downloading the whole file.")
                self._local_files[filename] = Path(
                    hf_hub_download(self.source, filename, revision=self.revision)
                )
        with open(self._local_files[filename], "rb") as f:
            f.seek(offset)
            return f.read(length)

    def _load_weight_map(self) -> dict[str, str]:
        index = "model.safetensors.index.json"
        if self._fs.exists(self._path(index)):
            with self._fs.open(self._path(index), "r") as f:
                return json.load(f)["weight_map"]
        single = "model.safetensors"
        if not self._fs.exists(self._path(single)):
            raise FileNotFoundError(f"No {index} or {single} in {self.source}")
        return {name: single for name in self._header(single)[1] if name != "__metadata__"}

    def _header(self, filename: str) -> tuple[int, dict]:
        """(byte offset of the data section, parsed JSON header) of one safetensors file."""
        if filename not in self._headers:
            (n,) = struct.unpack("<Q", self._read(filename, 0, 8))
            header = json.loads(self._read(filename, 8, n))
            self._headers[filename] = (8 + n, header)
        return self._headers[filename]

    # -- tensors -------------------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def info(self, name: str) -> dict:
        """dtype, shape and location of a tensor, read from its file's header."""
        if name not in self.weight_map:
            raise KeyError(f"{name!r} is not in {self.source}")
        filename = self.weight_map[name]
        data_start, header = self._header(filename)
        entry = header[name]
        start, end = entry["data_offsets"]
        return {
            "file": filename,
            "dtype": _DTYPES[entry["dtype"]],
            "shape": tuple(entry["shape"]),
            "offset": data_start + start,
            "nbytes": end - start,
        }

    def get(self, name: str, rows: Optional[Sequence[int]] = None) -> torch.Tensor:
        """A tensor in its stored dtype, or only the given rows (first-dimension indices) of it."""
        meta = self.info(name)
        shape = meta["shape"]
        if rows is not None and len(rows) <= _MAX_ROW_READS:
            row_shape = shape[1:]
            row_bytes = meta["nbytes"] // shape[0]
            out = []
            for r in rows:
                if not 0 <= r < shape[0]:
                    raise IndexError(f"row {r} out of range for {name} with shape {shape}")
                buf = self._read(meta["file"], meta["offset"] + r * row_bytes, row_bytes)
                out.append(_to_tensor(buf, meta["dtype"], row_shape))
            return torch.stack(out)
        buf = self._read(meta["file"], meta["offset"], meta["nbytes"])
        tensor = _to_tensor(buf, meta["dtype"], shape)
        return tensor if rows is None else tensor[list(rows)]


def _to_tensor(buf: bytes, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
    if len(buf) == 0:
        return torch.empty(shape, dtype=dtype)
    with warnings.catch_warnings():  # bytearray is writable; silence the non-writable buffer warning
        warnings.simplefilter("ignore")
        return torch.frombuffer(bytearray(buf), dtype=dtype).reshape(shape)
