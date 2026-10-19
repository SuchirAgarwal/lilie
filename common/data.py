"""
GPIC streaming for both trainers.
"""
from __future__ import annotations

import concurrent.futures
import functools
import io
import itertools
import json
import math
import os
import subprocess
import tempfile
import random
import signal
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds
from PIL import Image

HF_BASE = "https://huggingface.co/datasets/stanford-vision-lab/gpic/resolve/main"
# split -> (number of tars, number of samples)
SPLITS = {"train": (8000, 100_000_000), "val": (32, 200_000)}
IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")
CAPTION_TYPES = ("tag", "short", "medium", "long")


# ----------------------------------------------------------------------- transport
def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    path = Path.home() / ".cache" / "huggingface" / "token"
    return path.read_text().strip() if path.is_file() else ""


@functools.lru_cache(maxsize=None)
def _curl_auth() -> str:
    token = _hf_token()
    if not token:
        return ""
    path = Path.home() / ".cache" / "huggingface" / "curl_header.conf"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(f'header = "Authorization: Bearer {token}"\n')
    os.replace(tmp, path)
    return str(path)


def shard_url(split: str, index: int) -> str:
    return f"{HF_BASE}/{split}/gpic_{split}_{index:05d}.tar"


def _die_with_parent() -> None:
    """Have curl exit when its loader worker is killed (Linux; no-op elsewhere)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def _curl(url: str, output: str, pass_fds=()) -> None:
    """Download ``url`` to ``output`` (resumed and retried, so a dropped or throttled CDN
    connection costs a retry, not a shard)."""
    for attempt in range(30):
        proc = subprocess.run(
            ["curl", "-f", "-s", "-S", "-L", "--http1.1", "--retry", "10", "--retry-delay", "2",
             "--connect-timeout", "30", "--speed-limit", "500000", "--speed-time", "30",
             "-C", "-", "-o", output, *(["-K", _curl_auth()] if _curl_auth() else []), url],
            pass_fds=pass_fds, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            preexec_fn=_die_with_parent)
        if proc.returncode == 0:
            return
        print(f"[data] curl exit {proc.returncode} on {url} (attempt {attempt + 1}): "
              f"{proc.stderr.strip()[-200:]}" + (" (is $TMPDIR full?)" if proc.returncode == 23 else ""), flush=True)
    raise RuntimeError(f"could not download {url}")


def _fetch(url: str, cache_dir: str | None = None):
    """Return one tar open for reading. Training shards go into an anonymous temporary file
    (no name, so a killed worker leaves nothing behind); with ``cache_dir`` the shard is kept
    on disk and reused, which is what validation wants since it re-reads the same few shards."""
    if cache_dir is None:
        handle = tempfile.TemporaryFile(prefix="gpic-", suffix=".tar")
        _curl(url, f"/dev/fd/{handle.fileno()}", pass_fds=(handle.fileno(),))
        handle.seek(0)
        return handle
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.rsplit("/", 1)[1])
    if not os.path.exists(path):
        _curl(url, f"{path}.part")
        os.replace(f"{path}.part", path)
    return open(path, "rb")


def _download_shards(sources, cache_dir: str | None = None):
    """WebDataset stage: ``{"url"}`` -> ``{"url", "stream"}``. The next shard downloads in the
    background while the current one is being read, so each loader worker holds at most two
    shards (about 3 GB) in ``$TMPDIR``."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for source in sources:
            future = pool.submit(_fetch, source["url"], cache_dir)
            if pending is not None:
                yield {"url": pending[0], "stream": pending[1].result()}
            pending = (source["url"], future)
        if pending is not None:
            yield {"url": pending[0], "stream": pending[1].result()}


def shard_offset(resume_step: int, global_batch: int, split: str = "train") -> int:
    """Resume from the correct shard."""
    n_shards, n_samples = SPLITS[split]
    return (resume_step * global_batch // (n_samples // n_shards)) % n_shards


# ---------------------------------------------------------------------- transforms
class LILIETransform:
    """LILIE encoder training: resize the short side (bicubic), center crop, scale to [-1, 1]."""

    def __init__(self, image_size: int):
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.transform(image.convert("RGB")).numpy()


def center_crop_arr(image: Image.Image, image_size: int) -> Image.Image:
    """Deterministic center crop (PixelGen's validation transform)."""
    image = image.convert("RGB")
    while min(*image.size) >= 2 * image_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
    scale = image_size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(image)
    y = (arr.shape[0] - image_size) // 2
    x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[y:y + image_size, x:x + image_size])


def random_crop_arr(image: Image.Image, image_size: int, min_crop_frac: float = 0.8,
                    max_crop_frac: float = 1.0) -> Image.Image:
    """Random-resized crop (PixelGen's training transform)."""
    image = image.convert("RGB")
    lo, hi = math.ceil(image_size / max_crop_frac), math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(lo, hi + 1)
    while min(*image.size) >= 2 * smaller_dim_size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
    scale = smaller_dim_size / min(*image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(image)
    y = random.randrange(arr.shape[0] - image_size + 1)
    x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[y:y + image_size, x:x + image_size])


@dataclass(frozen=True)
class JiTTransform:
    """Generator training: PixelGen's random-resized crop (training) or center crop (validation), scaled to [-1, 1]."""

    image_size: int
    is_val: bool = False

    def __call__(self, image: Image.Image) -> np.ndarray:
        crop = center_crop_arr if self.is_val else random_crop_arr
        arr = np.asarray(crop(image, self.image_size), dtype=np.float32) / np.float32(255.0)
        return np.transpose(arr * np.float32(2.0) - np.float32(1.0), (2, 0, 1)).copy()


# ------------------------------------------------------------------------- stream
def decode_sample(sample: dict, transform):
    """A grouped WebDataset sample -> ``(image, caption, caption_type)``."""
    meta = json.loads(sample["json"]) if isinstance(sample["json"], (bytes, str)) else sample["json"]
    if meta.get("caption_type") not in CAPTION_TYPES or not str(meta.get("caption", "")).strip():
        raise ValueError(f"sample {sample.get('__key__', '?')!r} has no usable caption")  # skipped by the handler
    for key in IMAGE_KEYS:
        if key in sample:
            with Image.open(io.BytesIO(sample[key])) as image:
                return transform(image.convert("RGB")), str(meta["caption"]), str(meta["caption_type"])
    raise KeyError(f"sample {sample.get('__key__', '?')!r} has no image member")


def _split_by_process(shards, index: int, count: int):
    """Round-robin shard split across JAX processes (each process is one host)."""
    yield from itertools.islice(shards, index, None, count)


def _forever(pipeline):
    while True:
        n = 0
        for sample in pipeline:
            yield sample
            n += 1
        if n == 0:
            raise RuntimeError("the GPIC stream yielded no samples; check HF access / HF_TOKEN")


class GPICStream(torch.utils.data.IterableDataset):

    def __init__(self, split: str, transform, *, process_index: int, process_count: int,
                 shuffle_buffer: int, infinite: bool, offset: int = 0):
        super().__init__()
        n_shards = SPLITS[split][0]
        if n_shards < process_count:
            raise ValueError(f"{split}: {n_shards} shards cannot cover {process_count} processes")
        urls = [shard_url(split, i) for i in range(n_shards)]
        self.urls = urls[offset:] + urls[:offset]
        self.transform = transform
        self.process_index, self.process_count = process_index, process_count
        self.shuffle_buffer, self.infinite = shuffle_buffer, infinite
        # Validation re-reads the same few shards at every event: keep them in $TMPDIR/gpic-val.
        self.cache_dir = None if split == "train" else os.path.join(tempfile.gettempdir(), "gpic-val")

    def __iter__(self):
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.urls),
            functools.partial(_split_by_process, index=self.process_index, count=self.process_count),
            wds.split_by_worker,
            functools.partial(_download_shards, cache_dir=self.cache_dir),
            functools.partial(wds.tariterators.tar_file_expander, handler=wds.warn_and_continue),
            functools.partial(wds.tariterators.group_by_keys, handler=wds.warn_and_continue),
            # A corrupt image skips one sample instead of killing the worker.
            wds.map(lambda s: decode_sample(s, self.transform), handler=wds.warn_and_continue),
            wds.shuffle(self.shuffle_buffer),
        )
        yield from _forever(pipeline) if self.infinite else pipeline


def _collate(batch):
    images, captions, caption_types = zip(*batch)
    return np.stack(images).astype(np.float32, copy=False), list(captions), list(caption_types)


def build_loader(split: str, transform, *, local_batch: int, num_workers: int, shuffle_buffer: int,
                 process_index: int, process_count: int, resume_step: int = 0,
                 infinite: bool = True, prefetch_factor: int = 4) -> torch.utils.data.DataLoader:
    """
    A CPU DataLoader over this process's share of ``split``.
    """
    offset = shard_offset(resume_step, local_batch * process_count, split) if resume_step else 0
    if offset and process_index == 0:
        print(f"[data] resuming the {split} stream at shard offset {offset}", flush=True)
    stream = GPICStream(split, transform, process_index=process_index, process_count=process_count,
                        shuffle_buffer=shuffle_buffer, infinite=infinite, offset=offset)
    workers = min(num_workers, SPLITS[split][0] // process_count)
    return torch.utils.data.DataLoader(
        stream, batch_size=local_batch, num_workers=workers, collate_fn=_collate, drop_last=True,
        multiprocessing_context="spawn" if workers else None,
        prefetch_factor=prefetch_factor if workers else None)


def read_captions_jsonl(path: str) -> list[dict]:
    """The caption list used for sampling: one ``{"key", "caption", "caption_type"}`` per line."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
