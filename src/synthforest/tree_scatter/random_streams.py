from __future__ import annotations

import hashlib
import random


def _stream_seed(seed: int, stream_name: str) -> int:
    payload = f"{int(seed)}\0{str(stream_name)}".encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=16,
        person=b"fg-rng-stream",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def make_random_stream(seed: int, stream_name: str) -> random.Random:
    """Return a deterministic RNG keyed by scene seed and independent stream name."""
    return random.Random(_stream_seed(seed, stream_name))
