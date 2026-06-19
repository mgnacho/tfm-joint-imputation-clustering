from __future__ import annotations

import hashlib
from typing import Any


GUROBI_MAX_SEED = 2_000_000_000


def stable_seed(*parts: Any, base: int = 10_000) -> int:
    """Return a deterministic 32-bit seed without relying on Python's hash()."""
    payload = "|".join([str(base), *(repr(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def normalize_gurobi_seed(seed: int) -> int:
    """Transform any integer into the range accepted by Gurobi."""
    return int(seed) % (GUROBI_MAX_SEED + 1)