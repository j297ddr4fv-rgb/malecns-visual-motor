"""Shared splitting for the repository's intentionally different selectors."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Optional

import numpy as np


def comma_terms(
    spec: str, suffixes: Optional[Iterable[str]] = None,
) -> Iterator[tuple[str, str, Optional[str]]]:
    suffixes = None if suffixes is None else tuple(suffixes)
    for raw in spec.split(","):
        term = raw.strip()
        if not term:
            continue
        side = None
        if suffixes is None and "/" in term and term.rfind("/") == len(term) - 2:
            side, term = term[-1], term[:-2]
        elif suffixes is not None:
            suffix = next((value for value in suffixes if term.endswith(value)), None)
            if suffix is not None:
                side, term = suffix[-1], term[:-len(suffix)]
        yield raw, term, side


def intersect_terms(
    term: str, select: Callable[[str], Any], *, unique: bool = False,
) -> Any:
    pieces = iter(term.split("&"))
    result = select(next(pieces).strip())
    for piece in pieces:
        result = np.intersect1d(result, select(piece.strip()), assume_unique=unique)
    return result
