import json
from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_old_fast_format_data(
    path: Path,
) -> tuple[npt.NDArray, list[str], str | None, str]:
    """Load data from fast format."""
    with open(f"{path}_items.json") as file_handle:
        items = json.load(file_handle)
    tokens, unk_index, name = items["items"], items["unk_index"], items["name"]

    with open(f"{path}_vectors.npy", "rb") as file_handle:
        vectors = np.load(file_handle)

    unk_token = tokens[unk_index] if unk_index is not None else None
    return vectors, tokens, unk_token, name
