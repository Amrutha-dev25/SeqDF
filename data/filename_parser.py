"""
Parses filenames of the form {video_id}_{m1}_{m2}_{m3}.mp4 into structured labels.

Example: "000_DF_FS_F2F.mp4" -> {
    "video_id": "000",
    "m1": "DF", "m2": "FS", "m3": "F2F",
    "m1_idx": 0, "m2_idx": 3, "m3_idx": 1,
    "sequence_class": <int 0-59>,   # treating the ordered triple as one of 60 classes
}
"""
import os
import re
from itertools import permutations
from typing import Optional

import yaml


def _load_methods(config_path: str = None) -> list:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "data_config.yaml"
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["methods"]


METHODS = _load_methods()
METHOD_TO_IDX = {m: i for i, m in enumerate(METHODS)}
IDX_TO_METHOD = {i: m for i, m in enumerate(METHODS)}

# All 60 ordered 3-permutations of the 5 methods, indexed 0-59.
# This ordering is deterministic (itertools.permutations is stable) so it is safe
# to use as a fixed class id mapping across runs.
ALL_SEQUENCES = list(permutations(METHODS, 3))
SEQUENCE_TO_CLASS = {seq: i for i, seq in enumerate(ALL_SEQUENCES)}
CLASS_TO_SEQUENCE = {i: seq for i, seq in enumerate(ALL_SEQUENCES)}

# All 10 possible unordered 3-method sets (C(5,3) = 10), used for Stage A class
# balance checks and as the conditioning input to Stage B's sequence decoder.
from itertools import combinations as _combinations
ALL_SETS = [tuple(sorted(c)) for c in _combinations(METHODS, 3)]
SET_TO_IDX = {s: i for i, s in enumerate(ALL_SETS)}
IDX_TO_SET = {i: s for i, s in enumerate(ALL_SETS)}

# Build regex dynamically from configured method names, longest-first so e.g.
# "FSh" doesn't get accidentally matched by a hypothetical shorter prefix.
_method_pattern = "|".join(sorted(METHODS, key=len, reverse=True))
FILENAME_RE = re.compile(
    rf"^(?P<video_id>.+?)_(?P<m1>{_method_pattern})_(?P<m2>{_method_pattern})_(?P<m3>{_method_pattern})\.mp4$"
)


class FilenameParseError(ValueError):
    pass


def parse_filename(filename: str) -> dict:
    """
    Parse a single generated-video filename into its label dict.
    Raises FilenameParseError if the filename doesn't match the expected pattern
    or uses a method code not present in configs/data_config.yaml.
    """
    base = os.path.basename(filename)
    match = FILENAME_RE.match(base)
    if not match:
        raise FilenameParseError(
            f"Filename '{base}' does not match expected pattern "
            f"'{{video_id}}_{{m1}}_{{m2}}_{{m3}}.mp4' with methods {METHODS}"
        )

    video_id = match.group("video_id")
    m1, m2, m3 = match.group("m1"), match.group("m2"), match.group("m3")

    if len({m1, m2, m3}) != 3:
        raise FilenameParseError(
            f"Filename '{base}' repeats a method across steps ({m1},{m2},{m3}) - "
            f"expected an ordered permutation of 3 distinct methods."
        )

    sequence_class = SEQUENCE_TO_CLASS.get((m1, m2, m3))
    if sequence_class is None:
        raise FilenameParseError(
            f"Sequence ({m1},{m2},{m3}) from '{base}' is not in the 60 valid "
            f"3-permutations of {METHODS}."
        )

    # Multi-label set vector: 1 if method appears anywhere in (m1,m2,m3), else 0.
    # This is the PRIMARY label for Stage A set detection - order-agnostic.
    set_vector = [0] * len(METHODS)
    for m in (m1, m2, m3):
        set_vector[METHOD_TO_IDX[m]] = 1

    # Stage B uses this: given the known unordered set, which of the 6 possible
    # orderings of that specific set was actually used. Computed as the index of
    # (m1,m2,m3) within the sorted list of all 6 permutations of this video's set.
    from itertools import permutations as _permutations
    this_set_sorted = tuple(sorted([m1, m2, m3]))
    set_permutations = list(_permutations(this_set_sorted))
    ordering_within_set = set_permutations.index((m1, m2, m3))

    return {
        "filename": base,
        "video_id": video_id,
        "m1": m1, "m2": m2, "m3": m3,
        "m1_idx": METHOD_TO_IDX[m1],
        "m2_idx": METHOD_TO_IDX[m2],
        "m3_idx": METHOD_TO_IDX[m3],
        "sequence_class": sequence_class,
        "set_vector": ",".join(str(v) for v in set_vector),  # CSV-safe string e.g. "1,1,0,1,0"
        "set_sorted": "-".join(this_set_sorted),    # e.g. "DF-F2F-FS" - canonical set id, 10 possible values
        "ordering_within_set": ordering_within_set,  # 0-5, which of the 6 orderings of this set
    }


def try_parse_filename(filename: str) -> Optional[dict]:
    """Same as parse_filename but returns None instead of raising on failure.
    Use this when scanning a directory that might contain stray/corrupt filenames."""
    try:
        return parse_filename(filename)
    except FilenameParseError:
        return None


def class_to_methods(sequence_class: int) -> tuple:
    """Inverse lookup: class id -> (m1, m2, m3) tuple of method name strings."""
    if sequence_class not in CLASS_TO_SEQUENCE:
        raise ValueError(f"sequence_class {sequence_class} out of range [0, 59]")
    return CLASS_TO_SEQUENCE[sequence_class]


if __name__ == "__main__":
    # quick self-test
    test_cases = ["000_DF_FS_F2F.mp4", "092_NT_FSh_FS.mp4", "bad_file.mp4", "001_DF_DF_FS.mp4"]
    for tc in test_cases:
        result = try_parse_filename(tc)
        print(f"{tc:30s} -> {result}")
    print(f"\nTotal valid sequence classes: {len(ALL_SEQUENCES)}")
