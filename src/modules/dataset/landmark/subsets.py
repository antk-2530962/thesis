"""Canonical registry of MediaPipe Holistic landmark subsets.

Single source of truth for every landmark subset used across notebooks,
training scripts and evaluation — nothing should re-derive index lists.

Holistic row layout (GISLR parquet order, 543 rows/frame):

    rows   0-467  face      (468 face-mesh landmarks; row == face-mesh index)
    rows 468-488  left_hand (21)
    rows 489-521  pose      (33; holistic row = 489 + pose index)
    rows 522-542  right_hand(21)

All subsets store **sorted ascending holistic row indices** — the same
convention as the trained ME-126 run's ``landmarks.npy``
(``src/models/gislr/gru/20260715-190729/cache/landmarks.npy``).

Provenance:
- 1st-place lists (LIPS/EYES/NOSE/POSE): Kaggle GISLR winning solution
  (``src/gislr.0.competition.entry.1st.ipynb``, hoyso48).
- ME-126 derivation and evidence: ``docs/2026-07-15.md`` §4-§5.
- Subset discriminability scores: ``docs/2026-07-16.md`` /
  ``src/gislr.0.dataset.subset-comparison.ipynb`` (see ``SUBSETS`` docstrings).
"""

from dataclasses import dataclass, field

import numpy as np

N_LANDMARKS = 543  # holistic rows per frame (ROWS_PER_FRAME)
POSE_OFFSET = 489  # holistic row of pose landmark 0


def pose_rows(pose_indices: list[int]) -> list[int]:
    """Map MediaPipe Pose indices (0-32) to holistic row indices."""
    return [POSE_OFFSET + i for i in pose_indices]


# ============================================================
# Component groups (building blocks — face-mesh indices ARE holistic rows)
# ============================================================

#: 40 lip landmarks (1st-place ``LIP`` list) — kept on linguistic grounds
#: (mouthing), not motion: motion energy sees a flat ~0.003 band on the face.
LIPS_40: list[int] = sorted([
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
    314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
])

#: 16 right-eye landmarks (1st-place ``REYE``).
RIGHT_EYE_16: list[int] = sorted([
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
])

#: 16 left-eye landmarks (1st-place ``LEYE``).
LEFT_EYE_16: list[int] = sorted([
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
])

#: 4 nose landmarks (1st-place ``NOSE``).
NOSE_4: list[int] = sorted([1, 2, 98, 327])

#: eyes + nose, 36 — near-rigid head-pose anchor + non-manual cues.
EYES_NOSE_36: list[int] = sorted(RIGHT_EYE_16 + LEFT_EYE_16 + NOSE_4)

#: full left hand mesh, 21.
LEFT_HAND_21: list[int] = list(range(468, 489))

#: full right hand mesh, 21.
RIGHT_HAND_21: list[int] = list(range(522, 543))

#: both hands, 42 — the primary articulators.
HANDS_42: list[int] = LEFT_HAND_21 + RIGHT_HAND_21

#: upper-body pose {11-16 shoulders/elbows/wrists, 23-24 hips}, 8 — the
#: always-detected arm trajectory + spatial anchor (1st place drafted this as
#: ``POSE`` but shipped with it commented out).
UPPER_BODY_POSE_8: list[int] = pose_rows([11, 12, 13, 14, 15, 16, 23, 24])

#: pose wrist-adjacent hand points {17-22}, 6 — highest xy movers, but
#: duplicate the hand meshes when those are detected.
POSE_HAND_POINTS_6: list[int] = pose_rows([17, 18, 19, 20, 21, 22])

#: complete face mesh, 468.
FACE_ALL_468: list[int] = list(range(0, 468))

#: complete pose, 33.
POSE_ALL_33: list[int] = pose_rows(list(range(33)))


# ============================================================
# Subset registry
# ============================================================

@dataclass(frozen=True)
class LandmarkSubset:
    """A named landmark subset: sorted holistic row indices + provenance."""

    name: str
    indices: tuple[int, ...]
    description: str
    provenance: str
    #: filled in by the subset-comparison analysis (docs/2026-07-16.md);
    #: probe = balanced logistic-regression val accuracy on per-video
    #: descriptors, global scope (250 classes). None = not yet scored.
    probe_acc_global: float | None = field(default=None)

    def __post_init__(self) -> None:
        idx = tuple(sorted(set(self.indices)))
        if idx != self.indices:
            object.__setattr__(self, "indices", idx)
        if not idx or idx[0] < 0 or idx[-1] >= N_LANDMARKS:
            raise ValueError(f"{self.name}: indices out of range 0-{N_LANDMARKS - 1}")

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.indices, dtype=np.int64)

    def feature_columns(self, coords: str = "xyz") -> np.ndarray:
        """Flat feature-column indices for a landmark-major (543 × xyz) frame.

        ``coords``: which of "x","y","z" to keep (e.g. "xy" drops z). Matches
        the GRU feature-cache layout: column = landmark_row * 3 + coord.
        """
        coord_pos = ["xyz".index(c) for c in coords]
        return np.asarray(
            [i * 3 + c for i in self.indices for c in coord_pos], dtype=np.int64
        )


def _make(name: str, indices: list[int], description: str, provenance: str,
          probe_acc_global: float | None = None) -> LandmarkSubset:
    return LandmarkSubset(name, tuple(sorted(set(indices))), description,
                          provenance, probe_acc_global)


SUBSETS: dict[str, LandmarkSubset] = {s.name: s for s in [
    _make(
        "FULL_543",
        list(range(N_LANDMARKS)),
        "All holistic landmarks — the no-selection baseline.",
        "GRU baseline run 20260713-213000 (70.59% val acc).",
    ),
    _make(
        "FP_118",
        LIPS_40 + HANDS_42 + EYES_NOSE_36,
        "Kaggle 1st-place subset: lips + hands + eyes/nose, no pose.",
        "gislr.0.competition.entry.1st.ipynb POINT_LANDMARKS (hoyso48).",
    ),
    _make(
        "ME_126",
        LIPS_40 + HANDS_42 + EYES_NOSE_36 + UPPER_BODY_POSE_8,
        "FP_118 ∪ upper-body pose {11-16,23,24} — motion-energy keep set.",
        "docs/2026-07-15.md §4; GRU run 20260715-190729 (73.73% val acc).",
    ),
    _make(
        "ME_132",
        LIPS_40 + HANDS_42 + EYES_NOSE_36 + UPPER_BODY_POSE_8 + POSE_HAND_POINTS_6,
        "ME_126 + pose wrist-adjacent hand points {17-22} (the 'optional' row).",
        "docs/2026-07-15.md §4 optional row.",
    ),
    _make(
        "HANDS_42",
        HANDS_42,
        "Both hand meshes only — primary articulators, minimal subset.",
        "Component ablation floor.",
    ),
    _make(
        "HANDS_POSE_50",
        HANDS_42 + UPPER_BODY_POSE_8,
        "Hands + upper-body pose, no face at all — tests what face adds.",
        "Component ablation.",
    ),
]}


def get_subset(name: str) -> LandmarkSubset:
    """Lookup by name with a helpful error."""
    try:
        return SUBSETS[name]
    except KeyError:
        raise KeyError(f"Unknown subset {name!r}; available: {sorted(SUBSETS)}") from None


# sanity: sizes are part of each subset's name/contract
assert len(SUBSETS["FULL_543"]) == 543
assert len(SUBSETS["FP_118"]) == 118
assert len(SUBSETS["ME_126"]) == 126
assert len(SUBSETS["ME_132"]) == 132
assert len(SUBSETS["HANDS_42"]) == 42
assert len(SUBSETS["HANDS_POSE_50"]) == 50
