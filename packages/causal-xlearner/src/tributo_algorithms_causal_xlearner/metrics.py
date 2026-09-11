"""Distributed, bounded-memory evaluation for X-Learner predictions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from tributo_algorithms_causal_xlearner.model import QUADRANT_CODES

IDENTITY_COLUMN = "__tributo_xlearner_identity"
TREATMENT_COLUMN = "__tributo_xlearner_treatment"
OUTCOME_COLUMN = "__tributo_xlearner_outcome"
MU0_COLUMN = "mu0"
MU1_COLUMN = "mu1"
CATE_COLUMN = "cate"
QUADRANT_COLUMN = "quadrant"
CONTRIBUTION_COLUMN = "__tributo_xlearner_contribution"

QUADRANT_ORDER = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")
QUADRANT_META: Mapping[str, Mapping[str, str]] = {
    "persuadable": {
        "name": "intervention_sensitive",
        "description": "responds only when treated",
        "action": "prioritize_treatment",
    },
    "sure_thing": {
        "name": "natural_responder",
        "description": "responds with or without treatment",
        "action": "avoid_unnecessary_treatment",
    },
    "lost_cause": {
        "name": "non_responder",
        "description": "responds under neither condition",
        "action": "do_not_prioritize",
    },
    "sleeping_dog": {
        "name": "negative_responder",
        "description": "responds only without treatment",
        "action": "avoid_treatment",
    },
}

_PREFIX_FIELDS = (
    "gain",
    "cate_sum",
    "treated_count",
    "control_count",
    "treated_outcome_sum",
    "control_outcome_sum",
)


def _python_scalar(value: object) -> object:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _block_key(frame: Any) -> str:
    first = frame.iloc[0]
    last = frame.iloc[-1]
    payload = (
        _python_scalar(first[CATE_COLUMN]),
        _python_scalar(first[IDENTITY_COLUMN]),
        _python_scalar(last[CATE_COLUMN]),
        _python_scalar(last[IDENTITY_COLUMN]),
        int(len(frame)),
    )
    return hashlib.sha256(
        json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _summary_batch(batch: object) -> object:
    import pandas as pd

    frame = cast(Any, batch)
    if frame.empty:
        return pd.DataFrame()
    treatment = frame[TREATMENT_COLUMN].to_numpy(dtype=np.int8)
    outcome = frame[OUTCOME_COLUMN].to_numpy(dtype=np.float64)
    contribution = frame[CONTRIBUTION_COLUMN].to_numpy(dtype=np.float64)
    row: dict[str, object] = {
        "block_key": _block_key(frame),
        "first_cate": float(frame[CATE_COLUMN].iloc[0]),
        "first_identity": _python_scalar(frame[IDENTITY_COLUMN].iloc[0]),
        "count": int(len(frame)),
        "gain": float(contribution.sum()),
        "cate_sum": float(frame[CATE_COLUMN].sum()),
        "treated_count": int((treatment == 1).sum()),
        "control_count": int((treatment == 0).sum()),
        "treated_outcome_sum": float(outcome[treatment == 1].sum()),
        "control_outcome_sum": float(outcome[treatment == 0].sum()),
        "positive_contribution_count": int((contribution > 0).sum()),
        "zero_contribution_count": int((contribution == 0).sum()),
        "negative_contribution_count": int((contribution < 0).sum()),
    }
    quadrants = frame[QUADRANT_COLUMN].to_numpy()
    for name in QUADRANT_ORDER:
        mask = quadrants == name
        treated_mask = mask & (treatment == 1)
        control_mask = mask & (treatment == 0)
        row[f"quadrant.{name}.count"] = int(mask.sum())
        row[f"quadrant.{name}.treated_count"] = int(treated_mask.sum())
        row[f"quadrant.{name}.control_count"] = int(control_mask.sum())
        row[f"quadrant.{name}.treated_outcome_sum"] = float(outcome[treated_mask].sum())
        row[f"quadrant.{name}.control_outcome_sum"] = float(outcome[control_mask].sum())
    return pd.DataFrame([row])


def _identity_sort_key(value: object) -> tuple[int, object]:
    value = _python_scalar(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (float, np.floating)):
        return (1, float(value))
    if isinstance(value, bytes):
        return (2, value)
    return (3, str(value))


def _prefix_batch(
    batch: object,
    *,
    plans: Mapping[str, Mapping[str, object]],
) -> object:
    import pandas as pd

    frame = cast(Any, batch)
    columns = ("rank", *_PREFIX_FIELDS)
    if frame.empty:
        return pd.DataFrame({name: [] for name in columns})
    plan = plans.get(_block_key(frame))
    if plan is None:
        return pd.DataFrame({name: [] for name in columns})

    targets = tuple(_as_int(value) for value in cast(list[object], plan["local_ranks"]))
    base_rank = _as_int(plan["base_rank"])
    contribution = frame[CONTRIBUTION_COLUMN].to_numpy(dtype=np.float64)
    cate = frame[CATE_COLUMN].to_numpy(dtype=np.float64)
    treatment = frame[TREATMENT_COLUMN].to_numpy(dtype=np.int8)
    outcome = frame[OUTCOME_COLUMN].to_numpy(dtype=np.float64)
    prefixes = {
        "gain": np.cumsum(contribution),
        "cate_sum": np.cumsum(cate),
        "treated_count": np.cumsum(treatment == 1),
        "control_count": np.cumsum(treatment == 0),
        "treated_outcome_sum": np.cumsum(outcome * (treatment == 1)),
        "control_outcome_sum": np.cumsum(outcome * (treatment == 0)),
    }
    result: dict[str, list[object]] = {name: [] for name in columns}
    for local_rank in targets:
        index = local_rank - 1
        result["rank"].append(base_rank + local_rank)
        for field in _PREFIX_FIELDS:
            result[field].append(
                _as_float(plan[f"base_{field}"]) + prefixes[field][index]
            )
    return pd.DataFrame(result)


def _area(values: list[float]) -> float:
    return sum(
        (values[index] + values[index + 1]) / 2.0 for index in range(len(values) - 1)
    )


def _segment_bounds(model_gain: list[float], x_axis: list[float]) -> list[float]:
    peak_index = int(np.argmax(model_gain))
    right = x_axis[peak_index]
    peak_x = x_axis[peak_index]
    peak_gain = model_gain[peak_index]
    if peak_x <= 0 or peak_gain <= 0:
        return [float(right), float(right)]
    distances = [
        model_gain[index] / peak_gain - x_axis[index] / peak_x
        for index in range(peak_index + 1)
    ]
    return [float(x_axis[int(np.argmax(distances))]), float(right)]


def _prefix_at(
    prefixes: Mapping[int, Mapping[str, float]], rank: int, field: str
) -> float:
    if rank == 0:
        return 0.0
    return float(prefixes[rank][field])


def _build_report(
    *,
    summaries: list[Mapping[str, object]],
    prefixes: Mapping[int, Mapping[str, float]],
    rows: int,
    treated_rows: int,
    control_rows: int,
    fold_count: int,
    n_points: int,
    n_bins: int,
) -> dict[str, object]:
    ratio = treated_rows / control_rows
    x_axis = [index / (n_points - 1) for index in range(n_points)]
    curve_ranks = [int(rows * value) for value in x_axis]
    model_gain = [_prefix_at(prefixes, rank, "gain") for rank in curve_ranks]
    positive = sum(_as_int(item["positive_contribution_count"]) for item in summaries)
    zero = sum(_as_int(item["zero_contribution_count"]) for item in summaries)
    perfect_gain = [
        float(min(rank, positive) - ratio * max(0, rank - positive - zero))
        for rank in curve_ranks
    ]
    total_gain = model_gain[-1]
    random_gain = [total_gain * value for value in x_axis]
    random_area = _area(random_gain)
    perfect_denom = _area(perfect_gain) - random_area
    qini = (
        (_area(model_gain) - random_area) / perfect_denom if perfect_denom != 0 else 0.0
    )
    auuc = _area(model_gain) / ((n_points - 1) * total_gain) if total_gain != 0 else 0.5

    bins = min(n_bins, rows)
    bin_size = rows // bins
    ascending_bounds = [index * bin_size for index in range(bins)] + [rows]
    calibration: list[dict[str, object]] = []
    for start, end in zip(ascending_bounds, ascending_bounds[1:]):
        descending_start = rows - end
        descending_end = rows - start

        interval = {
            field: _prefix_at(prefixes, descending_end, field)
            - _prefix_at(prefixes, descending_start, field)
            for field in _PREFIX_FIELDS
        }

        treated = int(round(interval["treated_count"]))
        control = int(round(interval["control_count"]))
        predicted = interval["cate_sum"] / (end - start)
        valid = treated > 0 and control > 0
        actual = (
            interval["treated_outcome_sum"] / treated
            - interval["control_outcome_sum"] / control
            if valid
            else None
        )
        calibration.append(
            {
                "predicted": round(float(predicted), 4),
                "actual": None if actual is None else round(float(actual), 4),
                "count": end - start,
                "valid": valid,
            }
        )

    quadrant_segments = []
    for name in QUADRANT_ORDER:
        count = sum(_as_int(item[f"quadrant.{name}.count"]) for item in summaries)
        treated = sum(
            _as_int(item[f"quadrant.{name}.treated_count"]) for item in summaries
        )
        control = sum(
            _as_int(item[f"quadrant.{name}.control_count"]) for item in summaries
        )
        treated_outcome = sum(
            _as_float(item[f"quadrant.{name}.treated_outcome_sum"])
            for item in summaries
        )
        control_outcome = sum(
            _as_float(item[f"quadrant.{name}.control_outcome_sum"])
            for item in summaries
        )
        t1_rate = treated_outcome / treated if treated else None
        t0_rate = control_outcome / control if control else None
        effect = (
            t1_rate - t0_rate if t1_rate is not None and t0_rate is not None else None
        )
        quadrant_segments.append(
            {
                "type": name,
                **dict(QUADRANT_META[name]),
                "label": QUADRANT_CODES[name],
                "count": count,
                "percentage": round(count / rows, 3),
                "t0Rate": None if t0_rate is None else round(t0_rate, 4),
                "t1Rate": None if t1_rate is None else round(t1_rate, 4),
                "treatmentEffect": None if effect is None else round(effect, 4),
            }
        )

    ate = _prefix_at(prefixes, rows, "cate_sum") / rows
    return {
        "ate": round(float(ate), 4),
        "qini": round(float(qini), 4),
        "auuc": round(float(auuc), 4),
        "treated_rows": treated_rows,
        "control_rows": control_rows,
        "cross_fit_folds": fold_count,
        "evaluation": {
            "method": "cross_fitted_oof",
            "sampleSize": rows,
        },
        "quadrant": {"total": rows, "segments": quadrant_segments},
        "upliftCurve": {
            "xAxis": x_axis,
            "modelGain": [round(float(value), 4) for value in model_gain],
            "randomGain": [round(float(value), 4) for value in random_gain],
            "perfectGain": [round(float(value), 4) for value in perfect_gain],
            "segmentBounds": _segment_bounds(model_gain, x_axis),
        },
        "calibration": {"points": calibration},
    }


def evaluate_xlearner_dataset(
    dataset: object,
    *,
    rows: int,
    treated_rows: int,
    control_rows: int,
    fold_count: int,
    n_points: int = 21,
    n_bins: int = 10,
) -> Mapping[str, object]:
    """Evaluate cross-fitted predictions with distributed sorting and reductions.

    Only one bounded record per Ray block and the requested curve/calibration
    prefixes are returned to the Driver. Prediction rows remain in Ray's object
    store and spill layer.
    """
    if rows < 1 or treated_rows < 1 or control_rows < 1:
        raise ValueError("X-Learner evaluation requires both treatment groups")
    if n_points < 2 or n_bins < 1:
        raise ValueError("invalid X-Learner evaluation resolution")

    scored = (
        cast(Any, dataset)
        .sort([CATE_COLUMN, IDENTITY_COLUMN], descending=[True, False])
        .materialize()
    )
    summaries = cast(
        list[Mapping[str, object]],
        scored.map_batches(
            _summary_batch,
            batch_format="pandas",
            batch_size=None,
        ).take_all(),
    )
    if not summaries or sum(_as_int(item["count"]) for item in summaries) != rows:
        raise RuntimeError("X-Learner metric reduction lost prediction rows")
    summaries.sort(
        key=lambda item: (
            -_as_float(item["first_cate"]),
            _identity_sort_key(item["first_identity"]),
        )
    )

    x_axis = [index / (n_points - 1) for index in range(n_points)]
    target_ranks = {int(rows * value) for value in x_axis}
    bins = min(n_bins, rows)
    bin_size = rows // bins
    ascending_bounds = [index * bin_size for index in range(bins)] + [rows]
    target_ranks.update(rows - value for value in ascending_bounds)
    target_ranks.discard(0)

    plans: dict[str, dict[str, object]] = {}
    base_rank = 0
    base_values: dict[str, float] = dict.fromkeys(_PREFIX_FIELDS, 0.0)
    for summary in summaries:
        count = _as_int(summary["count"])
        local_ranks = sorted(
            rank - base_rank
            for rank in target_ranks
            if base_rank < rank <= base_rank + count
        )
        plans[str(summary["block_key"])] = {
            "base_rank": base_rank,
            "local_ranks": local_ranks,
            **{f"base_{field}": base_values[field] for field in _PREFIX_FIELDS},
        }
        base_rank += count
        for field in _PREFIX_FIELDS:
            base_values[field] += _as_float(summary[field])

    prefix_rows = cast(
        list[Mapping[str, object]],
        scored.map_batches(
            _prefix_batch,
            batch_format="pandas",
            batch_size=None,
            fn_kwargs={"plans": plans},
        ).take_all(),
    )
    prefixes = {
        _as_int(item["rank"]): {
            field: _as_float(item[field]) for field in _PREFIX_FIELDS
        }
        for item in prefix_rows
    }
    if set(prefixes) != target_ranks:
        raise RuntimeError("X-Learner metric reduction missed requested prefixes")
    return _build_report(
        summaries=summaries,
        prefixes=prefixes,
        rows=rows,
        treated_rows=treated_rows,
        control_rows=control_rows,
        fold_count=fold_count,
        n_points=n_points,
        n_bins=n_bins,
    )


__all__ = [
    "CATE_COLUMN",
    "CONTRIBUTION_COLUMN",
    "IDENTITY_COLUMN",
    "MU0_COLUMN",
    "MU1_COLUMN",
    "OUTCOME_COLUMN",
    "QUADRANT_COLUMN",
    "TREATMENT_COLUMN",
    "evaluate_xlearner_dataset",
]
