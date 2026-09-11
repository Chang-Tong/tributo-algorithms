"""Tests for official distributed X-Learner."""

from __future__ import annotations

import numpy as np
import pytest
from tributo_algorithms_causal_xlearner import X_LEARNER_DESCRIPTOR
from tributo_algorithms_causal_xlearner.contracts import XLearnerConfigValidator
from tributo_algorithms_causal_xlearner.exporter import XLearnerONNXExporter
from tributo_algorithms_causal_xlearner.metrics import (
    CATE_COLUMN,
    CONTRIBUTION_COLUMN,
    IDENTITY_COLUMN,
    MU0_COLUMN,
    MU1_COLUMN,
    OUTCOME_COLUMN,
    QUADRANT_COLUMN,
    TREATMENT_COLUMN,
    _build_report,
    evaluate_xlearner_dataset,
)
from tributo_algorithms_causal_xlearner.model import (
    QUADRANT_CODES,
    STAGES,
    XLearnerModel,
)


def test_xlearner_owns_stable_algorithm_spec_and_stages() -> None:
    assert X_LEARNER_DESCRIPTOR.name == "x_learner"
    assert X_LEARNER_DESCRIPTOR.package_name == ("tributo-algorithms-causal-xlearner")
    distribution = X_LEARNER_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.policy.component_stages == STAGES


def test_xlearner_config_contract_requires_five_stage_inputs() -> None:
    value = {
        "data": {
            "feature_columns": ["x0", "x1"],
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "identity_col": "identity",
        },
        "model": {},
        "training": {},
        "ray": {"storage_path": "/tmp/xlearner"},
        "output": {"bundle_uri": "/tmp/xlearner-bundle"},
    }
    assert XLearnerConfigValidator().validate(value) == value


def test_xlearner_onnx_exporter_uses_core_v2_and_runtime_validation() -> None:
    assert XLearnerONNXExporter.api_version == 2
    assert XLearnerONNXExporter.output_format == "onnx"
    assert XLearnerONNXExporter.output_flavor_id == "onnx-runtime-v1"
    assert {
        binding.validator_id for binding in XLearnerONNXExporter.validator_bindings
    } == {
        "structure-v1",
        "onnx-runtime-v1",
    }


class _Booster:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def predict(self, _matrix: object) -> np.ndarray:
        return self.values


def test_xlearner_quadrants_follow_potential_response_semantics() -> None:
    model = XLearnerModel(
        {
            "mu0": _Booster([0.1, 0.8, 0.1, 0.8]),
            "mu1": _Booster([0.8, 0.8, 0.1, 0.1]),
            "tau0": _Booster([0.2, 0.0, 0.0, -0.2]),
            "tau1": _Booster([0.2, 0.0, 0.0, -0.2]),
            "propensity": _Booster([0.5, 0.5, 0.5, 0.5]),
        },
        feature_names=("x0",),
        response_threshold=0.5,
        propensity_clip=(0.01, 0.99),
    )

    result = model.predict(np.arange(4, dtype=np.float32).reshape(-1, 1))

    assert result.quadrant.tolist() == [
        "persuadable",
        "sure_thing",
        "lost_cause",
        "sleeping_dog",
    ]
    assert QUADRANT_CODES == {
        "persuadable": 1,
        "sure_thing": 2,
        "lost_cause": 3,
        "sleeping_dog": 4,
    }


def test_xlearner_report_contains_complete_uplift_evaluation() -> None:
    summary: dict[str, object] = {
        "positive_contribution_count": 1,
        "zero_contribution_count": 2,
        "negative_contribution_count": 1,
    }
    for name, treatment, outcome in (
        ("persuadable", 1, 1),
        ("sure_thing", 0, 0),
        ("lost_cause", 1, 0),
        ("sleeping_dog", 0, 1),
    ):
        summary[f"quadrant.{name}.count"] = 1
        summary[f"quadrant.{name}.treated_count"] = treatment
        summary[f"quadrant.{name}.control_count"] = 1 - treatment
        summary[f"quadrant.{name}.treated_outcome_sum"] = outcome if treatment else 0
        summary[f"quadrant.{name}.control_outcome_sum"] = (
            outcome if not treatment else 0
        )
    prefixes = {
        1: {
            "gain": 1.0,
            "cate_sum": 0.4,
            "treated_count": 1.0,
            "control_count": 0.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 0.0,
        },
        2: {
            "gain": 1.0,
            "cate_sum": 0.6,
            "treated_count": 1.0,
            "control_count": 1.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 0.0,
        },
        3: {
            "gain": 1.0,
            "cate_sum": 0.5,
            "treated_count": 2.0,
            "control_count": 1.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 0.0,
        },
        4: {
            "gain": 0.0,
            "cate_sum": 0.2,
            "treated_count": 2.0,
            "control_count": 2.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 1.0,
        },
    }

    report = _build_report(
        summaries=[summary],
        prefixes=prefixes,
        rows=4,
        treated_rows=2,
        control_rows=2,
        fold_count=2,
        n_points=5,
        n_bins=2,
    )

    assert report["ate"] == 0.05
    assert report["upliftCurve"]["modelGain"] == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert report["evaluation"] == {
        "method": "cross_fitted_oof",
        "sampleSize": 4,
    }
    assert report["calibration"]["points"] == [
        {"predicted": -0.2, "actual": -1.0, "count": 2, "valid": True},
        {"predicted": 0.3, "actual": 1.0, "count": 2, "valid": True},
    ]
    assert [segment["label"] for segment in report["quadrant"]["segments"]] == [
        1,
        2,
        3,
        4,
    ]


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_xlearner_evaluation_reduces_a_partitioned_ray_dataset() -> None:
    import tempfile

    import ray

    with tempfile.TemporaryDirectory(prefix="tributo-ray-", dir="/tmp") as ray_dir:
        ray.init(
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
            _temp_dir=ray_dir,
        )
        try:
            rows = [
                {
                    IDENTITY_COLUMN: 1,
                    TREATMENT_COLUMN: 1,
                    OUTCOME_COLUMN: 1.0,
                    MU0_COLUMN: 0.1,
                    MU1_COLUMN: 0.9,
                    CATE_COLUMN: 0.8,
                    QUADRANT_COLUMN: "persuadable",
                    CONTRIBUTION_COLUMN: 1.0,
                },
                {
                    IDENTITY_COLUMN: 2,
                    TREATMENT_COLUMN: 0,
                    OUTCOME_COLUMN: 0.0,
                    MU0_COLUMN: 0.8,
                    MU1_COLUMN: 0.8,
                    CATE_COLUMN: 0.2,
                    QUADRANT_COLUMN: "sure_thing",
                    CONTRIBUTION_COLUMN: 0.0,
                },
                {
                    IDENTITY_COLUMN: 3,
                    TREATMENT_COLUMN: 1,
                    OUTCOME_COLUMN: 0.0,
                    MU0_COLUMN: 0.1,
                    MU1_COLUMN: 0.1,
                    CATE_COLUMN: -0.1,
                    QUADRANT_COLUMN: "lost_cause",
                    CONTRIBUTION_COLUMN: 0.0,
                },
                {
                    IDENTITY_COLUMN: 4,
                    TREATMENT_COLUMN: 0,
                    OUTCOME_COLUMN: 1.0,
                    MU0_COLUMN: 0.9,
                    MU1_COLUMN: 0.1,
                    CATE_COLUMN: -0.8,
                    QUADRANT_COLUMN: "sleeping_dog",
                    CONTRIBUTION_COLUMN: -1.0,
                },
            ]
            dataset = ray.data.from_items(rows, override_num_blocks=2)

            report = evaluate_xlearner_dataset(
                dataset,
                rows=4,
                treated_rows=2,
                control_rows=2,
                fold_count=2,
                n_points=5,
                n_bins=2,
            )

            assert report["upliftCurve"]["modelGain"] == [
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
            ]
            assert report["evaluation"] == {
                "method": "cross_fitted_oof",
                "sampleSize": 4,
            }
        finally:
            ray.shutdown()
