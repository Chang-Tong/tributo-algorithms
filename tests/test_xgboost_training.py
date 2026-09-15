"""Tests for the public, application-neutral XGBoost training runtime."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost
from tributo_algorithms_boosting import TrainingReporter, run_xgboost_training, stages
from tributo_algorithms_boosting import training as runtime


class _Dataset:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.random_seed: int | None = None

    def count(self) -> int:
        return self.rows

    def random_shuffle(self, seed: int) -> _Dataset:
        self.random_seed = seed
        return self

    def split_proportionately(self, proportions: list[float]) -> tuple[_Dataset, ...]:
        sizes = [round(self.rows * value) for value in proportions]
        sizes.append(self.rows - sum(sizes))
        return tuple(_Dataset(size) for size in sizes)


class _BlockedDataset:
    def __init__(self, blocks: int) -> None:
        self.blocks = blocks
        self.repartition_call: tuple[int, bool, bool] | None = None

    def num_blocks(self) -> int:
        return self.blocks

    def repartition(
        self, blocks: int, *, strict: bool, shuffle: bool
    ) -> _BlockedDataset:
        self.repartition_call = (blocks, strict, shuffle)
        self.blocks = blocks
        return self


class _PredictionDataset:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def map_batches(self, function: Any, **kwargs: Any) -> _PredictionDataset:
        self.frame = function(self.frame, **kwargs["fn_kwargs"])
        return self

    def materialize(self) -> _PredictionDataset:
        return self

    def iter_batches(self, **kwargs: Any) -> Any:
        del kwargs
        yield self.frame

    def sort(self, column: str, *, descending: bool) -> _PredictionDataset:
        return _PredictionDataset(
            self.frame.sort_values(column, ascending=not descending).reset_index(
                drop=True
            )
        )


class _PreparedPredictionDataset(_PredictionDataset):
    def map_batches(self, function: Any, **kwargs: Any) -> _PredictionDataset:
        del function, kwargs
        return self


class _RawBooster:
    def save_raw(self, *, raw_format: str) -> bytes:
        assert raw_format == "ubj"
        return b"prepared-predictions"


class _Reporter:
    def phase(self, name: str) -> None:
        del name

    def metrics(self, item: Any) -> None:
        del item


def test_training_runtime_is_public_and_reporter_is_structural() -> None:
    assert callable(run_xgboost_training)
    assert isinstance(_Reporter(), TrainingReporter)


def test_worker_training_reads_bounded_batches_instead_of_collecting_shards() -> None:
    source = inspect.getsource(runtime._xgboost_train_loop)

    assert ".iter_batches(" in source
    assert ".to_pandas(" not in source
    assert "prefetch_batches=1" in source
    assert "xgboost.DMatrix(train_iter)" in source


def test_stage_runner_standardizes_legacy_argument_to_num_rounds() -> None:
    source = inspect.getsource(stages.XGBoostStageRunner.fit)

    assert '"num_rounds": num_boost_round' in source
    assert '"num_boost_round": num_boost_round' not in source


def test_split_dataset_preserves_all_three_way_rows() -> None:
    dataset = _Dataset(3_000_000)

    _, _, _, rows = runtime._split_dataset(
        dataset,
        {
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "split_strategy": "RANDOM",
            "seed": 42,
        },
    )

    assert dataset.random_seed == 42
    assert rows == {
        "total": 3_000_000,
        "train": 2_100_000,
        "validation": 450_000,
        "test": 450_000,
    }


def test_split_dataset_rejects_lossy_or_empty_split_contracts() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        runtime._split_dataset(
            _Dataset(100),
            {"train_ratio": 0.8, "validation_ratio": 0.2, "test_ratio": 0.2},
        )
    with pytest.raises(ValueError, match="too small"):
        runtime._split_dataset(
            _Dataset(2),
            {"train_ratio": 0.5, "validation_ratio": 0.25, "test_ratio": 0.25},
        )


def test_worker_block_guard_repartitions_for_complete_worker_coverage() -> None:
    dataset = _BlockedDataset(1)

    result = runtime._ensure_worker_blocks(
        dataset, worker_count=2, row_count=100, name="validation"
    )

    assert result is dataset
    assert dataset.repartition_call == (2, True, False)


def test_round_metrics_normalize_error_and_join_train_validation() -> None:
    item = runtime._metrics_item(
        2,
        10,
        {
            "train": {"logloss": [0.6, 0.5], "error": [0.3, 0.2]},
            "eval": {"logloss": [0.65, 0.55], "error": [0.4, 0.25]},
        },
    )

    assert item == {
        "current_round": 2,
        "total_rounds": 10,
        "progress_percent": 20.0,
        "metrics": [
            {"metric_name": "loss", "train": 0.5, "eval": 0.55},
            {"metric_name": "accuracy", "train": 0.8, "eval": 0.75},
        ],
    }


def test_round_metrics_accept_xgboost_standard_deviation_samples() -> None:
    item = runtime._metrics_item(
        1,
        3,
        {"train": {"auc": [(0.75, 0.02)]}, "eval": {"auc": [(0.7, 0.03)]}},
    )

    assert item is not None
    assert item["metrics"] == [{"metric_name": "auc", "train": 0.75, "eval": 0.7}]


def test_early_stopping_callback_saves_best_iteration() -> None:
    features = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    train_labels = (features[:, 0] > 0).astype(np.float32)
    validation_labels = 1.0 - train_labels
    dtrain = xgboost.DMatrix(features, label=train_labels)
    deval = xgboost.DMatrix(features, label=validation_labels)
    history: dict[str, dict[str, list[float]]] = {}

    booster = xgboost.train(
        {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 7},
        dtrain,
        num_boost_round=30,
        evals=[(deval, "eval")],
        evals_result=history,
        verbose_eval=False,
        callbacks=runtime._training_callbacks(
            xgboost, xgboost.callback.TrainingCallback(), 2
        ),
    )

    assert len(history["eval"]["logloss"]) > booster.num_boosted_rounds()
    assert booster.num_boosted_rounds() == booster.best_iteration + 1


@pytest.mark.parametrize(
    ("task_type", "model", "num_class"),
    [
        ("BINARY_CLASSIFICATION", {"objective": "binary:hinge"}, None),
        (
            "MULTICLASS_CLASSIFICATION",
            {"objective": "multi:softmax", "num_class": 3},
            3,
        ),
        ("REGRESSION", {"objective": "binary:logistic"}, None),
    ],
)
def test_task_objective_mismatches_fail_closed(
    task_type: str, model: dict[str, Any], num_class: int | None
) -> None:
    with pytest.raises(ValueError, match="requires"):
        runtime._validated_model_params(
            {"model": model}, task_type=task_type, num_class=num_class
        )


def test_binary_evaluation_uses_every_test_row_and_builds_artifacts() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0] * 10,
            "label": [0, 0, 0, 1, 1, 1] * 10,
        }
    )
    booster = xgboost.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "seed": 7},
        xgboost.DMatrix(frame[["feature_a"]], label=frame["label"]),
        num_boost_round=5,
    )

    metrics, details = runtime._evaluation(
        _PredictionDataset(frame),
        booster,
        feature_names=("feature_a",),
        label_name="label",
        task_type="BINARY_CLASSIFICATION",
        num_class=None,
        artifacts={"roc_curve": True, "threshold_analysis": True},
    )

    assert metrics["auc"] == pytest.approx(1.0)
    assert sum(details["confusion_matrix"].values()) == len(frame)
    assert len(details["roc_curve"]["fpr"]) <= 100
    assert len(details["threshold_analysis"]["thresholds"]) == 19
    assert len(details["threshold_analysis"]["predicted_positive_rows"]) == 19


def test_binary_f1_uses_the_true_harmonic_mean_denominator() -> None:
    predictions = pd.DataFrame(
        {
            "label": [1, 1, 1, 0, 0, 0, 0],
            "probability": [0.9, 0.1, 0.1, 0.8, 0.7, 0.6, 0.1],
        }
    )

    metrics, details = runtime._evaluation(
        _PreparedPredictionDataset(predictions),
        _RawBooster(),
        feature_names=("unused",),
        label_name="label",
        task_type="BINARY_CLASSIFICATION",
        num_class=None,
        artifacts={},
    )

    assert details["confusion_matrix"] == {"tp": 1, "fp": 3, "fn": 2, "tn": 1}
    assert metrics["precision"] == pytest.approx(0.25)
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(2 / 7)


def test_multiclass_evaluation_builds_weighted_metrics_and_matrix() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1] * 10,
            "label": [0, 0, 1, 1, 2, 2] * 10,
        }
    )
    booster = xgboost.train(
        {"objective": "multi:softprob", "num_class": 3, "seed": 7},
        xgboost.DMatrix(frame[["feature_a"]], label=frame["label"]),
        num_boost_round=5,
    )

    metrics, details = runtime._evaluation(
        _PredictionDataset(frame),
        booster,
        feature_names=("feature_a",),
        label_name="label",
        task_type="MULTICLASS_CLASSIFICATION",
        num_class=3,
        artifacts={},
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert details["confusion_matrix"]["labels"] == ["0", "1", "2"]
    assert sum(map(sum, details["confusion_matrix"]["matrix"])) == len(frame)


def test_regression_evaluation_reports_complete_metric_family() -> None:
    frame = pd.DataFrame(
        {"feature_a": np.linspace(1.0, 10.0, 80), "label": np.linspace(2.0, 20.0, 80)}
    )
    booster = xgboost.train(
        {"objective": "reg:squarederror", "seed": 7},
        xgboost.DMatrix(frame[["feature_a"]], label=frame["label"]),
        num_boost_round=20,
    )

    metrics, details = runtime._evaluation(
        _PredictionDataset(frame),
        booster,
        feature_names=("feature_a",),
        label_name="label",
        task_type="REGRESSION",
        num_class=None,
        artifacts={},
    )

    assert set(metrics) == {"rmse", "mae", "mape", "r2"}
    assert metrics["r2"] > 0.95
    assert details == {}


def test_feature_importance_maps_positional_names() -> None:
    features = np.asarray([[0.0, 1.0], [0.1, 1.0], [0.9, 0.0], [1.0, 0.0]] * 10)
    labels = np.asarray([0, 0, 1, 1] * 10)
    booster = xgboost.train(
        {"objective": "binary:logistic", "seed": 7},
        xgboost.DMatrix(features, label=labels),
        num_boost_round=3,
    )

    importance = runtime._feature_importance(
        booster, ("business_feature_a", "business_feature_b")
    )

    assert {item["model_feature_name"] for item in importance} == {
        "business_feature_a",
        "business_feature_b",
    }
    assert importance[0]["importance_score"] > 0


def test_distributed_correlation_reduction_matches_pairwise_expectations() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
        }
    )
    first = runtime._correlation_batch_stats(frame.iloc[:2], list(frame.columns))
    second = runtime._correlation_batch_stats(frame.iloc[2:], list(frame.columns))

    result = runtime._finalize_correlation_stats(
        list(frame.columns),
        [*first.to_dict(orient="records"), *second.to_dict(orient="records")],
    )

    assert result is not None
    assert result["values"][0][1] == pytest.approx(1.0)
    assert result["values"][0][2] == 0.0
    assert result["values"][1][2] == 0.0
    assert [result["values"][index][index] for index in range(3)] == [1.0] * 3


def test_bundle_export_preserves_source_feature_names(tmp_path: Any) -> None:
    frame = pd.DataFrame(
        {"feature_a": [0.0, 0.1, 0.9, 1.0] * 10, "label": [0, 0, 1, 1] * 10}
    )
    booster = xgboost.train(
        {"objective": "binary:logistic", "seed": 7},
        xgboost.DMatrix(frame[["feature_a"]], label=frame["label"]),
        num_boost_round=2,
    )

    result = runtime._export_bundle(
        booster,
        feature_names=("feature_a",),
        bundle_uri=str(tmp_path / "bundle"),
        run_id="official-feature-export",
    )

    assert result["bundle_id"]
    assert result["manifest_sha256"]
    assert booster.feature_names == ["feature_a"]
