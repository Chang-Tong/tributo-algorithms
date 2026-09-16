"""High-level distributed XGBoost training built on public Ray APIs.

The entry point in this module is intentionally application-neutral.  Protocol
adapters can translate their request into Tributo's ingestion contract, call
``run_xgboost_training``, and translate the returned ``AlgorithmRunResult``.
All training, evaluation, evidence, and Bundle production stay in the official
algorithm package.
"""

from __future__ import annotations

import hashlib
import math
import queue as stdlib_queue
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from tributo.algorithms import (
    AlgorithmExecutionResult,
    AlgorithmRunResult,
    WorkerResources,
)
from tributo.data import IngestionGateway, IngestionRequest, RayDataHandle

from tributo_algorithms_boosting.data_config import CompleteCoverageDataConfig

_METRIC_NAMES = {
    "logloss": "loss",
    "mlogloss": "loss",
    "error": "accuracy",
    "merror": "accuracy",
    "aucpr": "average_precision",
}
_ERROR_METRICS = frozenset({"error", "merror"})
_DEFAULT_BATCH_ROWS = 65_536
_MetricSample = float | tuple[float, float]
_MetricHistory = Mapping[str, Mapping[str, Sequence[_MetricSample]]]


@runtime_checkable
class TrainingReporter(Protocol):
    """Optional, minimal callback surface for live training progress."""

    def phase(self, name: str) -> None:
        """Report a lifecycle phase change."""

    def metrics(self, item: Mapping[str, Any]) -> None:
        """Report one normalized round-metrics item."""


class _NullReporter:
    def phase(self, name: str) -> None:
        del name

    def metrics(self, item: Mapping[str, Any]) -> None:
        del item


class _EvidenceCollector:
    def __init__(self) -> None:
        self._records: dict[int, dict[str, Any]] = {}

    def record(self, value: dict[str, Any]) -> None:
        rank = value.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("XGBoost evidence rank must be an integer")
        if rank in self._records:
            raise ValueError(f"duplicate XGBoost evidence rank: {rank}")
        self._records[rank] = dict(value)

    def snapshot(self) -> list[dict[str, Any]]:
        return [self._records[index] for index in sorted(self._records)]


def _metric_value(raw_name: str, value: object) -> float:
    raw_value = value[0] if isinstance(value, tuple) else value
    result = float(cast(Any, raw_value))
    if raw_name in _ERROR_METRICS:
        result = 1.0 - result
    if not math.isfinite(result):
        raise ValueError("XGBoost produced a non-finite metric")
    return result


def _metrics_item(
    round_number: int,
    total_rounds: int,
    evals_log: _MetricHistory,
) -> dict[str, Any] | None:
    by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for scope, raw_metrics in evals_log.items():
        for raw_name, values in raw_metrics.items():
            if round_number > len(values):
                continue
            try:
                value = _metric_value(raw_name, values[round_number - 1])
            except (TypeError, ValueError):
                continue
            name = _METRIC_NAMES.get(raw_name, raw_name)
            if name not in by_name:
                by_name[name] = {"metric_name": name}
                order.append(name)
            by_name[name]["train" if scope == "train" else "eval"] = value
    metrics = [by_name[name] for name in order]
    if not metrics:
        return None
    return {
        "current_round": round_number,
        "total_rounds": total_rounds,
        "progress_percent": round(round_number / total_rounds * 100.0, 1),
        "metrics": metrics,
    }


def _training_callbacks(
    xgboost_module: Any,
    round_metrics_callback: Any,
    early_stopping_rounds: object,
) -> list[Any]:
    callbacks = [round_metrics_callback]
    if early_stopping_rounds is not None:
        callbacks.append(
            xgboost_module.callback.EarlyStopping(
                rounds=int(cast(Any, early_stopping_rounds)),
                data_name="eval",
                save_best=True,
            )
        )
    return callbacks


def _xgboost_train_loop(config: dict[str, Any]) -> None:
    """Train a synchronized booster from replayable, bounded Ray batches."""
    import gc
    import json
    import os

    import ray
    import xgboost
    from ray import train
    from ray.train import Checkpoint

    feature_names = list(config["feature_names"])
    label_name = str(config["label_name"])
    batch_rows = int(config.get("batch_rows", _DEFAULT_BATCH_ROWS))

    class _RayDatasetDataIter(xgboost.DataIter):
        """Replay a Ray Train shard without collecting it into one DataFrame."""

        def __init__(self, dataset_name: str, cache_prefix: str) -> None:
            super().__init__(cache_prefix=cache_prefix, release_data=True)
            self._dataset_name = dataset_name
            self._iterator: Any = None

        def reset(self) -> None:
            shard = train.get_dataset_shard(self._dataset_name)
            self._iterator = iter(
                shard.iter_batches(
                    batch_format="pandas",
                    batch_size=batch_rows,
                    prefetch_batches=1,
                )
            )

        def next(self, input_data: Any) -> bool:
            if self._iterator is None:
                self.reset()
            try:
                frame = next(self._iterator)
            except StopIteration:
                return False
            if frame.empty:
                return self.next(input_data)
            input_data(
                data=frame[feature_names].to_numpy(),
                label=frame[label_name].to_numpy(),
            )
            return True

    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    rounds_value = config.get("num_rounds", config.get("num_boost_round"))
    if (
        not isinstance(rounds_value, int)
        or isinstance(rounds_value, bool)
        or rounds_value < 1
    ):
        raise ValueError("num_rounds must be a positive integer")
    total_rounds = rounds_value
    interval = max(1, math.ceil(total_rounds / 100))
    metrics_queue = config.get("metrics_queue")

    class _RoundMetricsCallback(xgboost.callback.TrainingCallback):
        def __init__(self) -> None:
            self.published: set[int] = set()
            self.last_round = 0
            self.last_log: _MetricHistory = {}

        def after_iteration(
            self,
            model: xgboost.Booster,
            epoch: int,
            evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]],
        ) -> bool:
            del model
            self.last_round = epoch + 1
            if evals_log:
                self.last_log = evals_log
            if (
                metrics_queue is not None
                and rank == 0
                and (
                    self.last_round == 1
                    or self.last_round % interval == 0
                    or self.last_round == total_rounds
                )
            ):
                item = _metrics_item(self.last_round, total_rounds, evals_log)
                if item is not None:
                    with suppress(Exception):
                        metrics_queue.put_nowait(item)
                        self.published.add(self.last_round)
            return False

        def after_training(self, model: xgboost.Booster) -> xgboost.Booster:
            if (
                metrics_queue is not None
                and rank == 0
                and self.last_round not in self.published
            ):
                item = _metrics_item(self.last_round, total_rounds, self.last_log)
                if item is not None:
                    with suppress(Exception):
                        metrics_queue.put_nowait(item)
            return model

    with tempfile.TemporaryDirectory(prefix="tributo-xgboost-pages-") as cache_root:
        train_iter = _RayDatasetDataIter("train", f"{cache_root}/train")
        dtrain = xgboost.DMatrix(train_iter)
        if int(dtrain.num_row()) < 1:
            raise RuntimeError("XGBoost training shard is empty")
        evals: list[tuple[Any, str]] = [(dtrain, "train")]
        try:
            validation_shard = train.get_dataset_shard("validation")
        except KeyError:
            validation_shard = None
        validation_iter = None
        dvalidation = None
        if validation_shard is not None:
            validation_iter = _RayDatasetDataIter(
                "validation", f"{cache_root}/validation"
            )
            dvalidation = xgboost.DMatrix(validation_iter)
            if int(dvalidation.num_row()) > 0:
                evals.append((dvalidation, "eval"))

        early_stopping_rounds = config.get("early_stopping_rounds")
        callbacks = _training_callbacks(
            xgboost, _RoundMetricsCallback(), early_stopping_rounds
        )
        histories: dict[str, dict[str, list[float] | list[tuple[float, float]]]] = {}
        booster = xgboost.train(
            dict(config["params"]),
            dtrain,
            num_boost_round=total_rounds,
            evals=evals,
            evals_result=histories,
            verbose_eval=False,
            callbacks=callbacks,
        )
        rows_processed = int(dtrain.num_row())
        validation_rows_processed = (
            int(dvalidation.num_row()) if dvalidation is not None else 0
        )
        actual_rounds = max(
            (
                len(values)
                for scope_metrics in histories.values()
                for values in scope_metrics.values()
            ),
            default=0,
        )
        del evals, dtrain, train_iter, dvalidation, validation_iter
        gc.collect()

    raw = bytes(booster.save_raw(raw_format="ubj"))
    digest = hashlib.sha256(raw).hexdigest()
    runtime = ray.get_runtime_context()
    assigned = runtime.get_assigned_resources()
    evidence = {
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "rows_processed": rows_processed,
        "input_rows": {
            "train": rows_processed,
            "validation": validation_rows_processed,
        },
        "batch_count": actual_rounds,
        "collective_steps": actual_rounds,
        "model_state_digest": digest,
        "resources": {
            "num_cpus": float(assigned.get("CPU", 0.0)),
            "num_gpus": float(assigned.get("GPU", 0.0)),
            "custom": {
                str(name): float(value)
                for name, value in assigned.items()
                if name not in {"CPU", "GPU", "memory", "object_store_memory"}
            },
        },
    }
    binding_digest = config.get("binding_digest")
    if isinstance(binding_digest, str) and binding_digest:
        evidence["shard_id"] = hashlib.sha256(
            f"{binding_digest}:{rank}/{world_size}".encode("ascii")
        ).hexdigest()
    ray.get(config["evidence_actor"].record.remote(evidence))
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="tributo-xgboost-checkpoint-") as root:
            path = Path(root)
            booster.save_model(path / "model.ubj")
            (path / "feature_names.json").write_text(
                json.dumps(feature_names), encoding="utf-8"
            )
            train.report(
                {
                    "metric_history": histories,
                    "model_state_digest": digest,
                    "worker_pid": os.getpid(),
                },
                checkpoint=Checkpoint.from_directory(path),
            )
    else:
        train.report({"model_state_digest": digest})


class _MetricsPump:
    def __init__(
        self, metrics_queue: Any, reporter: TrainingReporter, rounds: int
    ) -> None:
        self._queue = metrics_queue
        self._reporter = reporter
        self._rounds = rounds
        self._stop = threading.Event()
        self._published: set[int] = set()
        self._thread = threading.Thread(
            target=self._run,
            name="tributo-xgboost-training-metrics",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _publish(self, item: object) -> None:
        if not isinstance(item, Mapping):
            return
        current = item.get("current_round")
        metrics = item.get("metrics")
        if (
            not isinstance(current, int)
            or current < 1
            or current > self._rounds
            or current in self._published
            or not isinstance(metrics, list)
            or not metrics
        ):
            return
        self._reporter.metrics(dict(item))
        self._published.add(current)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._publish(self._queue.get(timeout=0.2))
            except stdlib_queue.Empty:
                continue
            except Exception:
                if not self._stop.is_set():
                    self._stop.wait(0.2)

    def finish(self, histories: _MetricHistory) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        while True:
            try:
                self._publish(self._queue.get_nowait())
            except Exception:
                break
        actual_rounds = max(
            (
                len(values)
                for metrics in histories.values()
                for values in metrics.values()
            ),
            default=0,
        )
        for round_number in (1, actual_rounds):
            if round_number < 1 or round_number in self._published:
                continue
            item = _metrics_item(round_number, self._rounds, histories)
            if item is not None:
                self._publish(item)
        with suppress(Exception):
            self._queue.shutdown(force=True)


def _split_dataset(
    dataset: Any,
    training: Mapping[str, Any],
) -> tuple[Any, Any | None, Any | None, dict[str, int]]:
    total = int(dataset.count())
    train_ratio = float(training.get("train_ratio", 0.7))
    validation_ratio = float(training.get("validation_ratio", 0.0))
    test_ratio = float(training.get("test_ratio", 0.3))
    if total < 1:
        raise ValueError("training dataset is empty")
    if any(
        not math.isfinite(value) or value < 0 or value > 1
        for value in (train_ratio, validation_ratio, test_ratio)
    ) or not math.isclose(
        train_ratio + validation_ratio + test_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("data split ratios must be within [0,1] and sum to 1")
    if train_ratio <= 0:
        raise ValueError("training.train_ratio must be positive")
    strategy = str(training.get("split_strategy", "RANDOM")).upper()
    if strategy not in {"RANDOM", "TIME_ORDERED"}:
        raise ValueError("training.split_strategy must be RANDOM or TIME_ORDERED")
    source = (
        dataset.randomize_block_order(seed=int(training.get("seed", 42)))
        if strategy == "RANDOM"
        else dataset
    )
    if validation_ratio > 0 and test_ratio > 0:
        train_dataset, validation_dataset, test_dataset = source.split_proportionately(
            [train_ratio, validation_ratio]
        )
    elif validation_ratio > 0:
        train_dataset, validation_dataset = source.split_proportionately([train_ratio])
        test_dataset = None
    elif test_ratio > 0:
        train_dataset, test_dataset = source.split_proportionately([train_ratio])
        validation_dataset = None
    else:
        train_dataset, validation_dataset, test_dataset = source, None, None
    rows = {
        "total": total,
        "train": int(train_dataset.count()),
        "validation": (
            int(validation_dataset.count()) if validation_dataset is not None else 0
        ),
        "test": int(test_dataset.count()) if test_dataset is not None else 0,
    }
    if sum(rows[name] for name in ("train", "validation", "test")) != total:
        raise RuntimeError("training split did not preserve every input row")
    if any(
        ratio > 0 and rows[name] == 0
        for name, ratio in (
            ("train", train_ratio),
            ("validation", validation_ratio),
            ("test", test_ratio),
        )
    ):
        raise ValueError("training dataset is too small for the requested split")
    return train_dataset, validation_dataset, test_dataset, rows


def _ensure_worker_blocks(
    dataset: Any,
    *,
    worker_count: int,
    row_count: int,
    name: str,
) -> Any:
    if row_count < worker_count:
        raise ValueError(f"{name} dataset has fewer rows than XGBoost workers")
    try:
        block_count = dataset.num_blocks()
    except NotImplementedError:
        dataset = dataset.materialize()
        block_count = dataset.num_blocks()
    if block_count < worker_count:
        dataset = dataset.repartition(worker_count, strict=True, shuffle=False)
    return dataset


def _load_booster(checkpoint: Any, feature_names: tuple[str, ...]) -> Any:
    import xgboost

    if checkpoint is None:
        raise RuntimeError("XGBoost training returned no checkpoint")
    with checkpoint.as_directory() as directory:
        booster = xgboost.Booster()
        booster.load_model(Path(directory) / "model.ubj")
    booster.feature_names = list(feature_names)
    return booster


def _predict_batch(
    batch: Any,
    *,
    booster_raw: bytes,
    feature_names: list[str],
    label_name: str,
    task_type: str,
    num_class: int | None,
) -> Any:
    import pandas as pd
    import xgboost

    booster = xgboost.Booster()
    booster.load_model(bytearray(booster_raw))
    values = booster.predict(xgboost.DMatrix(batch[feature_names]))
    result: dict[str, Any] = {"label": batch[label_name].to_numpy()}
    if task_type == "MULTICLASS_CLASSIFICATION":
        classes = int(num_class or values.shape[1])
        for index in range(classes):
            result[f"probability_{index}"] = values[:, index]
    elif task_type == "BINARY_CLASSIFICATION":
        result["probability"] = values
    else:
        result["prediction"] = values
    return pd.DataFrame(result)


def _prediction_batches(dataset: Any) -> Iterator[Any]:
    yield from dataset.iter_batches(
        batch_format="pandas",
        batch_size=_DEFAULT_BATCH_ROWS,
        prefetch_batches=1,
    )


def _binary_ranking_metrics(
    predictions: Any,
    *,
    probability_column: str,
    positive_label: int,
    total_rows: int,
    total_positive: int,
    include_curve: bool,
) -> tuple[float | None, float | None, dict[str, list[float]] | None]:
    """Compute exact AUC/AP from a distributed score sort using O(1) state."""
    if total_positive < 1 or total_positive >= total_rows:
        return None, None, None
    total_negative = total_rows - total_positive
    true_positive = 0
    false_positive = 0
    previous_tpr = 0.0
    previous_fpr = 0.0
    auc = 0.0
    average_precision = 0.0
    pending_score: float | None = None
    pending_positive = 0
    pending_negative = 0
    curve_fpr = [0.0]
    curve_tpr = [0.0]
    curve_targets = (
        [math.ceil(total_rows * index / 99) for index in range(1, 100)]
        if include_curve
        else []
    )
    target_index = 0

    def _flush_group() -> None:
        nonlocal auc, average_precision, false_positive, previous_fpr
        nonlocal previous_tpr, target_index, true_positive
        if pending_score is None:
            return
        true_positive += pending_positive
        false_positive += pending_negative
        tpr = true_positive / total_positive
        fpr = false_positive / total_negative
        auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        average_precision += (
            (tpr - previous_tpr) * true_positive / (true_positive + false_positive)
        )
        previous_tpr = tpr
        previous_fpr = fpr
        consumed = true_positive + false_positive
        while (
            target_index < len(curve_targets)
            and consumed >= curve_targets[target_index]
        ):
            curve_fpr.append(fpr)
            curve_tpr.append(tpr)
            target_index += 1

    ordered = predictions.sort(probability_column, descending=True)
    for frame in _prediction_batches(ordered):
        labels = frame["label"].to_numpy()
        scores = frame[probability_column].to_numpy()
        for label, raw_score in zip(labels, scores, strict=True):
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError("XGBoost prediction contains a non-finite score")
            if pending_score is not None and score != pending_score:
                _flush_group()
                pending_positive = 0
                pending_negative = 0
            pending_score = score
            if int(label) == positive_label:
                pending_positive += 1
            else:
                pending_negative += 1
    _flush_group()
    curve = None
    if include_curve:
        if curve_fpr[-1] != 1.0 or curve_tpr[-1] != 1.0:
            curve_fpr.append(1.0)
            curve_tpr.append(1.0)
        curve = {"fpr": curve_fpr[:100], "tpr": curve_tpr[:100]}
    return auc, average_precision, curve


def _evaluation(
    dataset: Any | None,
    booster: Any,
    *,
    feature_names: tuple[str, ...],
    label_name: str,
    task_type: str,
    num_class: int | None,
    artifacts: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    if dataset is None:
        return {}, {}
    import numpy as np

    raw = bytes(booster.save_raw(raw_format="ubj"))
    predictions = dataset.map_batches(
        _predict_batch,
        batch_format="pandas",
        batch_size=_DEFAULT_BATCH_ROWS,
        fn_kwargs={
            "booster_raw": raw,
            "feature_names": list(feature_names),
            "label_name": label_name,
            "task_type": task_type,
            "num_class": num_class,
        },
    ).materialize()
    metrics: dict[str, float]
    details: dict[str, Any] = {}
    if task_type == "BINARY_CLASSIFICATION":
        thresholds = np.linspace(0.05, 0.95, 19)
        threshold_positive = np.zeros(len(thresholds), dtype=np.int64)
        threshold_true_positive = np.zeros(len(thresholds), dtype=np.int64)
        total_rows = 0
        total_positive = 0
        true_positive = false_positive = false_negative = true_negative = 0
        for frame in _prediction_batches(predictions):
            labels = frame["label"].to_numpy().astype(np.int64, copy=False)
            probability = frame["probability"].to_numpy(dtype=np.float64)
            if not np.isfinite(probability).all():
                raise ValueError("XGBoost prediction contains a non-finite score")
            positive = labels == 1
            predicted = probability >= 0.5
            total_rows += int(len(labels))
            total_positive += int(positive.sum())
            true_positive += int((predicted & positive).sum())
            false_positive += int((predicted & ~positive).sum())
            false_negative += int((~predicted & positive).sum())
            true_negative += int((~predicted & ~positive).sum())
            by_threshold = probability[:, None] >= thresholds
            threshold_positive += by_threshold.sum(axis=0)
            threshold_true_positive += (by_threshold & positive[:, None]).sum(axis=0)
        if total_rows < 1:
            raise ValueError("test dataset is empty")
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1_denominator = precision + recall
        f1 = 2.0 * precision * recall / f1_denominator if f1_denominator > 0.0 else 0.0
        metrics = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": (true_positive + true_negative) / total_rows,
        }
        auc, average_precision, curve = _binary_ranking_metrics(
            predictions,
            probability_column="probability",
            positive_label=1,
            total_rows=total_rows,
            total_positive=total_positive,
            include_curve=bool(artifacts.get("roc_curve", False)),
        )
        if auc is not None:
            metrics["auc"] = auc
        if average_precision is not None:
            metrics["average_precision"] = average_precision
        details["confusion_matrix"] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "tn": true_negative,
        }
        if curve is not None:
            details["roc_curve"] = curve
        if artifacts.get("threshold_analysis", False):
            precision_values = np.divide(
                threshold_true_positive,
                threshold_positive,
                out=np.zeros(len(thresholds), dtype=np.float64),
                where=threshold_positive > 0,
            )
            recall_values = threshold_true_positive / max(1, total_positive)
            f1_values = np.divide(
                2.0 * precision_values * recall_values,
                precision_values + recall_values,
                out=np.zeros(len(thresholds), dtype=np.float64),
                where=(precision_values + recall_values) > 0,
            )
            details["threshold_analysis"] = {
                "thresholds": [round(float(value), 2) for value in thresholds],
                "precision_values": precision_values.tolist(),
                "recall_values": recall_values.tolist(),
                "f1_values": f1_values.tolist(),
                "predicted_positive_rows": threshold_positive.tolist(),
            }
    elif task_type == "MULTICLASS_CLASSIFICATION":
        classes = int(num_class or 0)
        probability_columns = [f"probability_{index}" for index in range(classes)]
        matrix = np.zeros((classes, classes), dtype=np.int64)
        for frame in _prediction_batches(predictions):
            labels = frame["label"].to_numpy().astype(np.int64, copy=False)
            probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
            if not np.isfinite(probabilities).all():
                raise ValueError("XGBoost prediction contains a non-finite score")
            predicted = probabilities.argmax(axis=1)
            if ((labels < 0) | (labels >= classes)).any():
                raise ValueError("multiclass test labels are outside num_class")
            np.add.at(matrix, (labels, predicted), 1)
        support = matrix.sum(axis=1)
        total_rows = int(support.sum())
        if total_rows < 1:
            raise ValueError("test dataset is empty")
        diagonal = matrix.diagonal()
        predicted_count = matrix.sum(axis=0)
        precision_by_class = np.divide(
            diagonal,
            predicted_count,
            out=np.zeros(classes, dtype=np.float64),
            where=predicted_count > 0,
        )
        recall_by_class = np.divide(
            diagonal,
            support,
            out=np.zeros(classes, dtype=np.float64),
            where=support > 0,
        )
        f1_by_class = np.divide(
            2.0 * precision_by_class * recall_by_class,
            precision_by_class + recall_by_class,
            out=np.zeros(classes, dtype=np.float64),
            where=(precision_by_class + recall_by_class) > 0,
        )
        weights = support / total_rows
        metrics = {
            "f1": float((f1_by_class * weights).sum()),
            "precision": float((precision_by_class * weights).sum()),
            "recall": float((recall_by_class * weights).sum()),
            "accuracy": float(diagonal.sum() / total_rows),
        }
        class_auc: list[float] = []
        for index in range(classes):
            auc, _, _ = _binary_ranking_metrics(
                predictions,
                probability_column=f"probability_{index}",
                positive_label=index,
                total_rows=total_rows,
                total_positive=int(support[index]),
                include_curve=False,
            )
            if auc is None:
                class_auc = []
                break
            class_auc.append(auc)
        if class_auc:
            metrics["auc"] = float(np.asarray(class_auc) @ weights)
        details["confusion_matrix"] = {
            "labels": [str(index) for index in range(classes)],
            "matrix": matrix.tolist(),
        }
    else:
        total_rows = 0
        squared_error = absolute_error = absolute_percentage_error = 0.0
        label_sum = label_square_sum = 0.0
        epsilon = np.finfo(np.float64).eps
        for frame in _prediction_batches(predictions):
            labels = frame["label"].to_numpy(dtype=np.float64)
            predicted = frame["prediction"].to_numpy(dtype=np.float64)
            if not np.isfinite(labels).all() or not np.isfinite(predicted).all():
                raise ValueError("regression evaluation requires finite values")
            error = labels - predicted
            total_rows += int(len(labels))
            squared_error += float(np.square(error).sum())
            absolute_error += float(np.abs(error).sum())
            absolute_percentage_error += float(
                (np.abs(error) / np.maximum(np.abs(labels), epsilon)).sum()
            )
            label_sum += float(labels.sum())
            label_square_sum += float(np.square(labels).sum())
        if total_rows < 1:
            raise ValueError("test dataset is empty")
        mean_squared_error = squared_error / total_rows
        total_variance = label_square_sum - label_sum * label_sum / total_rows
        r2 = (
            1.0 - squared_error / total_variance
            if total_variance > 0
            else float(squared_error == 0.0)
        )
        metrics = {
            "rmse": math.sqrt(mean_squared_error),
            "mae": absolute_error / total_rows,
            "mape": absolute_percentage_error / total_rows,
            "r2": r2,
        }
    return metrics, details


def _feature_importance(
    booster: Any, feature_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    scores = booster.get_score(importance_type="gain")
    by_name = {
        name: float(scores.get(name, scores.get(f"f{index}", 0.0)))
        for index, name in enumerate(feature_names)
    }
    ordered = sorted(feature_names, key=lambda name: (-by_name[name], name))
    return [
        {
            "rank": index,
            "model_feature_name": name,
            "importance_score": by_name[name],
        }
        for index, name in enumerate(ordered, start=1)
    ]


def _correlation_batch_stats(batch: Any, feature_names: list[str]) -> Any:
    import numpy as np
    import pandas as pd

    width = len(feature_names)
    numeric = batch.reindex(columns=feature_names).apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(values)
    valid = finite.astype(np.float64, copy=False)
    clean = np.where(finite, values, 0.0)

    def _flat(matrix: Any) -> list[float]:
        return np.asarray(matrix, dtype=np.float64).reshape(width * width).tolist()

    return pd.DataFrame(
        {
            "_corr_count": [_flat(valid.T @ valid)],
            "_corr_sum": [_flat(clean.T @ valid)],
            "_corr_square_sum": [_flat(np.square(clean).T @ valid)],
            "_corr_cross_sum": [_flat(clean.T @ clean)],
        }
    )


def _finalize_correlation_stats(
    feature_names: list[str], rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    import numpy as np

    width = len(feature_names)
    if width == 0 or not rows:
        return None
    shape = (width, width)

    def _sum_field(name: str) -> Any:
        total = np.zeros(shape, dtype=np.float64)
        for row in rows:
            total += np.asarray(row[name], dtype=np.float64).reshape(shape)
        return total

    count = _sum_field("_corr_count")
    pair_sum = _sum_field("_corr_sum")
    square_sum = _sum_field("_corr_square_sum")
    cross_sum = _sum_field("_corr_cross_sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = cross_sum - pair_sum * pair_sum.T / count
        variance_x = square_sum - np.square(pair_sum) / count
        denominator = np.sqrt(
            np.maximum(variance_x, 0.0) * np.maximum(variance_x.T, 0.0)
        )
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=(count >= 2) & (denominator > 0.0),
        )
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return {
        "feature_names": feature_names,
        "values": np.round(correlation, 6).tolist(),
    }


def _distributed_correlation(
    dataset: Any, feature_names: tuple[str, ...]
) -> dict[str, Any] | None:
    names = list(feature_names)
    rows = (
        dataset.select_columns(names)
        .map_batches(
            _correlation_batch_stats,
            batch_format="pandas",
            batch_size=_DEFAULT_BATCH_ROWS,
            fn_kwargs={"feature_names": names},
        )
        .take_all()
    )
    return _finalize_correlation_stats(names, rows)


def _export_bundle(
    booster: Any,
    *,
    feature_names: tuple[str, ...],
    bundle_uri: str,
    run_id: str,
) -> dict[str, Any]:
    import json

    import xgboost
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    learner = json.loads(booster.save_config())["learner"]
    objective = str(learner["objective"]["name"])
    classification = objective.startswith(("binary:", "multi:"))
    class_count = max(2, int(learner["learner_model_param"]["num_class"]))
    output_schema = (
        (
            CheckpointField(name="label", dtype="int64", shape=("batch",)),
            CheckpointField(
                name="probabilities", dtype="float32", shape=("batch", class_count)
            ),
        )
        if classification
        else (CheckpointField(name="prediction", dtype="float32", shape=("batch", 1)),)
    )
    checkpoint_contract = ExportCheckpointV1(
        trainer_type="xgboost",
        architecture_id="xgboost",
        input_schema=(
            CheckpointField(
                name="float_input",
                dtype="float32",
                shape=("batch", len(feature_names)),
            ),
        ),
        output_schema=output_schema,
        preprocessing={"type": "none"},
        task_type="classification" if classification else "regression",
        framework="xgboost",
        framework_version=xgboost.__version__,
        checkpoint_format_version=1,
    )
    raw = bytes(booster.save_raw(raw_format="ubj"))
    export_booster = xgboost.Booster()
    export_booster.load_model(bytearray(raw))
    export_booster.feature_names = None
    source = ExportSource(
        source_kind="xgboost_result",
        model_object=export_booster,
        architecture_id="xgboost",
        feature_schema={"feature_names": list(feature_names)},
        metadata={
            "framework": "xgboost",
            "framework_versions": {"xgboost": xgboost.__version__},
            "objective": objective,
            "producer_distribution": "tributo-algorithms-boosting",
        },
        source_fingerprint=hashlib.sha256(raw).hexdigest(),
        checkpoint_contract=checkpoint_contract,
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=bundle_uri,
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="onnx-model",
                    format="onnx",
                    exporter_id="official-xgboost-onnx-v1",
                ),
                ExportTarget(
                    name="native-model",
                    format="ubj",
                    exporter_id="official-xgboost-ubj-v1",
                ),
            ],
            roles={"inference": "onnx-model", "native": "native-model"},
        ),
    )
    return {
        "bundle_id": bundle.bundle_id,
        "bundle_uri": bundle.canonical_uri,
        "execution_id": bundle.execution_id,
        "manifest_sha256": bundle.manifest_sha256,
    }


def _validated_model_params(
    algorithm_config: Mapping[str, Any],
    *,
    task_type: str,
    num_class: int | None,
) -> dict[str, Any]:
    params = algorithm_config.get("model")
    if not isinstance(params, Mapping):
        raise TypeError("model configuration must be an object")
    model_params = dict(params)
    objective = model_params.get("objective")
    if not isinstance(objective, str) or not objective:
        raise ValueError("model.objective is required")
    expected = {
        "BINARY_CLASSIFICATION": "binary:logistic",
        "MULTICLASS_CLASSIFICATION": "multi:softprob",
    }.get(task_type)
    if expected is not None and objective != expected:
        raise ValueError(
            f"{task_type} requires probability objective {expected!r}, got {objective!r}"
        )
    if task_type == "REGRESSION" and not objective.startswith("reg:"):
        raise ValueError(f"REGRESSION requires a reg:* objective, got {objective!r}")
    if task_type == "MULTICLASS_CLASSIFICATION":
        configured_classes = model_params.get("num_class")
        if configured_classes != num_class:
            raise ValueError("model.num_class must match the requested num_class")
    if isinstance(model_params.get("eval_metric"), tuple):
        model_params["eval_metric"] = list(model_params["eval_metric"])
    return model_params


def run_xgboost_training(
    *,
    ingestion_request: IngestionRequest,
    feature_names: tuple[str, ...],
    label_name: str,
    algorithm_config: Mapping[str, Any],
    worker_count: int,
    resources: WorkerResources,
    run_id: str,
    task_type: str,
    num_class: int | None = None,
    evaluation_artifacts: Mapping[str, Any] | None = None,
    reporter: TrainingReporter | None = None,
) -> AlgorithmRunResult:
    """Run split-aware distributed XGBoost and return a complete Tributo result."""
    import importlib.metadata

    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer
    from ray.util.queue import Queue

    if not feature_names or any(not name for name in feature_names):
        raise ValueError("feature_names must contain non-empty names")
    if not label_name:
        raise ValueError("label_name must be non-empty")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if task_type not in {
        "BINARY_CLASSIFICATION",
        "MULTICLASS_CLASSIFICATION",
        "REGRESSION",
    }:
        raise ValueError("task_type is not supported by XGBoost training")
    if task_type == "MULTICLASS_CLASSIFICATION" and (
        num_class is None or num_class < 2
    ):
        raise ValueError("multiclass XGBoost requires num_class >= 2")
    model_params = _validated_model_params(
        algorithm_config, task_type=task_type, num_class=num_class
    )
    callback: TrainingReporter = reporter or _NullReporter()
    artifacts = evaluation_artifacts or {}
    gateway = IngestionGateway()
    opened = gateway.open(ingestion_request)
    metrics_queue: Any | None = None
    evidence_actor: Any | None = None
    pump: _MetricsPump | None = None
    histories: _MetricHistory = {}
    try:
        if not isinstance(opened.handle, RayDataHandle):
            raise TypeError("training ingestion must return a RayDataHandle")
        dataset = opened.handle.dataset.materialize()
        training = algorithm_config.get("training", {})
        if not isinstance(training, Mapping):
            raise TypeError("training configuration must be an object")
        callback.phase("DATA_SPLITTING")
        train_dataset, validation_dataset, test_dataset, rows = _split_dataset(
            dataset, training
        )
        train_dataset = _ensure_worker_blocks(
            train_dataset,
            worker_count=worker_count,
            row_count=rows["train"],
            name="train",
        )
        if validation_dataset is not None:
            validation_dataset = _ensure_worker_blocks(
                validation_dataset,
                worker_count=worker_count,
                row_count=rows["validation"],
                name="validation",
            )

        rounds = int(training.get("num_rounds", 100))
        batch_rows = int(training.get("batch_rows", _DEFAULT_BATCH_ROWS))
        if rounds < 1 or batch_rows < 1:
            raise ValueError("training rounds and batch_rows must be positive")
        early_stopping_rounds = training.get("early_stopping_rounds")
        if early_stopping_rounds is not None and validation_dataset is None:
            raise ValueError("early stopping requires a validation split")
        if early_stopping_rounds is not None and int(early_stopping_rounds) < 1:
            raise ValueError("early_stopping_rounds must be positive")
        metrics_queue = Queue(maxsize=256, actor_options={"num_cpus": 0})
        collector_type = ray.remote(_EvidenceCollector).options(num_cpus=0)
        evidence_actor = collector_type.remote()
        datasets = {"train": train_dataset}
        datasets_to_split = ["train"]
        if validation_dataset is not None:
            datasets["validation"] = validation_dataset
            datasets_to_split.append("validation")
        ray_config = algorithm_config.get("ray", {})
        if not isinstance(ray_config, Mapping):
            raise TypeError("ray configuration must be an object")
        storage_path = ray_config.get("storage_path")
        if not isinstance(storage_path, str) or not storage_path:
            raise ValueError("ray.storage_path is required")
        worker_resources = {
            "CPU": resources.num_cpus,
            "GPU": resources.num_gpus,
            **dict(resources.custom),
        }
        if resources.memory_bytes is not None:
            worker_resources["memory"] = float(resources.memory_bytes)
        trainer = XGBoostTrainer(
            train_loop_per_worker=_xgboost_train_loop,
            train_loop_config={
                "feature_names": list(feature_names),
                "label_name": label_name,
                "params": model_params,
                "num_rounds": rounds,
                "batch_rows": batch_rows,
                "early_stopping_rounds": early_stopping_rounds,
                "metrics_queue": metrics_queue,
                "evidence_actor": evidence_actor,
            },
            scaling_config=ScalingConfig(
                num_workers=worker_count,
                use_gpu=resources.num_gpus > 0,
                placement_strategy="SPREAD",
                resources_per_worker=worker_resources,
            ),
            datasets=datasets,
            dataset_config=CompleteCoverageDataConfig(
                datasets_to_split=datasets_to_split
            ),
            run_config=RunConfig(
                name=f"tributo-xgboost-{run_id}", storage_path=storage_path
            ),
        )
        callback.phase("EXECUTING")
        pump = _MetricsPump(metrics_queue, callback, rounds)
        pump.start()
        train_result = trainer.fit()
        train_metrics = train_result.metrics or {}
        raw_histories = train_metrics.get("metric_history", {})
        if isinstance(raw_histories, Mapping):
            histories = raw_histories
        pump.finish(histories)
        pump = None
        metrics_queue = None

        evidence_handle = cast(Any, evidence_actor)
        evidence = ray.get(evidence_handle.snapshot.remote())
        ray.kill(evidence_handle, no_restart=True)
        evidence_actor = None
        digests = {item.get("model_state_digest") for item in evidence}
        if (
            len(evidence) != worker_count
            or len(digests) != 1
            or sum(int(item.get("rows_processed", 0)) for item in evidence)
            != rows["train"]
            or sum(
                int(
                    cast(Mapping[str, Any], item.get("input_rows", {})).get(
                        "validation", 0
                    )
                )
                for item in evidence
            )
            != rows["validation"]
        ):
            raise RuntimeError("distributed XGBoost worker evidence is incomplete")

        booster = _load_booster(train_result.checkpoint, feature_names)
        callback.phase("MATERIALIZING")
        evaluation, details = _evaluation(
            test_dataset,
            booster,
            feature_names=feature_names,
            label_name=label_name,
            task_type=task_type,
            num_class=num_class,
            artifacts=artifacts,
        )
        correlation_matrix = None
        if artifacts.get("correlation_matrix", False):
            correlation_matrix = _distributed_correlation(train_dataset, feature_names)
            if correlation_matrix is None:
                raise RuntimeError("requested correlation matrix could not be computed")
        output = algorithm_config.get("output", {})
        if not isinstance(output, Mapping):
            raise TypeError("output configuration must be an object")
        bundle_uri = output.get("bundle_uri")
        if not isinstance(bundle_uri, str) or not bundle_uri:
            raise ValueError("output.bundle_uri is required")
        outputs = _export_bundle(
            booster,
            feature_names=feature_names,
            bundle_uri=bundle_uri,
            run_id=run_id,
        )
        return AlgorithmRunResult(
            run_id=run_id,
            plan_id=hashlib.sha256(
                f"tributo-algorithms-boosting:{run_id}".encode()
            ).hexdigest(),
            execution=AlgorithmExecutionResult(
                status="succeeded",
                metrics={
                    "evaluation": evaluation,
                    "evaluation_details": details,
                    "feature_importance": _feature_importance(booster, feature_names),
                    "correlation_matrix": correlation_matrix,
                    "sample_rows": rows,
                },
                outputs=outputs,
            ),
            actual_versions={
                "ray": importlib.metadata.version("ray"),
                "xgboost": importlib.metadata.version("xgboost"),
            },
            input_provenance={"dataset_ref": opened.receipt.dataset_ref},
            worker_metadata={"workers": evidence},
        )
    finally:
        if pump is not None:
            pump.finish(histories)
        elif metrics_queue is not None:
            with suppress(Exception):
                metrics_queue.shutdown(force=True)
        if evidence_actor is not None:
            with suppress(Exception):
                ray.kill(evidence_actor, no_restart=True)
        opened.close()


__all__ = ["TrainingReporter", "run_xgboost_training"]
