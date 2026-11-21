import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


def read_complaints(path, columns, label_map):
    """Read documented intake text, priority and an intake date in ISO 8601."""
    if set(label_map.values()) != {0, 1}:
        raise ValueError("Label map must document both urgent (1) and non-urgent (0) labels.")
    labels = {str(key).strip().casefold(): value for key, value in label_map.items()}
    if len(labels) != len(label_map):
        raise ValueError("Label keys must be unique after case and whitespace normalisation.")
    records = []
    row_counts = Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must have these columns: {columns}")
        for row in reader:
            row_counts["input_rows"] += 1
            text, raw_label, raw_date = ((row[col] or "").strip() for col in columns)
            if not (text and raw_label and raw_date):
                row_counts["missing_required_value"] += 1
                continue
            if raw_label.casefold() not in labels:
                raise ValueError(f"Undocumented priority label: {raw_label!r}")
            try:
                day = date.fromisoformat(raw_date.split("T", 1)[0].split(" ", 1)[0])
            except ValueError as error:
                raise ValueError(f"Expected ISO intake date, got {raw_date!r}") from error
            key = " ".join(text.split()).casefold()
            records.append((day, text, labels[raw_label.casefold()], key))
    grouped_labels = {}
    for _, _, label, key in records:
        grouped_labels.setdefault(key, set()).add(label)
    seen, kept = set(), []
    for day, text, label, key in sorted(records):
        if len(grouped_labels[key]) > 1:
            row_counts["rows_with_conflicting_duplicates"] += 1
        elif key in seen:
            row_counts["duplicate_rows_removed"] += 1
        else:
            seen.add(key)
            kept.append((day, text, label))
    row_counts["retained_rows"] = len(kept)
    return kept, dict(row_counts)


def split_chronologically(records, train_through, validation_through):
    train_end, validation_end = map(date.fromisoformat, (train_through, validation_through))
    if train_end >= validation_end:
        raise ValueError("Training must end before validation ends.")
    splits = (
        [r for r in records if r[0] <= train_end],
        [r for r in records if train_end < r[0] <= validation_end],
        [r for r in records if r[0] > validation_end],
    )
    summary = {}
    for name, group in zip(("train", "validation", "test"), splits):
        counts = Counter(row[2] for row in group)
        if set(counts) != {0, 1}:
            raise ValueError(f"{name} must contain both classes. Audit chronology and label support.")
        if name in {"validation", "test"} and len(group) < 20:
            raise ValueError(f"{name} needs at least 20 complaints for the fixed review budgets.")
        summary[name] = {"first_date": group[0][0].isoformat(),
                         "last_date": group[-1][0].isoformat(),
                         "urgent": counts[1], "non_urgent": counts[0]}
    return splits, summary


def xy(rows):
    return [r[1] for r in rows], np.asarray([r[2] for r in rows])


def lexical_models(train, validation):
    x_train, y_train = xy(train)
    x_val, y_val = xy(validation)
    models = {}
    for name, classifier in (
        ("tfidf_logistic", LogisticRegression(max_iter=1000, random_state=0)),
        ("tfidf_svm", LinearSVC(random_state=0)),
    ):
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), classifier)
        model.fit(x_train, y_train)
        if name == "tfidf_logistic":
            models[name] = lambda texts, m=model: m.predict_proba(texts)[:, 1]
        else:
            margins = model.decision_function(x_val).reshape(-1, 1)
            calibrator = LogisticRegression(random_state=0).fit(margins, y_val)
            models[name] = lambda texts, m=model, c=calibrator: c.predict_proba(
                m.decision_function(texts).reshape(-1, 1)
            )[:, 1]
    return models


def fit_distilbert(train, validation, checkpoint, seed, epochs=2, batch_size=16, max_length=256):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    texts, labels = xy(train)
    order = np.random.default_rng(seed)

    def encode(texts, target):
        return tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                         return_tensors="pt").to(target)

    model.train()
    for _ in range(epochs):
        sequence = order.permutation(len(texts))
        for start in range(0, len(sequence), batch_size):
            received = sequence[start:start + batch_size]
            batch = encode([texts[i] for i in received], device)
            expected = torch.as_tensor(labels[received], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            model(**batch, labels=expected).loss.backward()
            optimizer.step()

    model.eval()

    def logits(texts):
        target = next(model.parameters()).device
        with torch.inference_mode():
            return np.vstack([
                model(**encode(texts[i:i + batch_size], target)).logits.cpu().numpy()
                for i in range(0, len(texts), batch_size)
            ]).astype(np.float64)

    x_val, y_val = xy(validation)
    validation_logits = logits(x_val)

    def nll(log_temp):
        scaled = validation_logits / np.exp(log_temp)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y_val)), y_val]))

    calibration = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    if not calibration.success:
        raise RuntimeError("Temperature scaling failed.")
    temperature = float(np.exp(calibration.x))

    def predict(texts):
        return softmax(logits(texts) / temperature, axis=1)[:, 1]

    return predict, lambda: model.to("cpu"), temperature


def review_metrics(labels, scores, capacity):
    slots = max(1, int(len(labels) * capacity))
    reviewed = np.argsort(-scores, kind="stable")[:slots]
    hits = int(labels[reviewed].sum())
    return {"reviewed": slots, "urgent_caught": hits,
            "urgent_capture": hits / int(labels.sum()), "review_precision": hits / slots}


def reliability_bins(labels, scores):
    bins = np.minimum((scores * 10).astype(int), 9)
    return [{"range_start": i / 10, "count": int(mask.sum()),
             "mean_predicted": float(scores[mask].mean()),
             "actual_urgent_rate": float(labels[mask].mean())}
            for i in range(10) if (mask := bins == i).any()]


def measure(labels, scores, predict, texts, capacities):
    predict(texts[:1])
    elapsed = []
    for text in texts[: min(100, len(texts))]:
        start = perf_counter_ns()
        predict([text])
        elapsed.append((perf_counter_ns() - start) / 1e6)
    return {
        "macro_f1_at_half": f1_score(labels, scores >= .5, average="macro", zero_division=0),
        "average_precision": average_precision_score(labels, scores),
        "brier_score": brier_score_loss(labels, scores),
        "reliability_bins": reliability_bins(labels, scores),
        "uncertainty_review_by_capacity": {
            str(c): review_metrics(labels, -np.abs(scores - .5), c) for c in capacities
        },
        "review_by_capacity": {str(c): review_metrics(labels, scores, c) for c in capacities},
        "cpu_latency_ms": {"p50": float(np.quantile(elapsed, .5)),
                           "p95": float(np.quantile(elapsed, .95)), "requests": len(elapsed)},
    }


def paired_block_interval(labels, first, second, days, capacity, draws=500):
    """Bootstrap paired rankings by seven-day blocks. Return no interval for short series."""
    days = np.asarray([(day - days[0]).days // 7 for day in days])
    blocks = [np.flatnonzero(days == block) for block in np.unique(days)]
    if len(blocks) < 8:
        return {"weekly_blocks": len(blocks), "interval": None}
    rng = np.random.default_rng(923)
    values = []
    for _ in range(draws):
        idx = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        if labels[idx].sum():
            a = review_metrics(labels[idx], first[idx], capacity)["urgent_capture"]
            b = review_metrics(labels[idx], second[idx], capacity)["urgent_capture"]
            values.append(a - b)
    return {"weekly_blocks": len(blocks), "valid_draws": len(values),
            "interval": np.quantile(values, [.025, .975]).tolist() if len(values) >= 100 else None}


def run_experiment(config):
    if not config["label_provenance"].strip():
        raise ValueError("Document who assigns the labels and what urgent means before running.")
    csv_path = Path(config["csv_path"])
    records, audit = read_complaints(csv_path, config["columns"], config["label_map"])
    (train, validation, test), splits = split_chronologically(
        records, config["train_through"], config["validation_through"]
    )
    x_val, y_val = xy(validation)
    x_test, y_test = xy(test)
    capacities = (.05, .10, .20)
    lexical = lexical_models(train, validation)
    strongest = max(lexical, key=lambda name: (
        review_metrics(y_val, lexical[name](x_val), .10)["urgent_capture"],
        average_precision_score(y_val, lexical[name](x_val)),
    ))
    predictions = {name: model(x_test) for name, model in lexical.items()}
    results = {name: measure(y_test, predictions[name], model, x_test, capacities)
               for name, model in lexical.items()}
    for seed in config["distilbert_seeds"]:
        predict, use_cpu, temperature = fit_distilbert(
            train, validation, config["checkpoint"], seed, epochs=config["epochs"]
        )
        scores = predict(x_test)
        use_cpu()
        name = f"distilbert_seed_{seed}"
        results[name] = measure(y_test, scores, predict, x_test, capacities)
        results[name]["temperature"] = temperature
        results[name]["capture_difference_vs_validation_selected_baseline"] = (
            paired_block_interval(y_test, scores, predictions[strongest], [r[0] for r in test], .10)
        )
        del predict, use_cpu
    with csv_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"config": config, "csv_sha256": digest, "audit": audit, "splits": splits,
            "random_review_expected_urgent_capture": {
                str(c): int(len(test) * c) / len(test) for c in capacities
            },
            "baseline_selected_on_validation": strongest,
            "majority_class_macro_f1": f1_score(
                y_test, np.full(len(y_test), int(xy(train)[1].mean() > .5)),
                average="macro", zero_division=0),
            "models": results}
