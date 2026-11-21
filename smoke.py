import csv
import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

from experiment import run_experiment

torch.set_num_threads(2)
with TemporaryDirectory(prefix="fyp-code-fixtures-") as work:
    work = Path(work)
    checkpoint = work / "untrained_fixture_checkpoint"
    checkpoint.mkdir()
    vocab = checkpoint / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\naccount\nlocked\nnow\nroutine\nstatement\ncase\n")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(checkpoint)
    DistilBertForSequenceClassification(
        DistilBertConfig(vocab_size=len(tokenizer), dim=32, hidden_dim=64, n_layers=1, n_heads=4)
    ).save_pretrained(checkpoint)
    src = work / "fixture.csv"
    ids = [str(i) for i in range(152)]
    with src.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["text", "label", "date"])
        writer.writeheader()
        for i, case in enumerate(ids):
            urgent = i % 4 == 1
            writer.writerow(dict(text=("account locked now" if urgent else "routine statement") + " case " + case,
                                 label="urgent" if urgent else "routine",
                                 date=(date(2020, 1, 1) + timedelta(days=i)).isoformat()))
        writer.writerow(dict(text="routine statement case 0", label="routine", date="2020-12-01"))
        writer.writerow(dict(text="", label="urgent", date="2020-01-01"))
    result = run_experiment(dict(csv_path=str(src), columns=("text","label","date"),
                                 label_map={"urgent":1,"routine":0}, label_provenance="Code-only fixture.",
                                 train_through="2020-02-19", validation_through="2020-03-20",
                                 checkpoint=str(checkpoint), distilbert_seeds=[13], epochs=1))
    assert result["audit"] == {"input_rows":154,"retained_rows":152,
                                "duplicate_rows_removed":1,"missing_required_value":1}
    assert len(result["models"]) == 3
    assert result["models"]["distilbert_seed_13"]["capture_difference_vs_validation_selected_baseline"]["valid_draws"] >= 100
    assert all(0 <= m["brier_score"] <= 1 for m in result["models"].values())
    print("PASS: chronological data audit, two lexical baselines, transformer training and calibration, paired intervals and latency.")
