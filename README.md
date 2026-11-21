# Customer Complaint Priority Experiments

Research code for comparing DistilBERT with TF-IDF logistic regression and linear SVM on chronological partitions of financial complaint intake text. The primary metric is the share of urgent cases captured when a reviewer can read 10% of the holdout. Additional outputs include 5% and 20% review budgets, average precision, macro-F1, Brier score, reliability bins, and CPU latency.

The end-to-end software check uses generated examples and a tiny randomly initialised DistilBERT. The repository contains no complaint records or trained weights.

## Check the software

Use Python 3.11 or newer with compatible PyTorch wheels.

```sh
python -m pip install -r requirements.txt
python smoke.py
```

## Run with research data

Provide an authorised CSV with English financial complaint intake text, documented priority labels and ISO-format intake dates. Record the data source, priority definitions, and time boundaries in a copy of `config.example.json`. Only use a label map supported by the source documentation. The baseline is selected on validation data before evaluation on the latest period. Check near-duplicate records before running. The code automatically audits exact duplicates and conflicts.

```sh
python run_experiment.py config.json
```

The first research run can download the DistilBERT checkpoint. The command writes aggregate metrics and an input fingerprint to `complaint_priority_results.json`. Keep input records and privately sourced outputs in the approved environment. For Colab or Jupyter, open `Complaint_Priority_Experiments.ipynb` and complete the configuration cell. Use Colab only for data permitted in that environment.

Read the CFPB field documentation before using its data. Its company response timeliness field is a subsequent outcome and cannot supply the documented urgency label required by this study.

- [CFPB complaint data documentation](https://www.consumerfinance.gov/complaint/data-use/)

## Financial complaint pilot

This run uses 3,000 existing CFPB complaints annotated by S-W Leong. The released High label is the positive class. Medium and Low are the negative class. These AI-assigned labels are a research proxy for urgency and require independent validation. All rows in the release fall on four consecutive dates. The test split is the last date.

1. Obtain the [pinned annotated CSV](https://github.com/S-W-Leong/complaint-classification/blob/f59453b2624a060d01e97bb8025668aa505a0561/data/processed/cfpb_labeled_3000.csv) from the source. Its SHA-256 should be `0e172b586d0febe638af85e864d07dc85455077f1b3410aed652d10e2e60ec22`. Place the CSV in your working directory or edit the file path in the pilot configuration.
2. Use the same Python dependencies and run `python run_experiment.py pilot.minilm.json --output pilot_reproduction.json`. The baseline and fine-tuning settings are fixed in the code and configuration. Run on CPU to compare with the archived benchmark.
3. Inspect the aggregated metrics in `pilot_metrics.json`. Latency varies by hardware. The bootstrap time-block interval is unavailable because the test set covers one day.

The compact transformer used in the pilot is `sentence-transformers/all-MiniLM-L6-v2`. The separate proposed operational evaluation requires independently validated labels and a longer holdout.
