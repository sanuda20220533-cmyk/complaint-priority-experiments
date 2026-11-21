# Customer Complaint Priority Experiments

Research code for comparing DistilBERT with TF-IDF logistic regression and linear SVM on chronological partitions of financial complaint intake text. The primary metric is the share of urgent cases captured when a reviewer can read 10% of the holdout. Additional outputs include 5% and 20% review budgets, average precision, macro-F1, Brier score, reliability bins, and CPU latency.

The end-to-end software check uses generated examples and a tiny randomly initialised DistilBERT. The repository contains no complaint records, trained weights, or empirical study results.

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
