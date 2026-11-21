"""Run the chronological experiment from a JSON configuration."""
import argparse
import json
import platform
from pathlib import Path

import scipy
import sklearn
import torch
import transformers

from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("complaint_priority_results.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    required = ("csv_path", "dataset_source", "label_provenance", "label_map",
                "train_through", "validation_through")
    missing = [key for key in required if not config.get(key)]
    if not config.get("transformer_seeds", config.get("distilbert_seeds")):
        missing.append("transformer_seeds")
    if missing:
        parser.error("Fill in these configuration fields: " + ", ".join(missing))

    torch.set_num_threads(2)
    report = run_experiment(config)
    report["config"]["csv_path"] = Path(config["csv_path"]).name
    report["config"]["checkpoint"] = config.get("checkpoint_id", config["checkpoint"])
    report["environment"] = {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "operating_system": platform.system(),
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "training_device": config.get("training_device") or (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                                                            else "Apple MPS" if torch.backends.mps.is_available() else "CPU"),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
