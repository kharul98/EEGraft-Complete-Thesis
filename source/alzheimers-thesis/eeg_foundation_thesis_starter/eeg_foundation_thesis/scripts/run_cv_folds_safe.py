"""Run safe development cross-validation experiments."""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
LOGS = ROOT / "logs/dev_cv5_2026"

MODELS = (
    "fusion",
    "no_skip",
    "high_only",
    "low_only"
)


def run_logged(command, log_path):

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nCOMMAND:",
        " ".join(map(str, command)),
        flush=True
    )

    print("LOG:", log_path, flush=True)

    with log_path.open("w", encoding="utf-8") as log:

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)

        status = process.wait()

    print("Process exit code:", status, flush=True)

    if status != 0:
        raise RuntimeError(
            f"Command failed. Inspect {log_path}"
        )


def validate_training(report_path, model, manifest, run_name):

    report = json.loads(
        report_path.read_text()
    )

    cfg = report.get("config", {})

    if (
        report.get("test_not_evaluated") is not True
        or report.get("test") is not None
    ):
        raise RuntimeError(
            "Existing experiment is not validation-only."
        )

    if cfg.get("validation_only") is not True:
        raise RuntimeError(
            "Missing validation-only setting."
        )

    if Path(
        cfg.get("manifest", "")
    ).resolve() != manifest.resolve():
        raise RuntimeError(
            "Existing experiment used another manifest."
        )

    if (
        cfg.get("model") != model
        or cfg.get("train_mode") != "frozen"
    ):
        raise RuntimeError(
            "Existing architecture mismatch."
        )

    if cfg.get("run_name") != run_name:
        raise RuntimeError("Run name mismatch.")

    if (
        int(cfg.get("seconds", 0)) != 4
        or int(cfg.get("max_windows_per_subject", 0)) != 24
        or int(cfg.get("seed", -1)) != 42
    ):
        raise RuntimeError(
            "Existing experimental settings mismatch."
        )


def run_one(fold, model):

    if fold not in range(5):
        raise ValueError("Fold must be between 0 and 4.")

    if model not in MODELS:
        raise ValueError("Unsupported decoder.")

    manifest = (
        ROOT /
        f"manifests/nested_cv5_dev2026/fold_{fold}.csv"
    )

    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    run_name = f"nestedcv5_f{fold}_{model}_s42"

    folder = RESULTS / run_name

    metrics = folder / "metrics.json"
    checkpoint = folder / "best.pt"

    outer_metrics = (
        folder / "outer_dev_metrics.json"
    )

    outer_predictions = (
        folder / "outer_dev_subject_predictions.csv"
    )

    print("\n" + "=" * 60)
    print("FOLD:", fold)
    print("MODEL:", model)
    print("EXPERIMENT:", run_name)
    print("=" * 60)

    if not folder.exists():

        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/train.py"),
            "--manifest", str(manifest),
            "--run-name", run_name,
            "--model", model,
            "--train-mode", "frozen",
            "--validation-only",
            "--seconds", "4",
            "--max-windows-per-subject", "24",
            "--batch-size", "8",
            "--epochs", "15",
            "--patience", "5",
            "--seed", "42"
        ]

        run_logged(
            command,
            LOGS / f"{run_name}_train.log"
        )

    elif not (
        metrics.is_file()
        and checkpoint.is_file()
    ):
        raise RuntimeError(
            "Incomplete experiment folder exists. "
            f"Preserving it for inspection: {folder}"
        )

    else:
        print(
            "Completed training checkpoint found. "
            "Skipping training."
        )

    if not (
        metrics.is_file()
        and checkpoint.is_file()
    ):
        raise RuntimeError(
            "Training did not generate both expected files."
        )

    validate_training(
        metrics,
        model,
        manifest,
        run_name
    )

    if (
        outer_metrics.is_file()
        and outer_predictions.is_file()
    ):

        previous = json.loads(
            outer_metrics.read_text()
        )

        if previous.get(
            "original_final_test_evaluated"
        ) is not False:
            raise RuntimeError(
                "Unexpected evaluation safety flag."
            )

        if Path(
            previous.get("manifest", "")
        ).resolve() != manifest.resolve():
            raise RuntimeError(
                "Existing outer-fold manifest mismatch."
            )

        print(
            "Outer-fold evaluation already completed."
        )

        return

    if (
        outer_metrics.exists()
        or outer_predictions.exists()
    ):
        raise RuntimeError(
            "Partial outer-fold results found. "
            "Preserving files for inspection."
        )

    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/evaluate_dev_fold_safe.py"),
        "--manifest", str(manifest),
        "--run-name", run_name,
        "--batch-size", "8"
    ]

    run_logged(
        command,
        LOGS / f"{run_name}_outer.log"
    )

    if not (
        outer_metrics.is_file()
        and outer_predictions.is_file()
    ):
        raise RuntimeError(
            "Outer evaluation did not generate both output files."
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        default="0,1,2,3,4"
    )

    parser.add_argument(
        "--models",
        default=",".join(MODELS)
    )

    args = parser.parse_args()

    folds = [
        int(x.strip())
        for x in args.folds.split(",")
    ]

    models = [
        x.strip()
        for x in args.models.split(",")
    ]

    if (
        not folds
        or len(set(folds)) != len(folds)
        or not set(folds).issubset(range(5))
    ):
        raise ValueError("Invalid or duplicated fold IDs.")

    if (
        not models
        or len(set(models)) != len(models)
        or not set(models).issubset(MODELS)
    ):
        raise ValueError("Invalid decoder names.")

    for fold in folds:
        for model in models:
            run_one(fold, model)

    print("\n========== FINISHED ==========")

    print("Selected development experiments completed.")
    print("Original 13 final-test patients remain reserved.")


if __name__ == "__main__":

    try:
        main()

    except Exception:
        traceback.print_exc()

        print(
            "\nExperiment runner stopped after an error."
        )

        print(
            "Previous results have been preserved."
        )

        sys.exit(1)
