"""Recompute the released analyses on CPU without retraining or GPU access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Skip the longer H1-H5 censored-model bootstrap")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/.matplotlib"))
    pin = json.loads((ROOT / "artifact_config.json").read_text())["formal_execution_alias"]
    common = ["--attempts-dir", "results/attempts", "--manifests-dir", "results/manifests",
              "--target-manifest", "results/target_sets/e3b.json", "--expected-formal-commit", pin]

    def run(script: str, *arguments: str) -> None:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, script, *arguments], check=True)

    run("analyze_e3b_repair.py", *common, "--output-dir", "output/control_curve")
    run("analyze_e3b_base_followup.py", "--output-dir", "output/base_followup")
    run("render_e3b_base_followup.py", "--analysis-json", "output/base_followup/four_cell_analysis.json",
        "--output-stem", "output/base_followup/four_cell_summary")
    if not args.quick:
        run("analyze_e3b_hypotheses.py", *common, "--output-dir", "output/hypotheses")
        # Render the manuscript capacity figure from the recomputed CSV.
        figure_dir = ROOT / "output/paper_figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "figures/fig_e3_capacity_combined.py", figure_dir / "fig_e3_capacity_combined.py")
        shutil.copyfile(ROOT / "output/hypotheses/e3b_h1_h5_curve.csv", figure_dir / "e3b_h1_h5_curve.csv")
        run("output/paper_figures/fig_e3_capacity_combined.py")
    print("Reproduction completed. See output/; reference_results/ remains unchanged.")


if __name__ == "__main__":
    main()
