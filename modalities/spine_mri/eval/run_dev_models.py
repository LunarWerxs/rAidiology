"""Development side of Phase 2 in one command, after run_split.py dev:
crops -> grader training (5 outer + 20 nested models) -> out-of-fold predictions -> predictions.csv
-> SCORECARD.md -> model card. It needs a GPU and takes hours:
  python -u modalities/spine_mri/eval/run_dev_models.py [--version 1] [--skip-train]
Each step is a separate process, so a crash names its step and the earlier outputs stay."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def step(*args):
    t0 = time.time()
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", *args], check=True, cwd=HERE)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)


def main():
    version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "1"
    if not os.path.exists(os.path.join(HERE, "..", "..", "..", "work", "eval", "spider", "dev", "crops.npz")):
        step("crops.py", "dev")
    if "--skip-train" not in sys.argv:
        step("grader.py", "train", "--version", version)
    step("grader.py", "predict", "dev", "--version", version)
    step("run_split.py", "--collect-only")
    step("score.py")
    step("model_card.py", "--version", version)
    print("dev models finished", flush=True)


if __name__ == "__main__":
    main()
