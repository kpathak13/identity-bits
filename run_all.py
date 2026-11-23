"""Run the whole pipeline: generate -> hygiene -> weights -> graph -> spine -> evaluate."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["generate", "hygiene", "weights", "graph", "spine", "evaluate",
          "stacking", "groundtruth", "populations"]

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=100_000,
                    help="people to simulate (default 100k -> ~240k records)")
    a = ap.parse_args()
    os.environ["N_PEOPLE"] = str(a.people)
    t0 = time.time()
    for s in STAGES:
        print("\n" + "=" * 72)
        r = subprocess.run([sys.executable, str(ROOT / "src" / f"{s}.py")])
        if r.returncode:
            sys.exit(f"stage {s} failed")
    print("\n" + "=" * 72)
    print(f"done in {time.time()-t0:.1f}s — see out/")

if __name__ == "__main__":
    main()
