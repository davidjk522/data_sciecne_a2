#20212663 김재경
import subprocess
import sys
import time

DATASETS = [f"u{i}" for i in range(1, 6)]

for ds in DATASETS:
    train = f"{ds}.base"
    test  = f"{ds}.test"
    print(f"\n{'='*50}")
    print(f"  {ds}  ({train} / {test})")
    print(f"{'='*50}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "recommender.py", train, test],
        text=True
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[ERROR] {ds} failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"[done] {ds} — {elapsed:.1f}s")

print("\n모든 데이터셋 완료!")
