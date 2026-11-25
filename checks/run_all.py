"""Run every numerical check in this folder, one at a time, single-threaded."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "exact_law.py",
    "phi_and_lemmaB.py",
    "lemmaB_pairing.py",
    "lemmaC_bound.py",
    "lemmaC_steps.py",
    "lemmaA_numerics.py",
]

env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
failed = []
for name in SCRIPTS:
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable, os.path.join(HERE, name)], cwd=HERE, env=env)
    print(f"----- {name}: exit {rc}, {time.time() - t0:.1f}s", flush=True)
    if rc != 0:
        failed.append(name)

print("\nAll checks ran." if not failed else f"\nFailed: {', '.join(failed)}")
sys.exit(1 if failed else 0)
