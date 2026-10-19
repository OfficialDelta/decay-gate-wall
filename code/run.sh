#!/bin/bash
# Haystack decay-init test (see PREDICTIONS.md). Resumable: re-run to continue; finished runs are skipped.
# Usage:  bash ~/Documents/haystack-test/run.sh        (JOBS=N to override parallelism)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
mkdir -p results logs

# 1) Python >= 3.10
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
[ -n "$PY" ] || { echo "Need Python 3.10+ (e.g. 'brew install python@3.12')"; exit 1; }

# 2) isolated venv with PyTorch (CPU is used on purpose; see README)
if [ ! -x .venv/bin/python ]; then
  echo "Creating venv with $PY ..."
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch
fi

# 3) the paper's public code (MIT), pinned to the exact commit tested
REPO="$DIR/dualgoose"
if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/JIBSIL/dualgoose.git "$REPO"
fi
git -C "$REPO" checkout -q 8ca624a465e22a26583716399ef9a216da1f5627

# 4) job list, most important first
: > jobs.txt
for s in 0 1 2 3 4 5 6 7 8 9; do              # A. main replication, paper budget
  echo "L64_baseline_s$s --arm baseline --seed $s --seq-len 64 --batch 256 --steps 3000" >> jobs.txt
  echo "L64_retain4_s$s --arm retain --retain-bias 4 --seed $s --seq-len 64 --batch 256 --steps 3000" >> jobs.txt
done
for s in 0 1 2 3 4; do                          # B. collision decoys (predicted NOT fixed)
  echo "L64coll_baseline_s$s --arm baseline --seed $s --seq-len 64 --collision --batch 256 --steps 3000" >> jobs.txt
  echo "L64coll_retain4_s$s --arm retain --retain-bias 4 --seed $s --seq-len 64 --collision --batch 256 --steps 3000" >> jobs.txt
done
for s in 0 1 2 3 4; do                          # C. long haystack, where the paper's curriculum struggles
  echo "L256_baseline_s$s --arm baseline --seed $s --seq-len 256 --batch 128 --steps 3000" >> jobs.txt
  echo "L256_retain4_s$s --arm retain --retain-bias 4 --seed $s --seq-len 256 --batch 128 --steps 3000" >> jobs.txt
  echo "L256_retain6_s$s --arm retain --retain-bias 6 --seed $s --seq-len 256 --batch 128 --steps 3000" >> jobs.txt
done

cat > job.sh <<'EOF'
#!/bin/bash
name="$1"; shift
D="$(cd "$(dirname "$0")" && pwd)"
[ -f "$D/results/$name.json" ] && exit 0
"$D/.venv/bin/python" -I "$D/run_arm.py" --repo "$D/dualgoose" --out "$D/results/$name.json" "$@" > "$D/logs/$name.log" 2>&1 \
  && echo "finished $name" || echo "FAILED $name (see logs/$name.log)"
EOF
chmod +x job.sh

# 5) parallelism = performance cores (each run is single-threaded)
JOBS="${JOBS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)}"
TOTAL=$(wc -l < jobs.txt | tr -d ' ')
echo "Running $TOTAL runs, $JOBS at a time. Leave this window open; the computer is kept awake while it runs."
echo "Progress: ls $DIR/results | wc -l    Summary any time: .venv/bin/python summarize.py"
caffeinate -i xargs -P "$JOBS" -L 1 ./job.sh < jobs.txt
echo
.venv/bin/python summarize.py
echo "ALL DONE"
