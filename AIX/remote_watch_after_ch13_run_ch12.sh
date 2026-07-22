#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/anaconda3/bin:$HOME/.local/bin:$PATH
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/

cd "$HOME/100knock"
mkdir -p logs

while pgrep -u "$USER" -f 'Chapter 13|knock91.py|knock92.py|knock93.py' >/dev/null; do
  if pgrep -u "$USER" -f 'Chapter 14|knock95.py --vocab-size|knock96.py --epochs 5|knock97.py --batch-sizes' >/dev/null; then
    pkill -u "$USER" -f 'Chapter 14|knock95.py --vocab-size|knock96.py --epochs 5|knock97.py --batch-sizes' || true
    break
  fi
  sleep 20
done

if pgrep -u "$USER" -f 'Chapter 14|knock95.py --vocab-size|knock96.py --epochs 5|knock97.py --batch-sizes' >/dev/null; then
  pkill -u "$USER" -f 'Chapter 14|knock95.py --vocab-size|knock96.py --epochs 5|knock97.py --batch-sizes' || true
fi

if pgrep -u "$USER" -f 'knock98.py|knock99.py|Chapter 12' >/dev/null; then
  echo "Chapter 12 already running"
  exit 0
fi

if ! mkdir logs/ch12_run.lock 2>/dev/null; then
  echo "Chapter 12 is already scheduled or running"
  exit 0
fi

python - <<'PY' 2>&1 | tee logs/ch12_prepare_sst2.log
from pathlib import Path
from datasets import load_dataset

base = Path.home() / "100knock" / "Chapter 8" / "data" / "SST-2"
base.mkdir(parents=True, exist_ok=True)
if not (base / "train.tsv").exists() or not (base / "dev.tsv").exists():
    ds = load_dataset("glue", "sst2")
    for split, name in [("train", "train.tsv"), ("validation", "dev.tsv")]:
        with (base / name).open("w", encoding="utf-8") as f:
            f.write("sentence\tlabel\n")
            for row in ds[split]:
                sentence = row["sentence"].replace("\t", " ").replace("\n", " ")
                f.write(f"{sentence}\t{row['label']}\n")
print("SST-2 ready:", base)
PY

cd "$HOME/100knock/Chapter 12"
python knock90.py 2>&1 | tee ../logs/ch12_knock90.log
python knock91.py 2>&1 | tee ../logs/ch12_knock91.log
python knock92.py 2>&1 | tee ../logs/ch12_knock92.log
python knock93.py 2>&1 | tee ../logs/ch12_knock93.log
python knock94.py 2>&1 | tee ../logs/ch12_knock94.log
python knock95.py 2>&1 | tee ../logs/ch12_knock95.log
python knock96.py --max-examples 200 2>&1 | tee ../logs/ch12_knock96.log
python knock97.py --epochs 3 --batch-size 16 2>&1 | tee ../logs/ch12_knock97.log
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock98.py --epochs 1 --batch-size 2 --gradient-accumulation-steps 8 2>&1 | tee ../logs/ch12_knock98_legacy.log
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock99.py --epochs 1 --batch-size 1 --gradient-accumulation-steps 8 2>&1 | tee ../logs/ch12_knock99_legacy.log
