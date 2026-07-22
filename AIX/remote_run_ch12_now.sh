#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/100knock"
mkdir -p logs

export PATH=/usr/local/anaconda3/bin:$HOME/.local/bin:$PATH
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/

python - <<'PY' 2>&1 | tee logs/ch12_prepare_sst2.log
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

base = Path.home() / "100knock" / "Chapter 8" / "data" / "SST-2"
base.mkdir(parents=True, exist_ok=True)
train_path = base / "train.tsv"
dev_path = base / "dev.tsv"
if not train_path.exists() or not dev_path.exists():
    zip_path = base / "SST-2.zip"
    url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    print("Downloading", url)
    urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for member, out_name in [("SST-2/train.tsv", "train.tsv"), ("SST-2/dev.tsv", "dev.tsv")]:
            with archive.open(member) as src, (base / out_name).open("wb") as dst:
                dst.write(src.read())
print("SST-2 ready:", base)
print("train bytes:", train_path.stat().st_size)
print("dev bytes:", dev_path.stat().st_size)
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
