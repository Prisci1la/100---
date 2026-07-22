#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/anaconda3/bin:$HOME/.local/bin:$PATH
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/

cd "$HOME/100knock/Chapter 12"

python knock98.py --epochs 1 --batch-size 2 --gradient-accumulation-steps 8 --max-train-examples 2000 2>&1 | tee ../logs/ch12_knock98_legacy_single.log
python knock99.py --epochs 1 --batch-size 1 --gradient-accumulation-steps 8 --max-train-examples 1000 2>&1 | tee ../logs/ch12_knock99_legacy_single.log
