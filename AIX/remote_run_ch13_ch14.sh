#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/100knock"
mkdir -p logs

export PATH=/usr/local/anaconda3/bin:$PATH
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/

cd "$HOME/100knock/Chapter 13"
python knock90.py 2>&1 | tee ../logs/ch13_knock90.log
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock91.py --epochs 5 --batch-size 64 2>&1 | tee ../logs/ch13_knock91.log
python knock92.py --sentence "京都 は 美しい 。" 2>&1 | tee ../logs/ch13_knock92.log
python knock93.py --split test 2>&1 | tee ../logs/ch13_knock93.log

cd "$HOME/100knock/Chapter 14"
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock95.py --vocab-size 8000 2>&1 | tee ../logs/ch14_knock95.log
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock96.py --epochs 5 2>&1 | tee ../logs/ch14_knock96.log
python -m torch.distributed.launch --nproc_per_node=8 --use_env knock97.py --batch-sizes 32,64 --lrs 1e-3,1e-4,1e-5,1e-6 --optimizers adam,adamw,radam 2>&1 | tee ../logs/ch14_knock97.log
