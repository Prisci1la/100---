#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/100knock"
mkdir -p logs

export PATH=/usr/local/anaconda3/bin:$PATH
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/

python -m pip install --user --proxy=http://proxy.uec.ac.jp:8080/ \
  "safetensors==0.3.3" \
  "tokenizers<0.14" \
  "transformers==4.30.2" \
  "datasets==2.13.2" \
  "peft==0.3.0" \
  "trl==0.4.7" \
  "accelerate==0.20.3" \
  "sacrebleu==2.3.3" \
  fugashi \
  unidic-lite \
  sentencepiece \
  "bitsandbytes<0.42" \
  2>&1 | tee logs/install_deps.log
