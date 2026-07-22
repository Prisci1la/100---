#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/100knock"
mkdir -p logs

export PATH=/usr/local/anaconda3/bin:$PATH
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/

rm -rf "$HOME/.cache/pip" /tmp/pip-* "$HOME"/tmp/pip-* 2>/dev/null || true

python -m pip install --user --proxy=http://proxy.uec.ac.jp:8080/ \
  "safetensors==0.3.3" \
  "tokenizers<0.14" \
  "transformers==4.30.2" \
  "datasets==2.13.2" \
  "sacrebleu==2.3.3" \
  fugashi \
  unidic-lite \
  sentencepiece \
  "bitsandbytes<0.42" \
  2>&1 | tee logs/install_light_deps_main.log

python -m pip install --user --no-deps --proxy=http://proxy.uec.ac.jp:8080/ \
  "accelerate==0.20.3" \
  "peft==0.3.0" \
  "trl==0.4.7" \
  2>&1 | tee logs/install_light_deps_nodeps.log

python - <<'PY' 2>&1 | tee logs/check_imports.log
mods = ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "sacrebleu", "fugashi", "sentencepiece", "tensorboard"]
for mod in mods:
    try:
        module = __import__(mod)
        print(mod, "OK", getattr(module, "__version__", ""))
    except Exception as exc:
        print(mod, "MISSING", type(exc).__name__, str(exc)[:200])
PY
