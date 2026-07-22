#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/anaconda3/bin:$PATH
cd "$HOME/100knock"

python - <<'PY'
mods = ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "sacrebleu", "fugashi", "sentencepiece", "tensorboard"]
for mod in mods:
    try:
        module = __import__(mod)
        print(mod, "OK", getattr(module, "__version__", ""))
    except Exception as exc:
        print(mod, "MISSING", type(exc).__name__, str(exc)[:200])
PY
