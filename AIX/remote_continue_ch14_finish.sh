#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/anaconda3/bin:$HOME/.local/bin:$PATH
export http_proxy=http://proxy.uec.ac.jp:8080/
export https_proxy=http://proxy.uec.ac.jp:8080/

mkdir -p "$HOME/100knock/logs"
cd "$HOME/100knock/Chapter 14"

echo "===== Chapter 14 finish run started: $(date) ====="

python knock94.py --beam-sizes 1,2,5,10 --max-examples 200 \
  2>&1 | tee ../logs/ch14_knock94_limited.log

python - <<'PY' 2>&1 | tee ../logs/ch14_knock95_eval_limited.log
import matplotlib.pyplot as plt
import torch

from chapter14_utils import (
    CHECKPOINT_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    evaluate_bleu_for_beam,
    get_device,
    load_checkpoint,
    load_token_lines,
    write_json,
)

device = get_device()
checkpoint = CHECKPOINT_DIR / "knock95_subword_mt.pt"
model, src_vocab, tgt_vocab, _config = load_checkpoint(checkpoint, device)
dev_src = load_token_lines(DATA_DIR / "subword" / "dev.ja.sp")
dev_ref = [" ".join(line) for line in load_token_lines(DATA_DIR / "subword" / "dev.en.sp")]
beam_sizes = [1, 2, 5, 10]
scores = []
for beam in beam_sizes:
    score, _ = evaluate_bleu_for_beam(model, dev_src, dev_ref, src_vocab, tgt_vocab, beam, max_examples=200, device=device)
    scores.append(score)
    print(f"subword beam={beam}: BLEU={score:.2f}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
write_json(OUTPUT_DIR / "knock95_subword_beam_bleu_limited.json", dict(zip(beam_sizes, scores)))
plt.figure(figsize=(8, 5))
plt.plot(beam_sizes, scores, marker="o")
plt.xlabel("Beam size")
plt.ylabel("BLEU")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "knock95_subword_beam_bleu_limited.png", dpi=120)
PY

python -m torch.distributed.launch --nproc_per_node=8 --use_env knock96.py \
  --epochs 1 \
  --batch-size 32 \
  --max-train-examples 40000 \
  --max-dev-examples 200 \
  --max-train-bleu-examples 200 \
  --log-dir outputs/tensorboard_limited \
  2>&1 | tee ../logs/ch14_knock96_limited.log

python -m torch.distributed.launch --nproc_per_node=8 --use_env knock97.py \
  --batch-sizes 32,64 \
  --lrs 1e-4,1e-5 \
  --optimizers adam,adamw \
  --epochs 1 \
  --max-train-examples 10000 \
  --max-dev-examples 100 \
  2>&1 | tee ../logs/ch14_knock97_limited.log

python knock98.py \
  --ja-path "../Chapter 13/data/processed/test.ja.tok" \
  --en-path "../Chapter 13/data/processed/test.en.tok" \
  --output checkpoints/domain_adapted_mt_limited.pt \
  --epochs 1 \
  --batch-size 32 \
  --max-lines 10000 \
  2>&1 | tee ../logs/ch14_knock98_limited.log

python - <<'PY' 2>&1 | tee ../logs/ch14_knock99_smoke.log
from pathlib import Path

from knock99 import create_app

checkpoint = Path("checkpoints/domain_adapted_mt_limited.pt")
if not checkpoint.exists():
    checkpoint = Path("../Chapter 13/checkpoints/transformer_mt.pt")

app = create_app(str(checkpoint), beam_size=2, max_len=40)
client = app.test_client()
response = client.post("/", data={"sentence": "京都 は 美しい 。"})
print("status_code:", response.status_code)
print("contains_result:", b"Result" in response.data)
print("checkpoint:", checkpoint)
PY

echo "===== Chapter 14 finish run finished: $(date) ====="
