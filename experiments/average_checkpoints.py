"""
Post-training, zero-extra-GPU-time enhancement: average the LoRA adapter
weights across the last N epoch checkpoints (checkpoint averaging / a form of
stochastic weight averaging), a well-established, cheap technique for
smoothing out noise from the final epochs of NMT training.

`train_nllb_lora.py` already uses `load_best_model_at_end` to pick the single
best checkpoint by eval_loss and saves it to `.../adapter/`. This script
builds an ALTERNATIVE candidate (`.../adapter_averaged/`) from the last
`--n` epoch checkpoints still on disk (save_total_limit keeps a few around).
Both are then evaluated (see evaluate.py) and whichever scores higher on the
real dev sets is the one used for final predictions -- costs nothing but a
few seconds of CPU tensor averaging plus one extra eval pass.

Run: conda run -n mo python average_checkpoints.py --run-dir ../outputs/checkpoints/nllb600m_lora_dialectmt_v2 --n 3
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src"))

import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=3, help="average the last N checkpoints found")
    ap.add_argument("--out-name", default="adapter_averaged")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if len(ckpts) < 2:
        print(f"Only {len(ckpts)} checkpoint(s) found in {run_dir} -- averaging needs >=2, skipping.")
        return
    ckpts = ckpts[-args.n:]
    print(f"Averaging {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

    state_dicts = [load_file(str(c / "adapter_model.safetensors")) for c in ckpts]
    keys = state_dicts[0].keys()
    assert all(sd.keys() == keys for sd in state_dicts), "checkpoint adapter shapes/keys don't match"

    averaged = {}
    for k in keys:
        stacked = sum(sd[k].float() for sd in state_dicts) / len(state_dicts)
        averaged[k] = stacked.to(state_dicts[0][k].dtype)

    out_dir = run_dir / args.out_name
    out_dir.mkdir(exist_ok=True, parents=True)
    save_file(averaged, str(out_dir / "adapter_model.safetensors"))

    # copy the (identical across checkpoints) config/tokenizer files from the last checkpoint
    for fname in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = ckpts[-1] / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    meta = {"averaged_from": [c.name for c in ckpts], "n_checkpoints": len(ckpts)}
    (out_dir / "AVERAGING_INFO.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote averaged adapter to {out_dir}")


if __name__ == "__main__":
    main()
