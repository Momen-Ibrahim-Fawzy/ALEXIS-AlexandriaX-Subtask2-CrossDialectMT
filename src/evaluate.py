"""
Stage 6: evaluate baseline (zero-shot) vs. LoRA fine-tuned NLLB on:

  1. gold_dev.jsonl        -- real, human-written MSA<->Palestinian pairs, leak-
                                free w.r.t. the blind test file (hard numbers).
  2. finetune_dev_soft.jsonl -- held-out bronze-reverse (Egyptian/Lebanese as
                                source) pairs. Soft signal only: the "reference"
                                side is itself a real sentence but the pairing
                                is synthetic-generation-derived, not a human
                                translation -- reported separately and labeled
                                as such, never mixed into the headline numbers.

Writes outputs/dev_eval_report.json and a human-readable .md summary with
per-dialect-pair breakdowns and qualitative samples across ALL 6 test dialects.

Run: CUDA_VISIBLE_DEVICES=1 conda run -n mo python evaluate.py [--adapter PATH]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed.tensor  # noqa: F401 -- peft 0.20's PeftModel.from_pretrained checks
                                  # isinstance(x, torch.distributed.tensor.DTensor) but doesn't
                                  # import the submodule itself; in a non-distributed single-
                                  # process script it's never loaded otherwise, so the attribute
                                  # access raises AttributeError without this explicit import.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nllb_utils import translate
from lang_codes import NLLB_TO_DIALECT, TEST_DIALECTS
from spbleu_chrf import corpus_scores

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"

SAMPLE_SENTENCES = {
    "MSA": "ماذا افعل إن لم أستلم بطاقتي الجديدة؟",
}


def read_jsonl(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_model(model_name, adapter_path, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device).eval()
    return tok, model


def eval_gold_dev(tok, model, device, batch_size=8):
    recs = read_jsonl(DATA / "gold_dev.jsonl")
    if not recs:
        return {}
    results = {}
    for direction, src_key, tgt_key, src_d, tgt_d in [
        ("MSA->Palestinian", "msa", "pal", "MSA", "Palestinian"),
        ("Palestinian->MSA", "pal", "msa", "Palestinian", "MSA"),
    ]:
        hyps, refs = [], []
        for i in range(0, len(recs), batch_size):
            batch = recs[i:i + batch_size]
            srcs = [r[src_key] for r in batch]
            outs = translate(model, tok, srcs, src_d, tgt_d, device, num_beams=4)
            hyps.extend(outs)
            refs.extend(r[tgt_key] for r in batch)
        results[direction] = corpus_scores(hyps, refs)
    return results


def eval_by_direction(tok, model, device, jsonl_path, batch_size=8):
    """Generic per-(src_lang,tgt_lang)-direction eval over a {src_lang, tgt_lang,
    src_text, tgt_text} jsonl file. Used for both finetune_dev_soft.jsonl
    (bronze-reverse proxy, Egyptian/Lebanese) and silver_dev.jsonl (mined,
    real-native-text proxy, the 20 directions among the 5 labeled dialects --
    i.e. most of the pairs that actually matter for the test-set score)."""
    recs = read_jsonl(jsonl_path)
    if not recs:
        return {}
    by_pair = defaultdict(list)
    for r in recs:
        by_pair[(r["src_lang"], r["tgt_lang"])].append(r)
    results = {}
    for (src_code, tgt_code), items in by_pair.items():
        src_d, tgt_d = NLLB_TO_DIALECT[src_code], NLLB_TO_DIALECT[tgt_code]
        hyps, refs = [], []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            srcs = [r["src_text"] for r in batch]
            outs = translate(model, tok, srcs, src_d, tgt_d, device, num_beams=4)
            hyps.extend(outs)
            refs.extend(r["tgt_text"] for r in batch)
        results[f"{src_d}->{tgt_d}"] = corpus_scores(hyps, refs)
    if results:
        results["MACRO_AVG"] = {
            "spBLEU": round(sum(v["spBLEU"] for v in results.values()) / len(results), 3),
            "chrF++": round(sum(v["chrF++"] for v in results.values()) / len(results), 3),
            "n_directions": len(results),
        }
    return results


def qualitative_samples(tok, model, device):
    samples = []
    src_text = SAMPLE_SENTENCES["MSA"]
    for tgt in TEST_DIALECTS:
        out = translate(model, tok, [src_text], "MSA", tgt, device, num_beams=4)[0]
        samples.append({"src_dialect": "MSA", "src_text": src_text, "tgt_dialect": tgt, "hyp": out})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir; omit for zero-shot baseline")
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M", help="NLLB backbone to evaluate")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load_model(args.model, args.adapter, device)

    report = {"tag": args.tag, "adapter": args.adapter, "model": args.model}
    print(f"[{args.tag}] evaluating gold_dev (real MSA<->Palestinian, leak-free) ...")
    report["gold_dev"] = eval_gold_dev(tok, model, device)
    print(json.dumps(report["gold_dev"], indent=2, ensure_ascii=False))

    print(f"[{args.tag}] evaluating silver_dev (mined, real-native-text proxy, "
          f"the 20 directions among Moroccan/Saudi/Tunisian/Palestinian/MSA -- "
          f"most of the actual test-scoring pairs) ...")
    report["silver_dev"] = eval_by_direction(tok, model, device, DATA / "silver_dev.jsonl")
    print(json.dumps(report["silver_dev"].get("MACRO_AVG", {}), indent=2, ensure_ascii=False))

    print(f"[{args.tag}] evaluating soft dev (synthetic-derived, Egyptian/Lebanese) ...")
    report["soft_dev"] = eval_by_direction(tok, model, device, DATA / "finetune_dev_soft.jsonl")
    print(json.dumps(report["soft_dev"], indent=2, ensure_ascii=False))

    print(f"[{args.tag}] qualitative samples across all 6 test dialects ...")
    report["qualitative_samples"] = qualitative_samples(tok, model, device)
    for s in report["qualitative_samples"]:
        print(f"  {s['tgt_dialect']:>12}: {s['hyp']}")

    OUT.mkdir(exist_ok=True, parents=True)
    out_path = OUT / f"dev_eval_report_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
