"""
Official-metric helpers: spBLEU (sacreBLEU + FLORES-200 SentencePiece
tokenizer) and chrF++, matching the task's stated primary/secondary metrics.

Gotcha (documented for reproducibility): sacrebleu's built-in `flores200`
tokenizer downloads its SentencePiece model from `https://tinyurl.com/...`,
which is unreachable from this environment. The FLORES-200 SPM model is the
exact same artifact NLLB-200 was trained with (both are the "SPM-200"
tokenizer from the NLLB paper), so we point sacrebleu at a local copy of the
cached `facebook/nllb-200-distilled-600M` tokenizer's
`sentencepiece.bpe.model` instead. Done once via `setup_local_flores200_spm()`.
"""
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from sacrebleu.metrics import BLEU, CHRF

_SACREBLEU_SPM_DIR = Path.home() / ".sacrebleu" / "models"
_SACREBLEU_SPM_PATH = _SACREBLEU_SPM_DIR / "flores200sacrebleuspm"


def _resolve_nllb_spm_path():
    """Resolve the cached sentencepiece.bpe.model via the HF cache API instead
    of a hardcoded absolute path -- portable across machines/usernames/cache
    locations (HF_HOME etc.) and across whichever commit hash the local cache
    happens to have, unlike a baked-in path to one specific snapshot dir."""
    return Path(hf_hub_download(
        "facebook/nllb-200-distilled-600M", "sentencepiece.bpe.model", local_files_only=True,
    ))


def setup_local_flores200_spm():
    _SACREBLEU_SPM_DIR.mkdir(parents=True, exist_ok=True)
    if not _SACREBLEU_SPM_PATH.exists():
        nllb_spm = _resolve_nllb_spm_path()
        shutil.copy(nllb_spm, _SACREBLEU_SPM_PATH)


_bleu = None
_chrf = None


def get_metrics():
    global _bleu, _chrf
    if _bleu is None:
        setup_local_flores200_spm()
        _bleu = BLEU(tokenize="flores200")
        _chrf = CHRF(word_order=2)  # chrF++
    return _bleu, _chrf


def corpus_scores(hyps, refs):
    """hyps: list[str]; refs: list[str] (single reference per hyp). Returns dict."""
    bleu, chrf = get_metrics()
    b = bleu.corpus_score(hyps, [refs])
    c = chrf.corpus_score(hyps, [refs])
    return {"spBLEU": round(b.score, 3), "chrF++": round(c.score, 3), "n": len(hyps)}
