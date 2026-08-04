import re
import sys
from collections import Counter

import pandas as pd

MARKERS = {
    "Moroccan": ["ديالي", "ديال", "واش", "بغيت", "دابا", "غادي", "ماشي", "كاين", "علاش", "بزاف", "كيفاش", "نتا", "راه"],
    "Saudi": ["وش", "ابغى", "أبغى", "كذا", "ابي", "أبي", "الحين", "وايد", "زين", "تراني", "ليش", "احين"],
    "Tunisian": ["نجم", "باش", "برشة", "توا", "هكا", "كيفاش", "ماعندي", "قاعد", "شنية", "يزي", "نحب", "فما"],
    "Palestinian": ["هيك", "هلق", "بدي", "شو", "ليش", "منيح", "كتير", "هاد", "هاي", "بدنا"],
    "Egyptian": ["مش", "عايز", "عاوز", "ازاي", "إزاي", "ليه", "دلوقتي", "كده", "أهو", "علشان", "خالص", "بتاع", "اهو"],
    "Lebanese": ["شو", "هيك", "هلق", "مش", "بدي", "كتير", "هيدا", "هيدي", "منيح", "تبعي", "تبعك"],
    "MSA": ["إن", "الذي", "التي", "لماذا", "ماذا", "هل", "إذا", "لأن", "لكي"],
}


def simple_tokenize(text):
    return re.findall(r"[؀-ۿ]+", str(text))


def main(csv_path):
    df = pd.read_csv(csv_path)
    print(f"=== Dialect-marker audit: {csv_path} ===\n")
    print("Rows = actual output's target dialect. Cols = marker-set. Self-column (bold via >) is what should be HIGH.")
    print(f"{'Target dialect':<14}" + "".join(f"{m:>12}" for m in MARKERS.keys()))
    for tgt_dialect, group in df.groupby("Target dialect"):
        text = " ".join(group["Target sentence"].astype(str).tolist())
        toks = simple_tokenize(text)
        n_tokens = max(len(toks), 1)
        counts = Counter(toks)
        row = []
        for mdialect, mwords in MARKERS.items():
            c = sum(counts.get(w, 0) for w in mwords)
            freq = round(1000 * c / n_tokens, 2)
            marker = ">" if mdialect == tgt_dialect else " "
            row.append(f"{marker}{freq:>10}")
        print(f"{tgt_dialect:<14}" + "".join(row) + f"   (n_rows={len(group)}, n_tokens={n_tokens})")


if __name__ == "__main__":
    main(sys.argv[1])
