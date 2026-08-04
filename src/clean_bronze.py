import json
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    recs = [json.loads(l) for l in f]

n_cleaned = 0
for r in recs:
    for key in ("src_text", "tgt_text"):
        cleaned = re.sub(r"^[_\s]+", "", str(r[key])).strip()
        if cleaned != r[key]:
            n_cleaned += 1
        r[key] = cleaned

with open(path, "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Cleaned {n_cleaned} field(s) with a leading-underscore/whitespace artifact across {len(recs)} records.")
