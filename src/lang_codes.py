"""
Dialect name <-> NLLB-200 FLORES-200 language-code mapping for DialectMTBench.

Verified directly against the cached facebook/nllb-200-distilled-600M tokenizer
(see EDA/EDA_INSIGHTS.md §9): NLLB-200 ships dedicated Arabic-script codes for
every one of this task's 7 varieties.
"""

# Codabench task varieties (7): MSA + Palestinian, Moroccan, Tunisian, Egyptian, Lebanese, Saudi
DIALECT_TO_NLLB = {
    "MSA": "arb_Arab",           # Modern Standard Arabic
    "Palestinian": "ajp_Arab",   # South Levantine Arabic
    "Lebanese": "apc_Arab",      # North Levantine Arabic
    "Moroccan": "ary_Arab",      # Moroccan Arabic
    "Tunisian": "aeb_Arab",      # Tunisian Arabic
    "Egyptian": "arz_Arab",      # Egyptian Arabic
    "Saudi": "ars_Arab",         # Najdi Arabic (closest NLLB proxy for Saudi/Gulf)
}
NLLB_TO_DIALECT = {v: k for k, v in DIALECT_TO_NLLB.items()}

# The 6 dialects that actually appear in the blind test set (no MSA there — see EDA §2)
TEST_DIALECTS = ["Egyptian", "Lebanese", "Moroccan", "Palestinian", "Saudi", "Tunisian"]

# Dialects that have real, human-written monolingual text with a recoverable
# 77-way banking-intent label in public ArBanking77 (EDA §5) -> eligible for
# intent-mined silver pairs. Egyptian/Lebanese are NOT in this set (EDA §5/§6).
LABELED_DIALECTS = ["MSA", "Palestinian", "Moroccan", "Saudi", "Tunisian"]

# Dialects with zero public data of any kind -> synthetic/backtranslated only.
COLD_START_DIALECTS = ["Egyptian", "Lebanese"]


def to_nllb(dialect: str) -> str:
    try:
        return DIALECT_TO_NLLB[dialect]
    except KeyError:
        raise ValueError(f"Unknown dialect '{dialect}'. Known: {list(DIALECT_TO_NLLB)}")
