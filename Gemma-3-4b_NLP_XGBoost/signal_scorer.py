"""
signal_scorer.py  —  SHIM (single source of truth)
---------------------------------------------------
This file is intentionally NOT a copy of the scorer. The real, canonical
implementation lives in ../DP/signal_scorer.py. Previously this folder held a
near-identical duplicate that differed only in two hardcoded paths; the two
copies could (and did) drift out of sync, which silently breaks feature
alignment between training-data scoring (DP) and prediction (here).

Now this module just loads ../DP/signal_scorer.py and re-exports its public API,
so `import signal_scorer as ss` in this folder uses the EXACT same code, config,
and feature formulas as the DP batch pipeline. Edit the scorer in ONE place: DP.

Public API is unchanged (CONFIG, extract_policy_flags, extract_ner_features,
compute_composite_scores, load_spacy, load_sbert, embedding_score,
novelty_score, burst_position_score, relative_signal_strength,
score_single_post, classify_post, main, ...).
"""
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANONICAL = os.path.normpath(os.path.join(_HERE, "..", "DP", "signal_scorer.py"))

if not os.path.exists(_CANONICAL):
    raise FileNotFoundError(
        f"❌ Canonical scorer not found at {_CANONICAL}. "
        f"signal_scorer.py here is a shim that re-exports ../DP/signal_scorer.py."
    )

# Load the canonical module under a distinct name so it does not collide with
# this shim (both files are named signal_scorer.py).
_spec = importlib.util.spec_from_file_location("signal_scorer_canonical", _CANONICAL)
_canon = importlib.util.module_from_spec(_spec)
sys.modules["signal_scorer_canonical"] = _canon
_spec.loader.exec_module(_canon)

# Re-export every public name. The function objects keep the canonical module's
# globals (CONFIG, POLICY_FLAGS, ...), so they behave identically here.
globals().update({k: v for k, v in vars(_canon).items() if not k.startswith("_")})

# Keep a handle to the underlying module for anyone who wants it.
canonical = _canon


if __name__ == "__main__":
    main()  # noqa: F821  (re-exported from the canonical module)
