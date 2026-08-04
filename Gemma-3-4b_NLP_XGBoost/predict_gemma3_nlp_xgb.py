"""
predict_gemma3_nlp_xgb.py
-------------------------
Predict: post -> [Gemma-3-4B embedding + NLP scorer features] -> XGBoost -> impacts.
Full standalone Gemma port of the pipeline (gemma retired 2026-07-17 after
the NLP-only challenger proved its embeddings added nothing). All gates
(NLP sigmoid, temporal, endorsement, self-news, chain, reiteration,
commentary, event-window domain gating) and magnitude calibration included.

Requires signal_scorer.py (shim) + gemma_embedder.py in this folder.
Run:  uv run python predict_gemma3_nlp_xgb.py
"""
import os
# FA2 install is broken in this venv — tell unsloth to go straight to
# Xformers instead of probing FA2 and warning (same speed either way).
os.environ.setdefault("UNSLOTH_FORCE_XFORMERS", "1")
try:
    import unsloth  # noqa: F401 — must import BEFORE transformers for its patches
except ImportError:
    pass
import re, json, argparse, datetime
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import signal_scorer as ss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NY = 'America/New_York'

_HERE          = os.path.dirname(os.path.abspath(__file__))
_ENTITIES_FILE = os.path.join(_HERE, "..", "DP", "influence_accounts.json")

# ---------------------------------------------------------------------------
# EVENT-WINDOW DOMAIN GATING (single source: DP/event_manager.py, loaded by
# path like the signal_scorer shim). A keyword domain (crypto, COVID, Fed...)
# only contributes to the NLP signal while a matching events.json window is
# LIVE at the post date — a 2017 'vaccine' post is not a pandemic post.
# Mirrors DP/build_final_training_set (training-side) exactly.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
_EM_PATH = os.path.normpath(os.path.join(_HERE, "..", "DP", "event_manager.py"))
_em_spec = _ilu.spec_from_file_location("event_manager_canonical", _EM_PATH)
_em_mod  = _ilu.module_from_spec(_em_spec)
_em_spec.loader.exec_module(_em_mod)
_EM = _em_mod.EventManager()

_MACRO_COMPONENTS = [        # (flag column, weight, events.json domain)
    ('flag_crypto_policy',    4.0, 'crypto_policy'),
    ('flag_public_health',    4.0, 'public_health'),
    ('flag_interest_rate',    2.0, 'interest_rate'),
    ('flag_financial_system', 2.0, 'financial_system'),
    ('flag_stimulus',         1.0, 'stimulus'),
    ('flag_tax_policy',       1.0, 'tax_policy'),
    ('flag_energy_policy',    1.0, 'energy_policy'),
    ('flag_ai_chip_policy',   1.0, 'ai_chip_policy'),
]


def gemma_sentiment_from_emb(emb, cfg):
    """gemma's DESIGNED output (Araci 2019): 3-class sentiment
    probabilities from the model's own pooler+classifier applied to raw
    CLS vectors (first 768 dims of the cached 1536 embedding).
    Returns (n,3) [P(pos), P(neg), P(pos)-P(neg)], or None if the model
    set was trained without the sentiment head."""
    h = cfg.get("_sent_head")
    if h is None:
        return None
    pool = np.tanh(emb[:, :768] @ h["Wp"].T + h["bp"])
    lg = pool @ h["Wc"].T + h["bc"]
    lg = lg - lg.max(axis=1, keepdims=True)
    pr = np.exp(lg); pr /= pr.sum(axis=1, keepdims=True)
    p, n = int(h["pos"]), int(h["neg"])
    return np.column_stack([pr[:, p], pr[:, n],
                            pr[:, p] - pr[:, n]]).astype(np.float32)


def project_emb(emb, cfg):
    """Apply the training-time PCA projection when the model set was trained
    on compressed embeddings (config 'emb_pca'); identity for older models."""
    mats = cfg.get("_emb_pca_mats")
    if mats is None:
        return emb
    mean, comps = mats
    return ((emb - mean) @ comps.T).astype(np.float32)


def gated_macro_score(feats, post_date):
    """macro_risk_score recomputed with event-window gates (0..~14, /5 norm).
    `feats` is any dict-like with flag_* entries; `post_date` a datetime.date."""
    s = 0.0
    for colf, wt, domn in _MACRO_COMPONENTS:
        v = feats.get(colf)
        if v is not None and not pd.isna(v) and float(v) > 0:
            g = _EM.domain_activity(domn, post_date)
            if g > 0:
                s += wt * float(v) * g
    return s

def _rank0_handle(when=None) -> str:
    """Return the rank-0 TruthSocial primary handle ACTIVE at `when` (default now).

    primary_accounts now holds MULTIPLE rank-0 entries across eras/platforms
    (realDonaldTrump on X 2017-2021, POTUS on X 2021-2025, realDonaldTrump on
    TruthSocial 2024-2028). For a LIVE prediction we want the TruthSocial rank-0
    primary whose [active_from, expiration_date] window contains the prediction
    time -- not just the first one listed. Falls back to the first TruthSocial
    rank-0, then to 'us_president'.
    """
    try:
        with open(_ENTITIES_FILE, encoding="utf-8") as f:
            accounts = json.load(f).get("primary_accounts", [])
    except Exception:
        return "us_president"

    def _p(x):
        if x is None or str(x).strip().upper() in ("", "N/A", "NONE", "NULL"):
            return None
        try:
            return pd.Timestamp(x, tz="UTC")
        except Exception:
            return None

    now = pd.Timestamp(when, tz="UTC") if when is not None else pd.Timestamp.now(tz="UTC")
    ts = []
    for a in accounts:
        try:
            if int(a.get("rank", 99)) != 0:
                continue
        except (TypeError, ValueError):
            continue
        if a.get("platform") == "truthsocial":
            ts.append(a)

    # prefer the rank-0 TruthSocial primary whose active window contains `now`
    for a in ts:
        lo = _p(a.get("active_from"))
        hi = _p(a.get("active_to")) or _p(a.get("expiration_date"))
        if (lo is None or now >= lo) and (hi is None or now <= hi):
            return str(a.get("account", "")).lstrip("@") or "us_president"
    if ts:
        return str(ts[0].get("account", "")).lstrip("@") or "us_president"
    return "us_president"


def _build_handle_country_map() -> dict:
    """
    Build {handle_lower: country_alpha2} from all entries in influence_accounts.json.
    Covers primary_accounts, entities and institutions (all by 'account', with
    'twitter_handle' as a legacy fallback).
    """
    try:
        with open(_ENTITIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    out: dict = {}
    for a in data.get("primary_accounts", []):
        h = str(a.get("account") or a.get("twitter_handle") or "").lstrip("@")
        c = a.get("country", "")
        if h and c:
            out[h.lower()] = c
    # entities & institutions use the 'account' field (e.g. "@netanyahu",
    # "@DeptofWar"); older records may still carry 'twitter_handle'.
    for section in (data.get("entities", []),
                    data.get("institutions", {}).get("entries", [])):
        for e in section:
            h = str(e.get("account") or e.get("twitter_handle") or "").lstrip("@")
            c = e.get("country", "")
            if h and c:
                out[h.lower()] = c
    return out


_HANDLE_COUNTRY: dict = {}   # loaded lazily once


def _country_for(handle: str) -> str:
    """Return ISO alpha-2 country code for a handle, or '' if unknown."""
    global _HANDLE_COUNTRY
    if not _HANDLE_COUNTRY:
        _HANDLE_COUNTRY = _build_handle_country_map()
    return _HANDLE_COUNTRY.get(str(handle).lower(), "")

OUT_DIR = "gemma3_nlp_xgb_models"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

GATE_ENABLED = True
GATE_K       = 16.0
GATE_MID     = 0.45

TRADE_THRESHOLD = 0.1

TEMPORAL_PAST_PHRASES = [
    r"last night", r"yesterday", r"earlier today",
    r"overnight", r"earlier this week", r"last week", r"last month",
    r"a (?:few|couple of) (?:hours|days) ago", r"this past (?:weekend|week)",
    # weekly/periodic RECAPS ("Over the past week, Ukraine has endured...") —
    # summaries of already-public events; a Sunday recap must not wear
    # Monday's macro move (the DeepSeek-crash label incident, post #14)
    r"over the past (?:week|month|year|\d+ (?:days|weeks|months))",
]
TEMPORAL_FUTURE_PHRASES = [
    r"tomorrow", r"next week", r"next month", r"next year",
    r"this weekend", r"later today",
    r"soon", r"shortly", r"upcoming", r"in the coming (?:days|weeks|months)",
]
TEMPORAL_BREAKING = [
    r"\bbreaking\b", r"\bjust announced\b", r"\bmoments ago\b", r"\bright now\b",
    r"\bhappening now\b", r"\bas we speak\b", r"\bat this (?:very )?moment\b",
    r"\bat my direction\b", r"\bhereby announc(?:e|ing|ed)\b",
    r"\beffective immediately\b", r"\bwill begin immediately\b",
    r"\bwe got (?:him|her|them)\b",
    r"\bi (?:have |'ve |am |will |just |now |already |strongly |"
    r"formally |personally |immediately )*(?:ordered|directed|authorized|"
    r"launched|requested|demanded|instructed|called|spoke|"
    r"warned|signed|imposed|decided|asked|urged|pressed|"
    r"communicated|made clear|made it clear|finalized|concluded|reached a|"
    r"secured|completed|confirmed|announced)\b",
    r"\ba (?:letter|statement|message) (?:sent|addressed|delivered|issued)\b",
    r"\bnot anymore\b",
]
TEMPORAL_AMBIGUOUS = {
    r"this morning":   (5, 12),
    r"this afternoon": (12, 18),
    r"this evening":   (18, 24),
    r"tonight":        (18, 24),
}
TEMPORAL_DAMP = 0.15

TEMPORAL_PAST_DAMP_PREMARKET  = 0.5
TEMPORAL_PAST_DAMP_SESSION    = 0.15
TEMPORAL_PAST_DAMP_AFTERHOURS = 0.3

# Poster-local timezones: time-of-day phrases ("last night", "this morning")
# are read on the POSTER's clock, not New York's. Zelensky's "last night" at
# 03:41 EDT is 10:41 in Kyiv. Covers every country in influence_accounts.json.
COUNTRY_TZ = {
    'US': 'America/New_York',  'UA': 'Europe/Kyiv',      'RU': 'Europe/Moscow',
    'IL': 'Asia/Jerusalem',    'IR': 'Asia/Tehran',      'CN': 'Asia/Shanghai',
    'GB': 'Europe/London',     'FR': 'Europe/Paris',     'DE': 'Europe/Berlin',
    'IT': 'Europe/Rome',       'IN': 'Asia/Kolkata',     'SA': 'Asia/Riyadh',
    'AE': 'Asia/Dubai',        'BY': 'Europe/Minsk',     'AM': 'Asia/Yerevan',
    'AZ': 'Asia/Baku',         'CA': 'America/Toronto',  'MX': 'America/Mexico_City',
    'BR': 'America/Sao_Paulo', 'AR': 'America/Argentina/Buenos_Aires',
    'VE': 'America/Caracas',
}

# WAR-REPORT FIRST DISCLOSURE (geo accounts): an operational summary like
# "Last night, FP-5 Flamingo missiles successfully struck the Titan-Barrikady
# facility" is usually the FIRST public confirmation of the event — the
# market reacts at post time, so a past-time phrase must NOT damp it.
# (US primary account relays like "I was informed last night..." stay damped —
# those describe already-public info; first-person ACTIONS are separately
# covered by TEMPORAL_BREAKING.)
WAR_REPORT_PATTERNS = [
    r"\bsuccessfully struck\b",
    r"\bstruck\b[^.!?\n]{0,80}\b(?:facility|plant|refinery|depot|base|complex|"
    r"infrastructure|airfield|warehouse)\b",
    r"\bour (?:forces|troops|warriors|defenders|soldiers|air defense|missiles|"
    r"drones|long-range|units)\b",
    r"\b(?:was|were) (?:hit|destroyed|eliminated|neutralized|downed)\b",
    r"\bshot down\b",
    r"\bstrikes? (?:on|against)\b",
    r"\bcombat (?:operations|missions|work|engagements)\b",
    r"\bfrontline\b", r"\bair defense\b",
]


def temporal_factor(text, post_hour=None, post_ts=None, country=None):
    """Return (factor, label) for the predict-time temporal gate.

    post_hour : NY-local hour (legacy path, used when post_ts/country absent).
    post_ts   : tz-aware post timestamp; combined with `country` it yields the
                POSTER-LOCAL hour for interpreting time-of-day phrases.
    country   : ISO alpha-2 of the posting account (influence_accounts.json).
    """
    t = text.lower()
    has_future_time = any(re.search(p, t) for p in TEMPORAL_FUTURE_PHRASES)
    has_breaking     = any(re.search(p, t) for p in TEMPORAL_BREAKING)
    has_past_time    = any(re.search(p, t) for p in TEMPORAL_PAST_PHRASES)

    is_geo = bool(country) and country != "US"

    # war-report exemption BEFORE any past-time damping
    if is_geo and has_past_time and any(re.search(p, t) for p in WAR_REPORT_PATTERNS):
        return 1.0, f"war-report/first-disclosure ({country} account -- new info at post time)"

    # poster-local hour for phrase interpretation
    local_hour, clock = post_hour, "NY"
    if post_ts is not None and country in COUNTRY_TZ:
        try:
            local_hour = post_ts.tz_convert(COUNTRY_TZ[country]).hour
            clock = country
        except Exception:
            pass

    if has_past_time and not (has_future_time or has_breaking):
        if local_hour is None:
            return TEMPORAL_DAMP, "past/stale"
        if is_geo:
            # geo poster: NY session tiers don't apply to their clock — flat damp
            return TEMPORAL_DAMP, (f"past/stale (geo account, {clock} local "
                                   f"{local_hour:02d}h, non-operational recap)")
        if local_hour < 9:
            return TEMPORAL_PAST_DAMP_PREMARKET, "past/stale (posted premarket -- not yet traded by cash equities)"
        if local_hour < 16:
            return TEMPORAL_PAST_DAMP_SESSION, "past/stale (posted during session -- already traded)"
        return TEMPORAL_PAST_DAMP_AFTERHOURS, "past/stale (posted after close -- only FX/crypto/futures react overnight)"
    if has_future_time or has_breaking:
        return 1.0, "future/new-info"

    if local_hour is not None:
        for phrase, (start, end) in TEMPORAL_AMBIGUOUS.items():
            if re.search(phrase, t):
                if local_hour >= end:
                    return TEMPORAL_DAMP, f"past/stale ('{phrase.strip()}' already over on {clock} clock)"
                return 1.0, f"future/ongoing ('{phrase.strip()}' still ahead on {clock} clock)"

    return 1.0, "neutral"


# ---------------------------------------------------------------------------
# ENDORSEMENT / CEREMONIAL-NOISE GATE
# Political endorsements ("Congressman X is a Tremendous Advocate...") are
# stuffed with policy buzzwords (border, taxes, military, regulations), so
# policy_intensity_score rates them like policy announcements and the NLP
# gate lets them through (observed: NLP signal 0.68-0.75, gate x0.99, TRADE).
# They carry ZERO tradeable information — a biography of someone's positions
# is not an action. Detected -> factor 0.0 -> never traded (hard skip).
# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH: the patterns now live in DP/scorer_config.json and
# are applied by DP/signal_scorer.py at SCORING time, so endorsement posts
# never enter training, the embedding cache, HIGH_SIGNAL or the Gemma analyst
# run. This gate stays as the last line of defence for live posts, reading the
# SAME list — a divergent local copy is how scoring and trading drift apart.
ENDORSEMENT_PATTERNS = list(getattr(ss, "ENDORSEMENT_PATTERNS", []))
if not ENDORSEMENT_PATTERNS:                 # older scorer_config.json
    ENDORSEMENT_PATTERNS = [
        r"complete and total endorsement",
        r"has my (?:complete|full|total|strong)[^.!?\n]{0,30}endorsement",
        r"\brunning for (?:re-?election|congress|the senate|senate|governor|office)\b",
        r"\b(?:get out and )?vote for\b",
        r"\b(?:great )?honor to endorse\b",
        r"\bhappy birthday\b",
    ]
    print("  ⚠️  scorer_config.json has no endorsement_patterns — using the "
          "built-in fallback list (add them to the config to keep scoring and "
          "trading in sync)")
ENDORSEMENT_DAMP = float(getattr(ss, "ENDORSEMENT_DAMP", 0.0))


def endorsement_factor(text):
    """Return (factor, label): (0.0, 'endorsement/noise') when the post is a
    political endorsement / ceremonial post, else (1.0, '')."""
    t = str(text).lower()
    for p in ENDORSEMENT_PATTERNS:
        if re.search(p, t):
            return ENDORSEMENT_DAMP, "endorsement/noise (hard skip)"
    return 1.0, ""


# ---------------------------------------------------------------------------
# SELF-NEWS-SHARE GATE ("Trump upends China's oil scheme: https://...")
# A poster sharing press coverage ABOUT THEMSELVES is relaying something that
# ALREADY HAPPENED and was already published — the market priced it when the
# article ran, not when the re-post appears. The tell: the poster's own name
# as a THIRD-PERSON subject (people don't use their own name as a pronoun
# when announcing their own action) + a URL + no first-person pronouns.
# First-person posts ("I have just signed... https://") are never gated.
# ---------------------------------------------------------------------------
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i'm|i've|i'll|i'd|my|me|we|we're|we've|our|us)\b", re.I)
_URL_ONLY_RE = re.compile(r"https?://\S+")

_HANDLE_NAME: dict = {}   # {handle_lower: display name} — lazy, from influence_accounts.json


def _display_name_for(handle: str) -> str:
    global _HANDLE_NAME
    if not _HANDLE_NAME:
        try:
            with open(_ENTITIES_FILE, encoding="utf-8") as f:
                data = json.load(f)
            for a in data.get("primary_accounts", []):
                h = str(a.get("account") or a.get("twitter_handle") or "").lstrip("@")
                if h:
                    _HANDLE_NAME[h.lower()] = a.get("account_name", a.get("name", ""))
            for section in (data.get("entities", []),
                            data.get("institutions", {}).get("entries", [])):
                for e in section:
                    h = str(e.get("account") or e.get("twitter_handle") or "").lstrip("@")
                    if h:
                        _HANDLE_NAME[h.lower()] = e.get("name", "")
        except Exception:
            _HANDLE_NAME = {"__failed__": ""}
    return _HANDLE_NAME.get(str(handle).lower(), "")


def self_news_share_factor(text, account=None, account_name=None):
    """(0.0, label) when the post is third-person press coverage of the
    poster's own (past) actions with a link; else (1.0, '')."""
    t = str(text)
    if not _URL_ONLY_RE.search(t):
        return 1.0, ""
    name = (account_name or "").strip() or (_display_name_for(account) if account else "")
    tokens = [w for w in re.split(r"[\s.]+", str(name)) if len(w) > 2]
    if not tokens:
        return 1.0, ""
    surname = tokens[-1].lower()
    body = _URL_ONLY_RE.sub("", t).strip()
    if not re.search(rf"\b{re.escape(surname)}\b", body, re.I):
        return 1.0, ""
    if _FIRST_PERSON_RE.search(body):
        return 1.0, ""
    return 0.0, "self-news-share (3rd-person coverage of own past action -- already priced)"


# ---------------------------------------------------------------------------
# CHAIN GUARD (PRODUCTION) — posts from the same account within CHAIN_WINDOW_MIN
# share ONE market reaction: the leader moves the market, followers are
# elaboration/recap whose reaction window is already priced (and whose labels
# were contaminated in training). Only the chain LEADER trades; a follower
# passes only when its NLP signal is materially STRONGER than the leader's
# (the bombshell came second — it then becomes the new leader).
#
# State: in-process _CHAIN_STATE for a long-running production loop feeding
# posts chronologically, with a posts_scored DB lookup as COLD-START fallback
# (first prediction after process start still knows about a post from 20
# minutes ago). Requires post_ts + account; without them (ad-hoc interactive
# text) the guard is inert.
# ---------------------------------------------------------------------------
CHAIN_WINDOW_MIN   = 60
CHAIN_SIGNAL_DELTA = 0.10
_CHAIN_STATE: dict = {}    # {account_lower: {"ts": Timestamp, "sig": float}}

# ---------------------------------------------------------------------------
# REITERATION DAMP — a post that mostly RESTATES the account's recent posts
# has little market power left ("the market already reacted to the powerful
# posts"). Detection: LOW semantic novelty vs the account's recent posts,
# applied ONLY when the temporal gate says 'neutral'. That last condition is
# critical and data-calibrated: the blockade bombshell itself had novelty
# 0.31 (familiar Iran/Hormuz vocabulary) but carries 'Effective immediately'
# -> future/new-info -> NEVER damped here. A next-day restatement with no
# new-action cue (novelty 0.36, temporal neutral) gets damped to ~its
# novelty. Historical novelty quantiles: p25=0.45, p50=0.61.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# COMMENTARY GATE — two no-new-action shapes that were quietly missing most
# (backtest audit 2026-07-04, 10 posts):
#   RETRO-BRAG: status/achievement commentary ("Munitions Stockpiles have
#   never been higher", "accomplished during my First Term", "Tariffs have
#   made our Country Rich") — often posted nights/weekends, so their label is
#   the NEXT SESSION's unrelated macro move. ×0.15.
#   CONDITIONAL THREAT: deterrence language ("Any further attack WILL BE met
#   with...") — a warning, not an action; markets are habituated. ×0.35.
# Applied ONLY when the temporal gate says 'neutral' — enacted bombshells
# ("Effective immediately... will be BLOWN TO HELL") carry breaking/future
# cues and are never touched.
# ---------------------------------------------------------------------------
RETRO_BRAG_PATTERNS = [
    r"\b(?:was|were|has been|have been) (?:accomplished|achieved|completed|"
    r"rebuilt|renovated|modernized)\b",
    r"\bduring my (?:first|second) term\b",
    r"\bhave (?:never )?been (?:higher|better|stronger|greater|safer)\b",
    r"\bnever been (?:higher|better|stronger|greater)\b",
    r"\b(?:made|has made|have made) our country (?:rich|strong|powerful|safe|great)\b",
    r"\bvirtually unlimited\b",
    r"\bbegan long before i took office\b",
    r"\bwould have never happened\b",
    r"\bmore (?:nuclear weapons|\w+) than any other country\b",
]
CONDITIONAL_THREAT_PATTERNS = [
    r"\bany (?:further|additional|new)\b[^.!?\n]{0,60}\bwill be\b",
    r"\bif\b[^.!?\n]{0,80}\bwill be (?:met|eliminated|destroyed|blown|hit)\b",
    r"\bwill be met with\b",
    r"\bwarning[:!]",
]
RETRO_BRAG_DAMP  = 0.15
COND_THREAT_DAMP = 0.35


def commentary_factor(text):
    """(factor, label) for no-new-action commentary; (1.0, '') otherwise.
    Caller must apply ONLY on temporally-neutral posts."""
    t = str(text).lower()
    for p in RETRO_BRAG_PATTERNS:
        if re.search(p, t):
            return RETRO_BRAG_DAMP, ("retrospective-brag/status commentary "
                                     "(no new action) x%.2f" % RETRO_BRAG_DAMP)
    for p in CONDITIONAL_THREAT_PATTERNS:
        if re.search(p, t):
            return COND_THREAT_DAMP, ("conditional threat/deterrence warning "
                                      "(no enacted action) x%.2f" % COND_THREAT_DAMP)
    return 1.0, ""


REITER_NOVELTY_MAX = 0.45   # damp below this (25th percentile of history)
REITER_FLOOR       = 0.20
REITER_MEMORY_HRS  = 24
REITER_MEMORY_N    = 10
_RECENT_POSTS: dict = {}    # {account_lower: [(ts, sbert_emb | token_set), ...]}


def reiteration_factor(text, account=None, post_ts=None, sbert=None,
                       novelty=None, remember=True):
    """
    (factor, label) — factor < 1 when the post is a low-novelty reiteration.
    novelty: pass posts_scored.score_novelty when available (backtest path);
    else it is computed against the in-process recent-post memory using the
    loaded SBERT model (production path; token-Jaccard fallback without it).
    """
    key = str(account).lower() if account else None
    nov = novelty

    if nov is None or (isinstance(nov, float) and nov != nov):   # None/NaN
        nov = 1.0
        if key is not None and post_ts is not None:
            hist = [h for h in _RECENT_POSTS.get(key, [])
                    if (post_ts - h[0]).total_seconds() <= REITER_MEMORY_HRS * 3600]
            cur_repr = None
            if sbert is not None:
                try:
                    cur_repr = sbert.encode([str(text)[:512]], show_progress_bar=False)[0]
                    sims = [float((cur_repr @ h[1]) /
                                  ((cur_repr @ cur_repr) ** .5 * (h[1] @ h[1]) ** .5))
                            for h in hist if not isinstance(h[1], (set, frozenset))]
                    if sims:
                        nov = 1.0 - max(sims)
                except Exception:
                    cur_repr = None
            if cur_repr is None:                                  # Jaccard fallback
                cur_repr = set(str(text).lower().split())
                sims = [len(cur_repr & h[1]) / max(len(cur_repr | h[1]), 1)
                        for h in hist if isinstance(h[1], (set, frozenset))]
                if sims:
                    nov = 1.0 - max(sims)
            if remember:
                _RECENT_POSTS[key] = (hist + [(post_ts, cur_repr)])[-REITER_MEMORY_N:]

    if nov >= REITER_NOVELTY_MAX:
        return 1.0, ""
    f = max(REITER_FLOOR, round(float(nov), 2))
    return f, (f"reiteration (novelty {nov:.2f} vs recent posts, no new-action "
               f"cue) -- market already reacted x{f:.2f}")


def _db_last_post_signal(account, before_ts):
    """Cold-start fallback: newest posts_scored row from `account` inside the
    chain window before `before_ts`. Returns (ts, signal) or None."""
    try:
        import db as _db
        lo = (before_ts - pd.Timedelta(minutes=CHAIN_WINDOW_MIN)).isoformat()
        r = _db.query(
            f"SELECT date, policy_intensity_score, hawkish_risk_score, sample_weight "
            f"FROM posts_scored WHERE lower(account) = '{str(account).lower()}' "
            f"AND date >= '{lo}' AND date < '{before_ts.isoformat()}' "
            f"ORDER BY date DESC LIMIT 1")
        if r is None or r.empty:
            return None
        row = r.iloc[0]
        parts = []
        if pd.notna(row.get('policy_intensity_score')):
            parts.append(min(float(row['policy_intensity_score']) / 8.0, 1.0))
        if pd.notna(row.get('hawkish_risk_score')):
            parts.append(min(float(row['hawkish_risk_score']) / 5.0, 1.0))
        if pd.notna(row.get('sample_weight')):
            parts.append(float(row['sample_weight']))
        if not parts:
            return None
        return pd.to_datetime(row['date'], utc=True), float(np.mean(parts))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SEMANTIC CHAIN DAMP — the chain decision moved INSIDE Gemma
# ---------------------------------------------------------------------------
# The old rule was (same account) AND (<=60 min) AND (signal <= leader + 0.10)
# -> hard 0.0. It never read the text. It cannot tell
#     "25% tariff on China"  ->  "and on Mexico too"     (NEW, market-moving)
# from
#     "Great meeting today"  ->  "Really great meeting"  (pure restatement)
# and zeroes BOTH. Measured on the corpus: 69,496 posts — 36.6% of everything —
# are damped to zero by that text-blind rule.
#
# Semantic novelty is exactly what an encoder is FOR, and Gemma already
# embedded every post. NOTE the distinction from the failed memory experiment:
# using neighbour OUTCOMES to predict a move was dead (+0.001), because
# semantic similarity is not market impact. Using cosine to detect whether two
# posts SAY THE SAME THING is the encoder's core competence — a different
# question with a different answer.
#
# Graded, not binary:
#     cos >= CHAIN_REPEAT_HI  -> 0.0   true restatement, already priced
#     cos <= CHAIN_NOVEL_LO   -> 1.0   new content, trade it normally
#     between                 -> linear ramp
# Falls back to the old signal-delta rule when no embedding is available, so
# nothing breaks on a cold start or a pre-embedding model dir.
CHAIN_REPEAT_HI = float(os.environ.get("CHAIN_REPEAT_HI", "0.93"))
CHAIN_NOVEL_LO  = float(os.environ.get("CHAIN_NOVEL_LO", "0.78"))

# Prefer the thresholds CALIBRATED BY THE BUILD on real chain pairs. Gemma is a
# decoder and its embeddings are anisotropic — cosines bunch into a narrow band
# whose location depends on the encoder, so absolute cutoffs are guesswork.
# Measured on synthetic anisotropic data: fixed 0.93/0.78 damped 0 of 399 real
# pairs, i.e. silently disabled the guard. Training labels and live trading
# MUST use the same numbers or they disagree about what a restatement is.
def _load_chain_thresholds():
    global CHAIN_REPEAT_HI, CHAIN_NOVEL_LO
    if os.environ.get("CHAIN_REPEAT_HI"):
        return
    p = os.path.normpath(os.path.join(_HERE, "..", "DP", "chain_thresholds.json"))
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        hi, lo = float(d["repeat_hi"]), float(d["novel_lo"])
        if hi - lo >= 0.02:
            CHAIN_REPEAT_HI, CHAIN_NOVEL_LO = hi, lo
            print(f"  🎚️  chain thresholds from build calibration: "
                  f"repeat≥{hi:.3f} novel≤{lo:.3f} (n={d.get('n_pairs','?')} pairs)")
    except FileNotFoundError:
        pass
    except Exception as e:                                    # noqa: BLE001
        print(f"  ⚠️  chain_thresholds.json unreadable ({e}) — using defaults")


_load_chain_thresholds()


def _cos(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape or not a.size:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(a @ b / (na * nb))


def chain_factor(account, post_ts, signal, emb=None):
    """Damp factor for a post that lands inside another post's reaction window.

    `emb` = this post's Gemma embedding. When the leader's embedding is also
    known, GEMMA decides whether this is a restatement or new information;
    otherwise we fall back to the old signal-delta heuristic.
    """
    if account is None or post_ts is None:
        return 1.0, ""
    key = str(account).lower()
    prev = _CHAIN_STATE.get(key)
    if prev is None:
        hit = _db_last_post_signal(account, post_ts)
        if hit is not None:
            prev = {"ts": hit[0], "sig": hit[1], "emb": None}
    if prev is not None:
        gap = (post_ts - prev["ts"]).total_seconds()
        if 0 <= gap <= CHAIN_WINDOW_MIN * 60:
            cos = _cos(emb, prev.get("emb"))
            if cos is not None:
                if cos >= CHAIN_REPEAT_HI:
                    f, why = 0.0, "restates the leader"
                elif cos <= CHAIN_NOVEL_LO:
                    f, why = 1.0, "NEW information despite the window"
                else:
                    f = float((CHAIN_REPEAT_HI - cos) /
                              max(CHAIN_REPEAT_HI - CHAIN_NOVEL_LO, 1e-9))
                    why = "partial overlap"
                # a genuinely new post becomes the new chain leader
                _CHAIN_STATE[key] = {
                    "ts": post_ts,
                    "sig": prev["sig"] if f < 0.5 else signal,
                    "emb": prev.get("emb") if f < 0.5 else emb}
                if f >= 0.999:
                    return 1.0, ""
                return f, (f"chain x{f:.2f} ({gap/60:.0f}min after leader, "
                           f"gemma cos={cos:.3f} — {why})")
            # --- fallback: no embedding for the leader -> old rule ---------
            if signal <= prev["sig"] + CHAIN_SIGNAL_DELTA:
                _CHAIN_STATE[key] = {"ts": post_ts, "sig": prev["sig"],
                                     "emb": prev.get("emb")}
                return 0.0, (f"chain-follower ({gap/60:.0f}min after leader, "
                             f"Δsignal {signal - prev['sig']:+.2f} < "
                             f"+{CHAIN_SIGNAL_DELTA}, no embedding — "
                             f"signal-delta fallback)")
    _CHAIN_STATE[key] = {"ts": post_ts, "sig": signal, "emb": emb}
    return 1.0, ""


# (name, emoji, display label) — loaded DYNAMICALLY from DP/instruments.json
# (master registry). Add/remove instruments there, no code edits needed.
_INSTRUMENTS_FILE = os.path.join(_HERE, "..", "DP", "instruments.json")
with open(_INSTRUMENTS_FILE, encoding="utf-8") as _f:
    LABELS = [(k, v.get("emoji", ""), v.get("label", k))
              for k, v in json.load(_f)["instruments"].items()]

_tok = _bert = None

TRADE_ACCURACY = {}

def load(model_dir=None):
    global _tok, _bert, TRADE_ACCURACY
    model_dir = model_dir or OUT_DIR
    cfg = json.load(open(f"{model_dir}/config.json"))

    # EMBEDDING PCA (2026-07-16): models trained with compressed embeddings
    # (config 'emb_pca') need the SAME projection at predict time. Attach the
    # matrices to cfg so predict()/backtest project identically.
    if cfg.get("emb_pca"):
        _z = np.load(os.path.join(model_dir, "emb_pca.npz"))
        cfg["_emb_pca_mats"] = (_z["mean"], _z["components"])

    # gemma sentiment head (canonical 3-class probabilities) — see
    # gemma_sentiment_from_emb(). Saved by train when models use it.
    _shp = os.path.join(model_dir, "gemma_sent_head.npz")
    if os.path.exists(_shp):
        cfg["_sent_head"] = dict(np.load(_shp))

    # CONFIG-DRIFT NOTICE: scorer_config.json evolves daily (LLM-updated), but
    # this model only uses the feature list frozen at ITS train time
    # (cfg['nlp_features']). Newer flags are scored+stored in the DB but
    # ignored here until the next retrain — safe, never a crash. This just
    # makes the drift visible so you know when a retrain is worth it.
    try:
        cur_flags   = set(ss.CONFIG["policy_flags"].keys())
        model_feats = set(cfg["nlp_features"])
        new_flags     = sorted(cur_flags - model_feats)
        dropped_flags = sorted(f for f in model_feats
                               if f.startswith("flag_") and f not in cur_flags)
        if new_flags:
            print(f"  ℹ️  {len(new_flags)} newer flag(s) in scorer_config.json not in this "
                  f"model (stored in DB, ignored until retrain): {', '.join(new_flags[:5])}")
        if dropped_flags:
            print(f"  ℹ️  {len(dropped_flags)} flag(s) removed from config since training "
                  f"(fed as 0): {', '.join(dropped_flags[:5])}")
    except Exception:
        pass

    acc_path = f"{model_dir}/trade_accuracy.json"
    if os.path.exists(acc_path):
        with open(acc_path, encoding='utf-8') as f:
            acc_report = json.load(f)
        TRADE_ACCURACY = acc_report.get('instruments', {})
        win = acc_report.get('window', {})
        print(f"  \U0001f4ca Loaded TRADE/SKIP filtered accuracy from {acc_path} "
              f"(backtest window {win.get('since','?')} -> {win.get('until','?')})")
    else:
        TRADE_ACCURACY = {}

    # Gemma-3-4B loads lazily inside gemma_embedder on first embed call
    models = {}
    for inst,_,_ in LABELS:
        p = f"{model_dir}/{inst}_Impact.json"
        if os.path.exists(p):
            m = xgb.XGBRegressor(); m.load_model(p); models[inst]=m

    # ------------------------------------------------------------------
    # THE HEADS THAT ACTUALLY DECIDE (2026-08-02)
    # ------------------------------------------------------------------
    # Live prediction used to load ONLY the signed regressor above — the one
    # measured at mean|pred| 0.001% vs mean|actual| 0.318%, MdAPE 100%, i.e.
    # predicting zero. Everything that works lives in the three heads:
    #   __size  |move| in %      corr 0.218, MdAPE 47%  -> the number
    #   __dir   P(up)            -> the direction (sign(pred) scored 48.2%)
    #   __move  P(abnormal move) -> the gate
    # Loading them here is what puts backtest and production on the same model.
    clf_move, clf_dir, reg_size = {}, {}, {}
    for inst,_,_ in LABELS:
        for suf, store, ctor in (("__move", clf_move, xgb.XGBClassifier),
                                 ("__dir",  clf_dir,  xgb.XGBClassifier),
                                 ("__size", reg_size, xgb.XGBRegressor)):
            p = f"{model_dir}/{inst}_Impact{suf}.json"
            if os.path.exists(p):
                _m = ctor(); _m.load_model(p); store[inst] = _m
    cfg["_clf_move"], cfg["_clf_dir"], cfg["_reg_size"] = clf_move, clf_dir, reg_size
    if reg_size:
        _tr = [i for i in reg_size
               if cfg.get("move_gate", {}).get(i, {}).get("tradeable")]
        print(f"  \U0001f3af Heads: {len(clf_move)} move + {len(clf_dir)} dir + "
              f"{len(reg_size)} size | TRADEABLE: "
              + (", ".join(_tr) if _tr else "NONE"))
    else:
        print("  ⚠️  No size/move/dir heads in this model dir — falling back "
              "to the signed regressor (retrain to enable the real heads)")
    if model_dir != OUT_DIR:
        print(f"  \U0001f4c2 Models loaded from {model_dir} (non-default --model-dir)")
    nlp   = ss.load_spacy()
    sbert = ss.load_sbert()
    print(f"✅ Gemma-3-4B encoder + NLP scorer + {len(models)} XGBoost models\n")
    return cfg, models, nlp, sbert

# ============================================================================
# POST MEMORY — precedent lookup for the human in the loop
# ============================================================================
# "As a human, I can predict how much impact a post has from previous
# historical movements." This shows exactly that: the most similar past posts
# and what each instrument ACTUALLY did in the hour after them.
#
# Deliberately DECISION SUPPORT, not a feature. Measured 2026-08-02: feeding
# neighbour outcomes to XGBoost adds +0.001 (noise). Showing them to a trader
# is a different thing — it is evidence you can weigh, and it makes the
# model's number auditable instead of oracular.
_MEM = {"emb": None, "meta": None, "loaded": False}


def load_memory(limit=40000):
    """Cached past posts + their realised 1h moves, newest `limit` rows."""
    if _MEM["loaded"]:
        return _MEM
    _MEM["loaded"] = True
    try:
        import db as _db
        cols = ", ".join(f'"{i}_Impact"' for i, _, _ in LABELS)
        meta = _db.query(
            f"SELECT platform, id, date, text, {cols} "
            f"FROM training_set_FINAL WHERE sample_weight >= 0.3 "
            f"ORDER BY date DESC LIMIT {int(limit)}")
        if meta is None or not len(meta):
            return _MEM
        keys = (meta["platform"].astype(str) + "_" + meta["id"].astype(str)).tolist()
        lit = ",".join("'" + k.replace("'", "''") + "'" for k in keys)
        emb = _db.query(f"SELECT platform_id, embedding FROM gemma3_embeddings_v1 "
                        f"WHERE platform_id IN ({lit})")
        if emb is None or not len(emb):
            return _MEM
        pos = {k: i for i, k in enumerate(keys)}
        E = np.zeros((len(keys), len(emb["embedding"].iloc[0])), dtype=np.float32)
        seen = np.zeros(len(keys), dtype=bool)
        for pid, v in zip(emb["platform_id"], emb["embedding"]):
            i = pos.get(pid)
            if i is not None:
                E[i] = np.asarray(v, dtype=np.float32)
                seen[i] = True
        E = E[seen]
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        _MEM["emb"] = E
        _MEM["meta"] = meta[seen].reset_index(drop=True)
        print(f"  \U0001f9e0 Post memory: {len(E)} precedents with realised moves")
    except Exception as e:                                    # noqa: BLE001
        print(f"  ℹ️  post memory unavailable ({type(e).__name__}: {str(e)[:60]})")
    return _MEM


def find_precedents(emb, post_ts=None, k=5):
    """k most similar PAST posts. If post_ts is given, only strictly earlier
    ones are eligible — the same causality rule the trainer uses, so what you
    see here is what a live system could have seen."""
    M = load_memory()
    if M["emb"] is None:
        return []
    q = np.asarray(emb, dtype=np.float32).ravel()
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    sims = M["emb"] @ q
    meta = M["meta"]
    if post_ts is not None:
        try:
            older = pd.to_datetime(meta["date"], utc=True) < pd.Timestamp(post_ts).tz_convert("UTC")
            sims = np.where(older.values, sims, -2.0)
        except Exception:
            pass
    idx = np.argsort(-sims)[:k]
    return [{"sim": float(sims[i]), "row": meta.iloc[i]}
            for i in idx if sims[i] > -1.0]


def show_precedents(precs, insts=None, top_n=6):
    """Print precedents with what each instrument actually did."""
    if not precs:
        return
    insts = insts or [i for i, _, _ in LABELS][:top_n]
    print("\n" + "=" * 64)
    print("POST MEMORY — most similar past posts and what ACTUALLY happened")
    print("=" * 64)
    for p in precs:
        r = p["row"]
        when = str(r["date"])[:16]
        print(f"\n  [{p['sim']:.3f} similar]  {when}")
        print(f"    \"{str(r['text'])[:110]}\"")
        moves = []
        for inst in insts:
            v = r.get(f"{inst}_Impact")
            if v is None or (isinstance(v, float) and v != v):
                continue
            v = float(v)
            moves.append(f"{inst} {'+' if v >= 0 else ''}{v:.2f}%")
        if moves:
            print("    -> " + "  ".join(moves))
    # consensus across precedents = the trader's "last N times, it went this way"
    print("\n  CONSENSUS ACROSS PRECEDENTS")
    for inst in insts:
        vals = [float(p["row"][f"{inst}_Impact"]) for p in precs
                if p["row"].get(f"{inst}_Impact") is not None
                and float(p["row"][f"{inst}_Impact"]) == float(p["row"][f"{inst}_Impact"])]
        if len(vals) < 2:
            continue
        up = sum(1 for v in vals if v > 0)
        print(f"    {inst:<9} median |move| {np.median(np.abs(vals)):.2f}%   "
              f"up {up}/{len(vals)}   "
              f"{'agree UP' if up == len(vals) else ('agree DOWN' if up == 0 else 'split')}")
    print("=" * 64)


def gemma_embed(text):
    """Gemma-3-4B encoding (mean + last-token pooling, 5120 dims).
    Function name kept so every call site matches the canonical pipeline."""
    from gemma_embedder import embed_one
    return embed_one(text)

def parse_stamp(s):
    """yyyymmddhhmm, NY local time -> tz-aware Timestamp."""
    return pd.Timestamp(datetime.datetime.strptime(s.strip(), "%Y%m%d%H%M"), tz=NY)


def predict(text, cfg, models, nlp, sbert, post_ts=None,
            entity_weight: float = 1.0,
            event_weight:  float = 1.0,
            is_primary:    bool  = True,
            account:       str   = None):
    """Predict 1-hour market impacts for a post.

    entity_weight / event_weight / is_primary: supply these for geo-account posts
    so the NLP scorer applies the correct source-context weight (the same values
    used during training). Defaults (1.0, 1.0, True) match rank-0 US President / TruthSocial.
    """
    feats = ss.score_single_post(
        text, nlp=nlp, sbert=sbert,
        feature_cols=cfg['nlp_features'],
        entity_weight=entity_weight,
        event_weight=event_weight,
        is_primary=is_primary,
    )
    emb = gemma_embed(text)
    _sent = gemma_sentiment_from_emb(emb, cfg)
    if _sent is not None:   # inject canonical gemma sentiment features
        feats['gemma_pos'], feats['gemma_neg'], feats['gemma_sent'] = \
            (float(v) for v in _sent[0])
    # GEMMA ANALYST features: when the model set was trained with them,
    # Gemma READS this post live and its 23-impact view joins the features.
    if any(c.startswith('analyst_') for c in cfg['nlp_features']):
        import gemma_analyst as GA
        _a = GA.analyze_texts([text])[0]
        for _i, _c in enumerate(GA.ANALYST_COLS):
            feats[_c] = float(_a[_i])
    nlp_vec = np.array([[float(feats.get(c, 0.0)) for c in cfg['nlp_features']]])
    X = np.hstack([project_emb(emb, cfg), nlp_vec])

    import math
    # DOMAIN-WEIGHTED signal (0.25 policy / 0.45 domain / 0.30 weight) — same
    # formula as build_final_training_set.compute_nlp_signal. A flat mean let
    # structurally-low policy_intensity drag single-domain posts (crypto/
    # COVID/Fed) under the gate regardless of domain strength.
    pis = min(float(feats.get('policy_intensity_score') or 0.0) / 8.0, 1.0)
    dom = min(float(feats.get('hawkish_risk_score') or 0.0) / 5.0, 1.0)
    # macro term is EVENT-WINDOW GATED at the post's date (live = today)
    _pdate = post_ts.date() if post_ts is not None else datetime.date.today()
    dom = max(dom, min(gated_macro_score(feats, _pdate) / 5.0, 1.0))
    sw = float(feats.get('sample_weight') or 0.0)
    signal = float(0.25 * pis + 0.45 * dom + 0.30 * sw)
    gate = 1.0/(1.0+math.exp(-GATE_K*(signal-GATE_MID))) if GATE_ENABLED else 1.0
    mult = 1.0 if gate >= 0.5 else gate

    post_hour = post_ts.hour if post_ts is not None else None
    country = _country_for(account) if account else ("US" if is_primary else "")
    tfactor, tlabel = temporal_factor(text, post_hour=post_hour,
                                      post_ts=post_ts, country=country)
    efactor, elabel = endorsement_factor(text)
    if efactor < tfactor:            # endorsement gate overrides temporal
        tfactor, tlabel = efactor, elabel
    sfactor, slabel = self_news_share_factor(text, account=account)
    if sfactor < tfactor:            # self-news-share gate overrides both
        tfactor, tlabel = sfactor, slabel
    # emb is this post's Gemma vector — hand it to the chain guard so the
    # restatement-vs-new-information call is made by the model, not by a
    # signal-delta proxy.
    cfactor, clabel = chain_factor(account, post_ts, signal, emb=emb)
    if cfactor < tfactor:            # chain guard overrides everything
        tfactor, tlabel = cfactor, clabel
    # reiteration + commentary damps: ONLY for temporally-neutral posts (a
    # bombshell in familiar vocabulary is protected by breaking/future cues)
    _neutral = (tfactor == 1.0 and tlabel == "neutral")
    rfactor, rlabel = reiteration_factor(text, account=account, post_ts=post_ts,
                                         sbert=sbert)
    if _neutral and rfactor < tfactor:
        tfactor, tlabel = rfactor, rlabel
    if _neutral:
        mfactor, mlabel = commentary_factor(text)
        if mfactor < tfactor:
            tfactor, tlabel = mfactor, mlabel
    mult *= tfactor

    out = {}
    cal = cfg.get("calibration", {})   # per-instrument magnitude scale (train holdout fit)
    # ---- per-instrument TA block, exactly as train/backtest build it -------
    # cfg["tech_cols"] / ["global_tech"] freeze the recipe at train time. If
    # live bars are unavailable the block is zero-filled and we say so, rather
    # than silently feeding a different column layout than the model expects.
    _tcols = cfg.get("tech_cols") or []
    _gtech = cfg.get("global_tech") or []
    _ta_src = cfg.get("_live_ta") or {}          # {f"{inst}_{col}": value}

    def _X_for(inst):
        if not _tcols:
            return X
        cols = [f'{inst}_{c}' for c in _tcols] + list(_gtech)
        blk = np.array([[float(_ta_src.get(c, 0.0)) for c in cols]],
                       dtype=np.float32)
        return np.hstack([X, blk])

    _mv, _dr, _sz = (cfg.get("_clf_move", {}), cfg.get("_clf_dir", {}),
                     cfg.get("_reg_size", {}))
    _gate_cfg = cfg.get("move_gate", {})
    for inst,_,_ in LABELS:
        if inst not in models:
            continue
        Xi = _X_for(inst)
        raw = float(models[inst].predict(Xi)[0])
        rec = {"legacy_pred": raw * cal.get(inst, 1.0) * mult}
        g = _gate_cfg.get(inst, {})
        if inst in _sz:
            # SIZE: the honest magnitude, scale-corrected exactly as trained.
            # size_log => the head was fitted on log1p(|move|); revert first.
            sz = float(_sz[inst].predict(Xi)[0])
            if g.get("size_log"):
                sz = float(np.expm1(sz))
            sz = max(sz, 0.0)
            rec["size"] = sz * float(g.get("size_k", 1.0)) * mult
        if inst in _mv:
            rec["p_move"] = float(_mv[inst].predict_proba(Xi)[0, 1])
        if inst in _dr:
            pu = float(_dr[inst].predict_proba(Xi)[0, 1])
            rec["p_up"] = pu
            rec["dir"] = 1 if pu > 0.5 else -1
            rec["edge"] = abs(pu - 0.5)
        # TRADE decision — identical rule to backtest_simulator.py so live and
        # backtest can never disagree: tradeable AND p_move over its gate AND
        # the direction head has an edge AND the post was not damped.
        rec["tradeable"] = bool(g.get("tradeable", False))
        rec["trade"] = bool(
            rec.get("tradeable")
            and rec.get("p_move", 0.0) >= float(g.get("p_move_thr", 1.01))
            and rec.get("edge", 0.0) >= float(g.get("edge_min", 0.05))
            and mult == 1.0)
        # signed expected move: size head for magnitude, dir head for sign
        if "size" in rec and "dir" in rec:
            rec["expected"] = rec["size"] * rec["dir"]
        out[inst] = rec if _sz else rec["legacy_pred"]
    # POST MEMORY: attach the precedents so the caller can show its work.
    # Causal — only posts strictly older than this one are eligible.
    try:
        out["_precedents"] = find_precedents(emb, post_ts=post_ts, k=5)
    except Exception:
        out["_precedents"] = []
    return out, signal, gate, tfactor, tlabel


def show(text, r, signal, gate, tfactor, tlabel,
         account=None, account_name=None, is_primary=True,
         entity_weight=1.0, event_weight=1.0, country=None):
    print("\n" + "-"*64)
    # Source attribution line
    if is_primary:
        handle   = account or _rank0_handle()
        aname    = account_name or handle
        ctry     = country or _country_for(handle)
        ctry_str = f"  [{ctry}]" if ctry else ""
        print(f"@{handle} - {aname}{ctry_str}  (TruthSocial)")
    else:
        handle   = account or "unknown"
        aname    = account_name or handle
        ctry     = country or _country_for(handle)
        ctry_str = f"  [{ctry}]" if ctry else ""
        ew_str   = f"entity_w={entity_weight:.2f}  event_w={event_weight:.2f}"
        print(f"@{handle} - {aname}{ctry_str}  (X/Twitter)   {ew_str}")
    print(f"POST: {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"   NLP signal={signal:.3f}  gate x{gate:.2f}" +
          ("  (real signal - FULL move)" if gate >= 0.5 else "  (noise - damped)"))
    print(f"   temporal={tlabel}  factor x{tfactor:.2f}" +
          ("  (past, no future cue -> damped)" if tfactor < 1.0 else
           "  (future/breaking/neutral -> no extra damping)"))
    total_mult = gate * tfactor if gate < 0.5 else tfactor
    if tfactor < 1.0:
        print(f"   TOTAL x{total_mult:.2f} - STALE / ALREADY PRICED IN - DON'T TRADE")
    elif gate < 0.5:
        print(f"   TOTAL x{total_mult:.2f} - LOW-SIGNAL NOISE - DON'T TRADE")
    else:
        print(f"   TOTAL x{total_mult:.2f} - FULL MOVE - TRADEABLE")
    print("-"*64)
    damped = total_mult != 1.0
    print("PREDICTED 1-HOUR MARKET IMPACT (gemma+NLP->XGBoost):")
    for inst, emoji, name in LABELS:
        _rec = r.get(inst, 0.0)
        # NEW: predict() returns a dict per instrument when the size/dir/move
        # heads exist. Old float form still supported for pre-2026-08 models.
        if isinstance(_rec, dict):
            v = _rec.get("expected", _rec.get("legacy_pred", 0.0))
            decision = "TRADE" if _rec.get("trade") else "SKIP"
            _extra = ""
            if "p_move" in _rec:
                _extra = (f"  p_mv={_rec['p_move']:.2f}"
                          f" p_up={_rec.get('p_up', float('nan')):.2f}"
                          f" size={_rec.get('size', 0.0):.3f}%")
                if not _rec.get("tradeable"):
                    _extra += "  [untradeable: dir head failed OOS]"
        else:
            v = _rec
            decision = "SKIP" if (damped or abs(v) < TRADE_THRESHOLD) else "TRADE"
            _extra = ""
        arrow = "^" if v>0 else ("v" if v<0 else "-")
        bar = "#"*min(int(abs(v)*4),20)
        hist = TRADE_ACCURACY.get(inst)
        hist_str = ""
        if hist:
            if decision == "TRADE":
                if hist.get('trade_meaningful_acc') is not None:
                    hist_str = (f"  (filtered acc {hist['trade_meaningful_acc']*100:.1f}%, "
                                 f"n={hist['trade_meaningful_n']})")
                elif hist.get('trade_acc') is not None:
                    hist_str = f"  (hist TRADE acc {hist['trade_acc']*100:.1f}%, n={hist['trade_n']})"
            elif decision == "SKIP" and hist.get('skip_acc') is not None:
                hist_str = f"  (hist SKIP acc {hist['skip_acc']*100:.1f}%, n={hist['skip_n']})"
        print(f"  {emoji}  {name:<12} {arrow} {v:+.4f}%  {bar}  "
              f"{decision}{_extra}{hist_str}")
    print("-"*64)
    # What a trader would check next: has anything like this happened before,
    # and what did it do? Shown for the instruments the model wants to trade,
    # else the majors.
    _precs = r.get("_precedents") if isinstance(r, dict) else None
    if _precs:
        _want = [i for i, _, _ in LABELS
                 if isinstance(r.get(i), dict) and r[i].get("trade")]
        show_precedents(_precs, insts=(_want or None))
    print()


def main():
    ap = argparse.ArgumentParser(description="Interactive gemma+NLP+XGBoost predictor.")
    ap.add_argument("--time", metavar="yyyymmddhhmm",
                    help="NY local time the post was made; only used to resolve "
                         "ambiguous time-of-day phrases in the temporal gate. Optional.")
    ap.add_argument("--model-dir", default=None,
                    help=f"Directory with <INST>_Impact.json + config.json "
                         f"(default {OUT_DIR}). Point at gemma_nlp_xgb_models_live "
                         f"to use fine-tuned models from backtest_simulator.py --fine-tune.")
    ap.add_argument("--entity-weight", type=float, default=1.0,
                    help="Entity weight for the post account (0-1). "
                         "Default 1.0 (rank-0 primary account). "
                         "For geo accounts: mention_count/28 for leaders, "
                         "or fixed institution weight (e.g. IDF=0.70).")
    ap.add_argument("--event-weight", type=float, default=1.0,
                    help="Event weight for the post account (0-1.3). "
                         "Default 1.0 (no active events). "
                         "Use EventManager.get_account_multiplier(handle) for geo posts.")
    ap.add_argument("--geo-account", dest="is_primary", action="store_false", default=True,
                    help="Pass when predicting for a geo X/Twitter account post. "
                         "Sets is_primary=False so sample_weight applies "
                         "SOURCE_DISCOUNT correctly.")
    ap.add_argument("--account", default=None, metavar="HANDLE",
                    help="Twitter/X handle of the posting account (e.g. ZelenskyyUa). "
                         "Display only -- shown in the source attribution line.")
    ap.add_argument("--account-name", default=None, metavar="NAME",
                    help="Display name of the posting account (e.g. 'Volodymyr Zelenskyy'). "
                         "Display only -- shown in the source attribution line.")
    args = ap.parse_args()
    post_ts = parse_stamp(args.time) if args.time else None

    cfg, models, nlp, sbert = load(args.model_dir)
    print("="*64)
    print("="*64)
    print("  gemma + NLP + XGBoost -- 23 instruments")
    print(f"  NLP gate: {'ON' if GATE_ENABLED else 'OFF'}   Type 'quit' to exit")
    if post_ts is not None:
        print(f"  Post time: {post_ts} (used for ambiguous time-of-day phrases)")
    if args.is_primary:
        handle = args.account or _rank0_handle()
        aname  = args.account_name or handle
        ctry   = _country_for(handle)
        ctry_str = f"  [{ctry}]" if ctry else ""
        print(f"  Source: @{handle} - {aname}{ctry_str}  (TruthSocial / rank-0 primary)")
    else:
        handle = args.account or "unknown"
        aname  = args.account_name or handle
        ctry   = _country_for(handle)
        ctry_str = f"  [{ctry}]" if ctry else ""
        print(f"  Source: @{handle} - {aname}{ctry_str}  (X/Twitter geo)  "
              f"entity_w={args.entity_weight:.2f}  event_w={args.event_weight:.2f}")
    print("="*64+"\n")
    while True:
        try:
            t = input("Enter post: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not t: continue
        if t.lower() in ('quit','exit','q'): break
        r, sig, gate, tfactor, tlabel = predict(
            t, cfg, models, nlp, sbert, post_ts=post_ts,
            entity_weight=args.entity_weight,
            event_weight=args.event_weight,
            is_primary=args.is_primary,
            account=handle,
        )
        show(t, r, sig, gate, tfactor, tlabel,
             account=args.account, account_name=args.account_name,
             is_primary=args.is_primary)


if __name__ == "__main__":
    main()