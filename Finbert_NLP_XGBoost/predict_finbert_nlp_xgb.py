"""
predict_finbert_nlp_xgb.py
--------------------------
Predict: post -> [FinBERT embedding + NLP scorer features] -> XGBoost -> impacts.
Plus an optional NLP gate that damps obvious-noise posts (low NLP signal),
fixing FinBERT's endorsement-hallucination (Rick Allen -> NatGas +5% nonsense).

Requires signal_scorer.py + scorer_config.json in this folder.
Run:  uv run python predict_finbert_nlp_xgb.py
"""
import os, re, json, argparse, datetime
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import signal_scorer as ss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NY = 'America/New_York'

_HERE          = os.path.dirname(os.path.abspath(__file__))
_ENTITIES_FILE = os.path.join(_HERE, "..", "DP", "influence_accounts.json")

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

OUT_DIR = "finbert_nlp_xgb_models"
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
ENDORSEMENT_PATTERNS = [
    r"complete and total endorsement",
    r"has my (?:complete|full|total|strong)[^.!?\n]{0,30}endorsement",
    r"\b(?:congressman|congresswoman|senator|governor|sheriff|judge|mayor)\b"
    r"[^.!?\n]{0,120}\b(?:tremendous|incredible|fantastic|phenomenal|wonderful|"
    r"spectacular|outstanding|great)\b",
    r"\b(?:tremendous|incredible|fantastic|phenomenal|wonderful|outstanding)\b"
    r"[^.!?\n]{0,40}\b(?:champion|advocate|representative|leader|fighter|warrior)\b",
    r"\bis doing (?:a |an |truly )*(?:incredible|fantastic|tremendous|great|"
    r"amazing|outstanding)\b",
    r"\brunning for (?:re-?election|congress|the senate|senate|governor|office)\b",
    r"\b(?:get out and )?vote for\b",
    # NOTE: bare "great honor" / "congratulations to" are TOO GREEDY — they
    # also appear in real market news ("It is my Great Honor to announce our
    # Trade Agreement with Indonesia", "CONGRATULATIONS TO EVERYONE! ...
    # ceasefire"). Only the endorsement-specific forms are gated:
    r"\b(?:great )?honor to endorse\b",
    r"\bhappy birthday\b",
]
ENDORSEMENT_DAMP = 0.0   # hard skip — endorsements are never tradeable


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


def chain_factor(account, post_ts, signal):
    """(0.0, label) when this post is a chain-follower; else (1.0, '').
    Updates the in-process chain state either way."""
    if account is None or post_ts is None:
        return 1.0, ""
    key = str(account).lower()
    prev = _CHAIN_STATE.get(key)
    if prev is None:
        hit = _db_last_post_signal(account, post_ts)
        if hit is not None:
            prev = {"ts": hit[0], "sig": hit[1]}
    if prev is not None:
        gap = (post_ts - prev["ts"]).total_seconds()
        if 0 <= gap <= CHAIN_WINDOW_MIN * 60 and signal <= prev["sig"] + CHAIN_SIGNAL_DELTA:
            # follower: EXTEND the chain, keep the leader's signal on record
            _CHAIN_STATE[key] = {"ts": post_ts, "sig": prev["sig"]}
            return 0.0, (f"chain-follower ({gap/60:.0f}min after leader, "
                         f"Δsignal {signal - prev['sig']:+.2f} < +{CHAIN_SIGNAL_DELTA}) "
                         f"-- same reaction window, move already priced")
    _CHAIN_STATE[key] = {"ts": post_ts, "sig": signal}
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

    print(f"\U0001f4e5 Loading FinBERT on {DEVICE}...")
    _tok = AutoTokenizer.from_pretrained(cfg['finbert'])
    _bert = AutoModelForSequenceClassification.from_pretrained(
        cfg['finbert'], output_hidden_states=True).to(DEVICE).eval()
    models = {}
    for inst,_,_ in LABELS:
        p = f"{model_dir}/{inst}_Impact.json"
        if os.path.exists(p):
            m = xgb.XGBRegressor(); m.load_model(p); models[inst]=m
    if model_dir != OUT_DIR:
        print(f"  \U0001f4c2 Models loaded from {model_dir} (non-default --model-dir)")
    nlp   = ss.load_spacy()
    sbert = ss.load_sbert()
    print(f"✅ FinBERT + NLP scorer + {len(models)} XGBoost models\n")
    return cfg, models, nlp, sbert

def finbert_embed(text):
    """CLS + mean pooling -- must match training v2."""
    enc = _tok([str(text)[:512]], return_tensors="pt", padding=True,
               truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = _bert(**enc)
    last = out.hidden_states[-1]
    cls  = last[:, 0, :]
    mask = enc['attention_mask'].unsqueeze(-1).float()
    mean = (last*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
    return torch.cat([cls, mean], dim=1).cpu().numpy()

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
    nlp_vec = np.array([[float(feats.get(c, 0.0)) for c in cfg['nlp_features']]])
    X = np.hstack([finbert_embed(text), nlp_vec])

    import math
    parts = []
    if feats.get('policy_intensity_score') is not None:
        parts.append(min(float(feats['policy_intensity_score']) / 8.0, 1.0))
    # domain risk = stronger of war (hawkish) and non-war (macro: crypto/
    # COVID/Fed/banking) — hawkish-only gating killed every non-war era post
    dom = min(float(feats.get('hawkish_risk_score') or 0.0) / 5.0, 1.0)
    dom = max(dom, min(float(feats.get('macro_risk_score') or 0.0) / 5.0, 1.0))
    parts.append(dom)
    if feats.get('sample_weight') is not None:
        parts.append(float(feats['sample_weight']))
    signal = float(np.mean(parts)) if parts else float(feats.get('raw_score', 0.5))
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
    cfactor, clabel = chain_factor(account, post_ts, signal)
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
    for inst,_,_ in LABELS:
        if inst in models:
            raw = float(models[inst].predict(X)[0])
            out[inst] = raw * cal.get(inst, 1.0) * mult
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
    print("PREDICTED 1-HOUR MARKET IMPACT (FinBERT+NLP->XGBoost):")
    for inst, emoji, name in LABELS:
        v = r.get(inst, 0.0)
        arrow = "^" if v>0 else ("v" if v<0 else "-")
        bar = "#"*min(int(abs(v)*4),20)
        decision = "SKIP" if (damped or abs(v) < TRADE_THRESHOLD) else "TRADE"
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
        print(f"  {emoji}  {name:<12} {arrow} {v:+.4f}%  {bar}  {decision}{hist_str}")
    print("-"*64+"\n")


def main():
    ap = argparse.ArgumentParser(description="Interactive FinBERT+NLP+XGBoost predictor.")
    ap.add_argument("--time", metavar="yyyymmddhhmm",
                    help="NY local time the post was made; only used to resolve "
                         "ambiguous time-of-day phrases in the temporal gate. Optional.")
    ap.add_argument("--model-dir", default=None,
                    help=f"Directory with <INST>_Impact.json + config.json "
                         f"(default {OUT_DIR}). Point at finbert_nlp_xgb_models_live "
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
    print("  FinBERT + NLP + XGBoost -- 23 instruments")
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