"""
account_aliases.py
==================
Single source of truth for archived-account -> canonical-account normalization,
driven by influence_accounts.json.

Archived office handles (@WhiteHouse45, @WhiteHouse46, @POTUS45, @POTUS46Archive,
@VP45, @VP46archive, @DeptofDefense, ...) are per-administration snapshots of the
SAME underlying office account. For backtesting they must collapse to their
canonical handle, otherwise both the `account` column AND @mentions fragment by
administration and disrupt the continuous series.

The alias map is derived from influence_accounts.json, in priority order:
  1. an entry's explicit  "alias_of": "@Canonical"  field, else
  2. a canonical handle parsed from the entry's role/note text, matching
     "Archived @Canonical" or "renamed to @Canonical".

Keys and values are lowercase handles WITHOUT the leading '@'. Any pipeline
script can import this so the JSON stays the ONE place these mappings live.
"""
import os
import re
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
ENTITIES_FILE = os.path.join(_HERE, "influence_accounts.json")

_CANON_RE   = re.compile(r"(?:Archived|renamed to)\s+@(\w+)", re.IGNORECASE)
_MENTION_RE = re.compile(r"@(\w{1,15})")


def load_alias_map(path: str = ENTITIES_FILE) -> dict:
    """Return {archived_handle_lower: canonical_handle} (no '@')."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError):
        return {}

    sections = (data.get("primary_accounts", [])
                + data.get("entities", [])
                + data.get("institutions", {}).get("entries", [])
                + data.get("archives", {}).get("entries", []))

    amap: dict = {}
    for e in sections:
        handle = (e.get("account") or e.get("twitter_handle") or "").lstrip("@")
        if not handle:
            continue
        canon = e.get("alias_of")
        if not canon:
            blob = f"{e.get('role', '') or ''} {e.get('note', '') or ''}"
            m = _CANON_RE.search(blob)
            if m:
                canon = m.group(1)
        if not canon:
            continue
        canon = str(canon).lstrip("@")
        if canon.lower() != handle.lower():
            amap[handle.lower()] = canon
    return amap


def canonical_account(handle, amap: dict):
    """Map one account handle (with or without '@') to its canonical form."""
    if handle is None:
        return handle
    h = str(handle).lstrip("@")
    return amap.get(h.lower(), h)


def canonical_mentions(text, amap: dict):
    """Rewrite @archived mentions inside `text` to their canonical handle."""
    if not isinstance(text, str) or "@" not in text:
        return text

    def _sub(m):
        c = amap.get(m.group(1).lower())
        return "@" + c if c else m.group(0)

    return _MENTION_RE.sub(_sub, text)
