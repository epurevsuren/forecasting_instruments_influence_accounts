"""
event_manager.py
================
Manages geopolitical/macro events that modulate NLP scoring weight.

Each event has a status and priority that together produce a float weight
(0.0 – 1.3) used by the signal scorer to scale post-level scores when the
post references accounts or assets linked to that event.

USAGE
-----
  from event_manager import EventManager

  em = EventManager()

  # Weight for a single event
  em.get_weight("ukraine_war")           # → 1.0

  # All events affecting a tracked account
  em.get_account_events("ZelenskyyUa")   # → [event dicts …]

  # Highest applicable weight for an account (use as scorer multiplier)
  em.get_account_multiplier("netanyahu") # → 1.0

  # Update an event and persist to JSON
  em.update_event("us_china_tariff_war", status="active",
                   notes="90-day truce expired, tariffs reinstated")

  # Print current event table
  em.summary()

CLI
---
  python event_manager.py                        # summary table
  python event_manager.py --account ZelenskyyUa  # events for one account
  python event_manager.py --update ukraine_war --status paused --notes "Ceasefire signed"
"""

import os
import json
import argparse
from datetime import date, datetime, timezone
from typing import Optional

_HERE        = os.path.dirname(os.path.abspath(__file__))
EVENTS_FILE  = os.path.join(_HERE, "events.json")

# ----------------------------------------------------------------- weights ----

STATUS_WEIGHTS: dict[str, float] = {
    "escalating": 1.3,
    "active":     1.0,
    "paused":     0.4,
    "monitoring": 0.2,
    "ended":      0.1,
    "resolved":   0.05,
}

PRIORITY_MULTIPLIERS: dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
}


# -------------------------------------------------------------- main class ----

class EventManager:
    """
    Load, query, and update events.json.

    Weight formula
    --------------
    effective_weight = STATUS_WEIGHTS[status] * PRIORITY_MULTIPLIERS[priority]

    Example:
      ukraine_war  → active (1.0) * high (1.0)   = 1.00
      tariff_war   → paused (0.4) * high (1.0)   = 0.40
      covid19      → ended  (0.1) * medium (0.6) = 0.06
    """

    def __init__(self, path: str = EVENTS_FILE):
        self._path = path
        self._data = self._load()
        self._domain_cache: dict = {}   # (domain, iso-date) -> gate

    # ---------------------------------------------------------------- I/O ----

    def _load(self) -> dict:
        with open(self._path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def reload(self) -> None:
        """Re-read the JSON from disk (e.g. after manual edits)."""
        self._data = self._load()

    # ---------------------------------------------------------- accessors ----

    @property
    def events(self) -> list[dict]:
        return self._data.get("events", [])

    def get_event(self, event_id: str) -> Optional[dict]:
        """Return the event dict or None."""
        for ev in self.events:
            if ev["id"] == event_id:
                return ev
        return None

    def get_weight(self, event_id: str) -> float:
        """
        Effective scoring weight for one event.
        Returns 0.0 if event not found.
        """
        ev = self.get_event(event_id)
        if ev is None:
            return 0.0
        sw = STATUS_WEIGHTS.get(ev.get("status", ""), 0.0)
        pm = PRIORITY_MULTIPLIERS.get(ev.get("priority", ""), 0.0)
        return round(sw * pm, 4)

    def get_active_events(self, min_weight: float = 0.1) -> list[dict]:
        """
        Return events whose effective weight >= min_weight.
        Sorted by weight descending.
        """
        results = []
        for ev in self.events:
            w = self.get_weight(ev["id"])
            if w >= min_weight:
                results.append({**ev, "_weight": w})
        return sorted(results, key=lambda x: x["_weight"], reverse=True)

    def get_account_events(self, account_handle: str) -> list[dict]:
        """
        Return all events that list account_handle in affected_accounts.
        Handle matching is case-insensitive and strips leading @.
        """
        handle = account_handle.lstrip("@").lower()
        results = []
        for ev in self.events:
            affected = [a.lstrip("@").lower() for a in ev.get("affected_accounts", [])]
            if handle in affected:
                results.append({**ev, "_weight": self.get_weight(ev["id"])})
        return sorted(results, key=lambda x: x["_weight"], reverse=True)

    @staticmethod
    def _parse_date(x):
        if x is None or str(x).strip().upper() in ("", "N/A", "NONE", "NULL"):
            return None
        try:
            return date.fromisoformat(str(x).strip()[:10])
        except (ValueError, TypeError):
            return None

    def get_weight_at(self, event_id: str, at_date) -> float:
        """
        POINT-IN-TIME weight for back-simulation: 0.0 when `at_date` falls
        outside [start_date, end_date]. Inside the window:
          * CLOSED (historical) events count as ACTIVE — their status TODAY
            says 'ended', but at that date they were live (a 2020-03 post
            must feel covid at full weight, not today's 0.1 'ended' weight);
          * OPEN events (end_date N/A) keep their CURRENT status nuance
            (escalating 1.3 / paused 0.4 ...) — that IS the present.
        """
        ev = self.get_event(event_id)
        if ev is None:
            return 0.0
        d = at_date if isinstance(at_date, date) else self._parse_date(at_date)
        if d is None:
            return self.get_weight(event_id)
        start = self._parse_date(ev.get("start_date"))
        end   = self._parse_date(ev.get("end_date"))
        if start is not None and d < start:
            return 0.0
        if end is not None and d > end:
            return 0.0
        pm = PRIORITY_MULTIPLIERS.get(ev.get("priority", ""), 0.0)
        if end is None:      # still open -> live status nuance applies
            sw = STATUS_WEIGHTS.get(ev.get("status", ""), 0.0)
        else:                # closed historical event, in-window -> it was ACTIVE
            sw = STATUS_WEIGHTS["active"]
        return round(sw * pm, 4)

    def get_account_multiplier(self, account_handle: str, at_date=None) -> float:
        """
        Single multiplier for a tracked account = max weight across all
        events that affect it. Returns 1.0 if no events match (neutral).

        at_date (date | 'YYYY-MM-DD' | None): when given, weights are
        POINT-IN-TIME (see get_weight_at) so historical posts carry the
        event landscape of THEIR day, not today's statuses.
        """
        evs = self.get_account_events(account_handle)
        if not evs:
            return 1.0
        if at_date is None:
            return max(ev["_weight"] for ev in evs)
        w = max(self.get_weight_at(ev["id"], at_date) for ev in evs)
        return w if w > 0 else 1.0

    # ------------------------------------------------------------------
    # DOMAIN GATING — event-window awareness for NLP keyword scoring.
    # Maps event id -> scorer keyword domains (flag_<domain> in
    # scorer_config.json). Events added later by the LLM curator may carry
    # their own "domains" list in events.json, which takes precedence.
    # ------------------------------------------------------------------
    DOMAIN_FALLBACK = {
        "crypto_regulation_wave":   ["crypto_policy"],
        "election_2024":            ["crypto_policy"],
        "covid_crash_2020":         ["public_health", "stimulus"],
        "covid_stimulus_recovery":  ["public_health", "stimulus",
                                     "interest_rate", "tax_policy"],
        "covid19_pandemic":         ["public_health"],
        "fed_tightening_2018":      ["interest_rate"],
        "inflation_fed_hikes_2022": ["interest_rate", "energy_policy"],
        "svb_banking_crisis_2023":  ["financial_system"],
        "tax_cuts_boom_2017":       ["tax_policy", "stimulus"],
        "election_2016_trump_rally":["tax_policy", "stimulus"],
        "russia_energy_sanctions":  ["energy_policy"],
        "iran_strait_hormuz":       ["energy_policy"],
        "venezuela_political":      ["energy_policy"],
        "us_china_tech_war":        ["ai_chip_policy"],
        "ai_disruption":            ["ai_chip_policy"],
        "taiwan_strait_tensions":   ["ai_chip_policy"],
    }

    def domain_activity(self, domain: str, at_date) -> float:
        """Gate 0..1: is this keyword DOMAIN 'live' at this date?

        A domain is live while ANY event tagged with it is in-window; the
        gate is the strongest matching event's priority multiplier (1.0
        high / 0.6 medium / 0.3 low). Status is deliberately IGNORED here:
        an open event's status reflects TODAY, not the post's day — window
        membership is the honest point-in-time signal. 0.0 = dormant (a
        2017 'vaccine' post must NOT score as a pandemic post; a 2016
        'bitcoin' mention predates the regulation wave).
        """
        d = at_date
        if callable(getattr(d, "date", None)):   # datetime / pd.Timestamp (tz-safe)
            d = d.date()
        elif not isinstance(d, date):
            d = self._parse_date(d)
        key = (domain, d.isoformat() if d else None)
        cached = self._domain_cache.get(key)
        if cached is not None:
            return cached
        best = 0.0
        for ev in self.events:
            doms = ev.get("domains") or self.DOMAIN_FALLBACK.get(ev["id"], [])
            if domain not in doms:
                continue
            if d is not None:
                start = self._parse_date(ev.get("start_date"))
                end   = self._parse_date(ev.get("end_date"))
                if (start is not None and d < start) or (end is not None and d > end):
                    continue
            best = max(best, PRIORITY_MULTIPLIERS.get(ev.get("priority", ""), 0.0))
        self._domain_cache[key] = best
        return best

    def get_all_multipliers(self) -> dict[str, float]:
        """Return {account_handle: multiplier} for every account in any event."""
        handles: set[str] = set()
        for ev in self.events:
            for h in ev.get("affected_accounts", []):
                handles.add(h.lstrip("@"))
        return {h: self.get_account_multiplier(h) for h in sorted(handles)}

    # --------------------------------------------------------------- mutate ----

    def update_event(self, event_id: str, **kwargs) -> dict:
        """
        Update fields on an event and save to disk.

        Common kwargs: status, priority, end_date, notes
        Also auto-stamps updated_at.

        Example:
            em.update_event("us_china_tariff_war",
                            status="active",
                            notes="Truce expired 2025-08-12, tariffs reinstated")
        """
        ev = self.get_event(event_id)
        if ev is None:
            raise KeyError(f"Event not found: {event_id!r}")

        allowed = {"status", "priority", "end_date", "notes", "description",
                   "affected_accounts", "affected_assets", "tags"}
        for key, val in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Field {key!r} not updatable via update_event()")
            if key == "status" and val not in STATUS_WEIGHTS:
                raise ValueError(f"Unknown status {val!r}. Valid: {list(STATUS_WEIGHTS)}")
            if key == "priority" and val not in PRIORITY_MULTIPLIERS:
                raise ValueError(f"Unknown priority {val!r}. Valid: {list(PRIORITY_MULTIPLIERS)}")
            ev[key] = val

        ev["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._save()
        return ev

    # ------------------------------------------------------------- display ----

    def summary(self, min_weight: float = 0.0) -> None:
        """Print a formatted event table to stdout."""
        evs = [
            {**ev, "_weight": self.get_weight(ev["id"])}
            for ev in self.events
            if self.get_weight(ev["id"]) >= min_weight
        ]
        evs.sort(key=lambda x: x["_weight"], reverse=True)

        print(f"\n{'ID':<28} {'STATUS':<12} {'PRI':<8} {'WEIGHT':<8} {'START':<12} {'END':<12} NAME")
        print("-" * 110)
        for ev in evs:
            print(
                f"  {ev['id']:<26} "
                f"{ev.get('status','?'):<12} "
                f"{ev.get('priority','?'):<8} "
                f"{ev['_weight']:<8.2f} "
                f"{ev.get('start_date','?'):<12} "
                f"{ev.get('end_date','N/A'):<12} "
                f"{ev.get('name','')}"
            )
        print()


# -------------------------------------------------------------------- CLI ----

def _cli():
    parser = argparse.ArgumentParser(
        description="Query or update geopolitical events."
    )
    parser.add_argument("--account", metavar="HANDLE",
                        help="Show events affecting this account")
    parser.add_argument("--event", metavar="ID",
                        help="Show detail for one event ID")
    parser.add_argument("--update", metavar="ID",
                        help="Event ID to update (use with --status / --notes / --end-date)")
    parser.add_argument("--status",   help="New status value")
    parser.add_argument("--priority", help="New priority value (high/medium/low)")
    parser.add_argument("--end-date", dest="end_date", help="New end date (YYYY-MM-DD or N/A)")
    parser.add_argument("--notes",    help="Update notes field")
    parser.add_argument("--multipliers", action="store_true",
                        help="Print current multiplier per account")
    args = parser.parse_args()

    em = EventManager()

    if args.update:
        kwargs = {}
        if args.status:   kwargs["status"]   = args.status
        if args.priority: kwargs["priority"] = args.priority
        if args.end_date: kwargs["end_date"] = args.end_date
        if args.notes:    kwargs["notes"]    = args.notes
        ev = em.update_event(args.update, **kwargs)
        print(f"Updated {ev['id']!r}: weight={em.get_weight(ev['id']):.2f}")

    elif args.account:
        handle = args.account.lstrip("@")
        evs = em.get_account_events(handle)
        mult = em.get_account_multiplier(handle)
        print(f"\nEvents affecting @{handle}  (multiplier={mult:.2f})")
        print("-" * 60)
        for ev in evs:
            print(f"  [{ev['_weight']:.2f}] {ev['id']}  ({ev['status']}, {ev['priority']})")
            print(f"         {ev['name']}")
        if not evs:
            print("  (no matching events)")
        print()

    elif args.event:
        ev = em.get_event(args.event)
        if ev is None:
            print(f"Event not found: {args.event!r}")
        else:
            print(json.dumps({**ev, "_weight": em.get_weight(ev["id"])}, indent=2))

    elif args.multipliers:
        print(f"\n{'ACCOUNT':<26} MULTIPLIER")
        print("-" * 40)
        for handle, mult in em.get_all_multipliers().items():
            print(f"  {handle:<24} {mult:.2f}")
        print()

    else:
        em.summary()


if __name__ == "__main__":
    _cli()
