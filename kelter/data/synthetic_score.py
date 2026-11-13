"""Score-type (ordinal level) tasks. The level is computed by code from the
generated facts. Two families: support ticket priority and access risk.

Also `civil_to_score`, which re-asks a civil_comments record as a 4-level
rating using fixed toxicity-fraction bins.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List

from kelter.data.schema import options_from_criteria

SOURCE = "synthetic_score"

PRIORITY_LEVELS = [
    "Low: a single user is affected and a workaround exists",
    "Medium: a single user is blocked with no workaround, or several users have a workaround",
    "High: several users are blocked with no workaround",
    "Critical: a security exposure or a full outage of the service",
]
RISK_LEVELS = [
    "Minimal: read-only access to public data",
    "Moderate: read-only access to internal data",
    "Elevated: write access to internal data",
    "Severe: any access to restricted data, or admin rights",
]
NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lena"]
SERVICES = ["billing portal", "VPN", "build server", "wiki", "email", "CRM", "data warehouse", "chat"]


def _assemble(segs):
    state, prov, wit = "", [], []
    for text, ptype, decisive in segs:
        s = len(state)
        state += text
        prov.append([s, len(state), ptype])
        if decisive:
            wit.append([s, len(state)])
        state += "\n"
        prov[-1][1] = len(state)
    state = state[:-1]
    prov[-1][1] = len(state)
    wit = [[s, min(e, len(state))] for s, e in wit]
    return state, prov, wit


def priority_level(security: bool, outage: bool, users: int, workaround: bool) -> int:
    if security or outage:
        return 3
    if users > 1 and not workaround:
        return 2
    if users > 1 or not workaround:
        return 1
    return 0


def risk_level(data_class: str, access: str, admin: bool) -> int:
    if admin or data_class == "restricted":
        return 3
    if access == "write":
        return 2
    if data_class == "internal":
        return 1
    return 0


def _rec(i, family, state, prov, wit, instructions, levels, gold):
    crit = list(levels)
    return {
        "id": f"synthscore-{family}-{i:06d}", "source": SOURCE, "split": "train", "state": state,
        "provenance": prov, "question": {"type": "score", "instructions": instructions, "criteria": crit},
        "options": options_from_criteria("score", crit), "gold": str(gold), "null": False,
        "witness": wit or None, "domain": "policy", "meta": {"family": family},
    }


def generate(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name, svc = rng.choice(NAMES), rng.choice(SERVICES)
        if i % 2 == 0:
            security = rng.random() < 0.15
            outage = (not security) and rng.random() < 0.15
            users = rng.choice([1, 1, 2, 5, 12, 40])
            workaround = rng.random() < 0.5
            lvl = priority_level(security, outage, users, workaround)
            facts = [
                (f"Ticket from {name}: problem with the {svc}.", "user", False),
                (f"Affected users: {users}." if not outage else f"Status page: the {svc} is down for everyone.",
                 "tool_output", not security),
                ("Workaround: " + ("restarting the client works." if workaround else "none found."),
                 "tool_output", lvl in (0, 1, 2) and not (security or outage)),
            ]
            if security:
                facts.append((f"Security scan: credentials for the {svc} were exposed in a public log.", "tool_output", True))
            elif rng.random() < 0.3:
                facts.append(("Security scan: no findings.", "tool_output", False))
            head, rest = facts[0], facts[1:]
            rng.shuffle(rest)
            state, prov, wit = _assemble([head] + rest)
            instr = rng.choice([
                "Assign the ticket priority using only the reported facts.",
                "What priority level does this ticket warrant?",
            ])
            out.append(_rec(i, "priority", state, prov, wit, instr, PRIORITY_LEVELS, lvl))
        else:
            dc = rng.choice(["public", "internal", "restricted"])
            access = rng.choice(["read", "write"])
            admin = rng.random() < 0.15
            lvl = risk_level(dc, access, admin)
            segs = [
                # Third element: whether the segment is evidence (a witness). `lvl in (0, 1) and True` is just `lvl in (0, 1)`;
                # the `and True` changes nothing. So this line is evidence for levels 0, 1 and 2 and not for level 3
                # (Severe), which restricted data or admin rights decide whatever the access type; it is not a bug.
                (f"{name} requests {access} access to the {svc}.", "user", lvl == 2 or (lvl in (0, 1) and True)),
                (f"Catalog lookup: the {svc} holds {dc} data.", "tool_output", dc in ("restricted",) or lvl in (0, 1)),
                ("Requested role: " + ("administrator." if admin else "standard user."), "tool_output", admin),
            ]
            state, prov, wit = _assemble(segs)
            instr = "Rate the risk of granting this access request."
            out.append(_rec(i, "risk", state, prov, wit, instr, RISK_LEVELS, lvl))
    return out


CIVIL_LEVELS = [
    "Civil: no rudeness",
    "Mildly rude or dismissive",
    "Clearly rude or insulting",
    "Severely toxic, hateful or threatening",
]


def civil_to_score(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Toxicity fraction t (gold['yes']) to a level by fixed bins:
    [0, 0.2) -> 0, [0.2, 0.5) -> 1, [0.5, 0.8) -> 2, [0.8, 1] -> 3."""
    t = rec["gold"]["yes"]
    lvl = 0 if t < 0.2 else 1 if t < 0.5 else 2 if t < 0.8 else 3
    r = copy.deepcopy(rec)
    r["id"] = rec["id"] + "~score"
    r["question"] = {"type": "score", "instructions": "How toxic is this comment?", "criteria": list(CIVIL_LEVELS)}
    r["options"] = options_from_criteria("score", CIVIL_LEVELS)
    r["gold"] = str(lvl)
    return r
