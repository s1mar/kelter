"""Programmatic policy and rule tasks with code-computed answer keys.

Every item is assembled from segments (policy, request, tool-output facts).
Facts are sampled as raw values (days, amounts, paths, commands); the gold
label is then computed by `decide` from those raw facts with three-valued
logic, never hard-coded per item. Witness spans are the char spans of the
fact phrases that support the verdict, tracked through marker characters so
offsets are computed by `assemble` and never counted by hand.
"""
from __future__ import annotations

import posixpath
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schema import options_from_criteria, validate

SOURCE = "synthetic_policy"
DOMAIN = "policy"

M0 = "\x01"  # witness start marker (stripped by assemble)
M1 = "\x02"  # witness end marker

Node = Any  # str leaf | (op, child, child, ...)


def assemble(segs: List[Tuple[str, str]]):
    """segs: [(provenance_type, text_with_markers)] -> (state, provenance, witness).

    Every segment except the last is terminated with a newline. Text between
    M0 and M1 becomes a witness span.
    """
    parts: List[str] = []
    prov: List[List[Any]] = []
    wit: List[List[int]] = []
    pos = 0
    for i, (ptype, text) in enumerate(segs):
        if i < len(segs) - 1 and not text.endswith("\n"):
            text = text + "\n"
        start = pos
        buf: List[str] = []
        open_at: Optional[int] = None
        for ch in text:
            if ch == M0:
                open_at = pos
            elif ch == M1:
                wit.append([open_at, pos])
                open_at = None
            else:
                buf.append(ch)
                pos += 1
        prov.append([start, pos, ptype])
        parts.append("".join(buf))
    return "".join(parts), prov, wit


def mark(text: str) -> str:
    return M0 + text + M1


# ---------------------------------------------------------------- logic

def tri(node: Node, st: Dict[str, Optional[bool]]):
    """Three-valued (true, false, unknown) evaluation of a formula over facts. st maps each fact name to True,
    False or None (unknown). Returns (value, support_keys): the value and the fact names that justify it, which
    become the witness. "and": False if any child is False (support: the false children's facts), True if all are
    True (support: all of them), else unknown. Any other op is "or": True if some child is True, False if all are
    False, else unknown.

    First-true-child rule: for an "or" that is True, the support is that of the FIRST true child in the formula's
    order only, not of every true child. One sufficient reason is marked as evidence, so a different child order
    would mark different spans for the same answer."""
    if isinstance(node, str):
        v = st[node]
        return v, ([node] if v is not None else [])
    op, *kids = node
    res = [tri(k, st) for k in kids]
    if op == "and":
        falses = [r for r in res if r[0] is False]
        if falses:
            return False, [k for r in falses for k in r[1]]
        if all(r[0] is True for r in res):
            return True, [k for r in res for k in r[1]]
        return None, []
    trues = [r for r in res if r[0] is True]
    if trues:
        return True, list(trues[0][1])  # first true child only (see the docstring)
    if all(r[0] is False for r in res):
        return False, [k for r in res for k in r[1]]
    return None, []


def _dedup(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------- pools

NAMES = ["Alice Moreno", "Ben Okafor", "Chiara Rossi", "Dmitri Volkov", "Elena Park", "Farid Haddad",
         "Grace Liu", "Hana Sato", "Ivan Petrov", "Jamal Brooks", "Keiko Tanaka", "Luis Ortega",
         "Mina Patel", "Noah Fischer", "Olga Novak", "Pedro Alves", "Quinn Murphy", "Rita Shah",
         "Sam Walker", "Tariq Aziz"]
FILLERS = ["customer since 2019", "region: EU", "plan: standard", "language: en", "last login: yesterday",
           "contact preference: email", "tier: silver"]


def _name(rng):
    return rng.choice(NAMES)


# ---------------------------------------------------------------- families

class Fam:
    def __init__(self, name, formula, keys, params, facts, leafvals, policy, request, clauses,
                 title, doc_prob=0.2, destructive_permit=False, default_source="Lookup"):
        self.name = name
        self.formula = formula
        self.keys = keys
        self.params = params
        self.facts = facts
        self.leafvals = leafvals
        self.policy = policy
        self.request = request
        self.clauses = clauses
        self.title = title
        self.doc_prob = doc_prob
        self.destructive_permit = destructive_permit
        self.default_source = default_source


# refund -------------------------------------------------------------
def _refund_params(rng):
    return {"limit": rng.choice([14, 21, 30, 45, 60])}


def _refund_facts(rng, p, st):
    L = p["limit"]
    days = None
    if st["within"] is True:
        days = rng.randint(1, L)
    elif st["within"] is False:
        days = rng.randint(L + 1, L + 45)
    return {**p, "receipt": st["receipt"], "days": days, "defective": st["defective"]}


def _refund_leaves(f):
    return {"receipt": f["receipt"],
            "within": None if f["days"] is None else f["days"] <= f["limit"],
            "defective": f["defective"]}


def _refund_policy(rng, f):
    L = f["limit"]
    a = rng.choice(["A refund requires a receipt on file.",
                    "Refunds are only issued when the customer has a receipt on file.",
                    "No refund can be given without a receipt on record."])
    b = rng.choice([f"The purchase must be {L} days old or less.",
                    f"Items bought more than {L} days ago are not refundable.",
                    f"Refunds are limited to purchases made within the last {L} days."])
    c = rng.choice(["Defective items are exempt from the day limit.",
                    "The day limit does not apply to defective items.",
                    "A defect waives the day limit, but not the receipt requirement."])
    first = [a, b]
    rng.shuffle(first)
    # the exception sentence refers to the day limit, so it stays after both
    return first + [c]


def _refund_request(rng, f):
    n = rng.randint(1000, 9999)
    return rng.choice([f"Hi, this is {_name(rng)}. I would like a refund for order #{n}.",
                       f"Customer {_name(rng)} requests a refund on order #{n}.",
                       f"Please refund order #{n}, says {_name(rng)}."])


def _refund_clauses(rng, f):
    out = {}
    if f["receipt"] is not None:
        out["receipt"] = ("Order lookup", rng.choice(["receipt: on file", "receipt on file"]) if f["receipt"]
                          else rng.choice(["receipt: none", "no receipt on file"]))
    if f["days"] is not None:
        out["within"] = ("Order lookup", rng.choice([f"purchased {f['days']} days ago",
                                                     f"purchase date: {f['days']} days ago"]))
    if f["defective"] is not None:
        out["defective"] = ("Inspection record", "item flagged defective" if f["defective"]
                            else rng.choice(["no defect found", "item works as intended"]))
    return out


# access -------------------------------------------------------------
ROLES = ["engineer", "analyst", "contractor", "intern", "manager", "support agent", "auditor", "sre"]
RESOURCES = ["the production database", "the payroll dashboard", "the customer export bucket",
             "the build signing keys"]


def _access_params(rng):
    return {"allowed": rng.sample(ROLES, 3), "resource": rng.choice(RESOURCES)}


def _access_facts(rng, p, st):
    role = None
    if st["role"] is True:
        role = rng.choice(p["allowed"])
    elif st["role"] is False:
        role = rng.choice([r for r in ROLES if r not in p["allowed"]])
    return {**p, "role": role, "approval": st["approval"], "oncall": st["oncall"]}


def _access_leaves(f):
    return {"role": None if f["role"] is None else f["role"] in f["allowed"],
            "approval": f["approval"], "oncall": f["oncall"]}


def _access_policy(rng, f):
    roles = ", ".join(f["allowed"])
    a = rng.choice([f"Access to {f['resource']} requires one of these roles: {roles}, plus manager approval on the ticket.",
                    f"Only the roles {roles} may access {f['resource']}, and only with manager approval recorded."])
    b = rng.choice(["On-call staff may bypass the role and approval requirements.",
                    "Anyone currently on call is exempt from the role and approval checks."])
    return [a, b]


def _access_request(rng, f):
    return rng.choice([f"{_name(rng)} is requesting access to {f['resource']}.",
                       f"Access request from {_name(rng)} for {f['resource']}.",
                       f"Can {_name(rng)} be granted access to {f['resource']}?"])


def _access_clauses(rng, f):
    out = {}
    if f["role"] is not None:
        out["role"] = ("Directory lookup", f"role: {f['role']}")
    if f["approval"] is not None:
        out["approval"] = ("Ticket system", rng.choice(["manager approval: recorded", "manager approved the ticket"])
                           if f["approval"] else rng.choice(["manager approval: missing", "no manager approval on the ticket"]))
    if f["oncall"] is not None:
        out["oncall"] = ("On-call roster", "requester is on call" if f["oncall"] else "requester is not on call")
    return out


# expense ------------------------------------------------------------
GOOD_CATS = ["meals", "travel", "software", "training", "office supplies"]
BAD_CATS = ["gifts", "alcohol", "gambling", "entertainment"]


def _expense_params(rng):
    return {"limit": rng.choice([50, 100, 250, 500, 1000]), "prohibited": rng.sample(BAD_CATS, 2)}


def _expense_facts(rng, p, st):
    L = p["limit"]
    amount = None
    if st["amount"] is True:
        amount = rng.randint(5, L)
    elif st["amount"] is False:
        amount = rng.randint(L + 1, L + 400)
    cat = None
    if st["category"] is True:
        cat = rng.choice(GOOD_CATS + [c for c in BAD_CATS if c not in p["prohibited"]])
    elif st["category"] is False:
        cat = rng.choice(p["prohibited"])
    return {**p, "amount": amount, "receipt": st["receipt"], "category": cat}


def _expense_leaves(f):
    return {"amount": None if f["amount"] is None else f["amount"] <= f["limit"],
            "receipt": f["receipt"],
            "category": None if f["category"] is None else f["category"] not in f["prohibited"]}


def _expense_policy(rng, f):
    L = f["limit"]
    a = rng.choice([f"Approve an expense only if the amount is at most ${L}.",
                    f"Expenses above ${L} are never approved under this policy."])
    b = rng.choice(["A receipt must be attached.", "Claims without a receipt are rejected."])
    c = f"Prohibited categories: {' and '.join(f['prohibited'])}."
    return [a, b, c]


def _expense_request(rng, f):
    return rng.choice([f"{_name(rng)} submitted an expense claim for approval.",
                       f"Please review the expense claim filed by {_name(rng)}.",
                       f"Is {_name(rng)}'s expense claim approved?"])


def _expense_clauses(rng, f):
    out = {}
    if f["amount"] is not None:
        out["amount"] = ("Expense report", f"amount: ${f['amount']}")
    if f["category"] is not None:
        out["category"] = ("Expense report", f"category: {f['category']}")
    if f["receipt"] is not None:
        out["receipt"] = ("Receipt scanner", "receipt found" if f["receipt"] else "no receipt found")
    return out


# shell --------------------------------------------------------------
DESTRUCTIVE = ["rm", "truncate", "dd", "mkfs", "shred"]
BENIGN_CMDS = ["ls", "cat", "grep", "head", "wc", "stat"]
WORKSPACES = ["/srv/work/projA", "/home/agent/ws", "/workspace/app", "/opt/sandbox/job7"]


def is_destructive(cmd: str) -> bool:
    return cmd.split()[0] in DESTRUCTIVE


def path_in_workspace(path: str, ws: str) -> bool:
    n = posixpath.normpath(path)
    return n == ws or n.startswith(ws.rstrip("/") + "/")


def _shell_params(rng):
    return {"workspace": rng.choice(WORKSPACES)}


def _shell_facts(rng, p, st):
    ws = p["workspace"]
    cmd = None
    if st["benign"] is True:
        cmd = rng.choice(BENIGN_CMDS) + rng.choice(["", " -l", " -n 20"])
    elif st["benign"] is False:
        cmd = rng.choice(DESTRUCTIVE) + rng.choice(["", " -rf", " -f"])
    path = None
    if st["inpath"] is True:
        path = ws + "/" + rng.choice(["src/main.py", "build/", "tmp/cache", "data/out.csv"])
    elif st["inpath"] is False:
        path = rng.choice(["/etc/hosts", "/home/other/notes.txt", ws + "-old/data", ws + "/../secrets/keys"])
    return {**p, "command": cmd, "path": path, "confirm": st["confirm"]}


def _shell_leaves(f):
    return {"benign": None if f["command"] is None else not is_destructive(f["command"]),
            "inpath": None if f["path"] is None else path_in_workspace(f["path"], f["workspace"]),
            "confirm": f["confirm"]}


def _shell_policy(rng, f):
    a = rng.choice([f"Destructive commands ({', '.join(DESTRUCTIVE)}) need explicit user confirmation before they run.",
                    f"Commands that delete or overwrite data ({', '.join(DESTRUCTIVE)}) may run only after the user explicitly confirms."])
    b = rng.choice([f"Every command may only touch paths inside {f['workspace']}.",
                    f"The allowed workspace is {f['workspace']}; commands must not reach outside it."])
    out = [a, b]
    rng.shuffle(out)
    return out


def _shell_request(rng, f):
    return rng.choice(["Please run the cleanup step for the project.",
                       "The agent proposes the tool call below. Gate it.",
                       "Execute the pending command so the job can continue."])


def _shell_clauses(rng, f):
    out = {}
    if f["command"] is not None:
        out["benign"] = ("Pending tool call", f"command: {f['command']}")
    if f["path"] is not None:
        out["inpath"] = ("Pending tool call", f"target path: {f['path']}")
    if f["confirm"] is not None:
        out["confirm"] = ("Session log", "user explicitly confirmed this command" if f["confirm"]
                          else "no explicit user confirmation recorded")
    return out


# sharing ------------------------------------------------------------
RESTRICTED = ["payment card numbers", "health records", "access credentials"]
OPEN_CLASSES = ["public marketing material", "aggregate usage stats", "internal wiki pages", "product roadmap"]
EXT_DOMAINS = ["partner.example", "vendor.example", "mail.example"]


def _sharing_params(rng):
    return {"domain": rng.choice(["acme.example", "northwind.example", "initech.example"]),
            "restricted": rng.sample(RESTRICTED, 2)}


def _sharing_facts(rng, p, st):
    user = rng.choice(["ops", "dana", "legal", "kim", "audit"])
    rcpt = None
    if st["internal"] is True:
        rcpt = f"{user}@{p['domain']}"
    elif st["internal"] is False:
        rcpt = rng.choice([f"{user}@{rng.choice(EXT_DOMAINS)}", f"{user}@{p['domain']}.net",
                           f"{user}@mail.{p['domain']}"])
    cls = None
    if st["unrestricted"] is True:
        cls = rng.choice(OPEN_CLASSES + [r for r in RESTRICTED if r not in p["restricted"]])
    elif st["unrestricted"] is False:
        cls = rng.choice(p["restricted"])
    return {**p, "recipient": rcpt, "dataclass": cls, "dpa": st["dpa"]}


def _sharing_leaves(f):
    return {"internal": None if f["recipient"] is None else f["recipient"].endswith("@" + f["domain"]),
            "unrestricted": None if f["dataclass"] is None else f["dataclass"] not in f["restricted"],
            "dpa": f["dpa"]}


def _sharing_policy(rng, f):
    a = f"Restricted data classes: {' and '.join(f['restricted'])}."
    b = rng.choice([f"Recipients at @{f['domain']} may receive any unrestricted data.",
                    f"Internal recipients (addresses ending in @{f['domain']}) can get data that is not restricted."])
    c = rng.choice(["A signed DPA with the recipient overrides these limits.",
                    "If a signed DPA is on file for the recipient, the sharing is allowed regardless."])
    return [a, b, c]


def _sharing_request(rng, f):
    return rng.choice([f"{_name(rng)} wants to send a dataset to a recipient.",
                       f"Data sharing request from {_name(rng)}.",
                       f"{_name(rng)} asks to share an export. Is it allowed?"])


def _sharing_clauses(rng, f):
    out = {}
    if f["recipient"] is not None:
        out["internal"] = ("Recipient directory", f"recipient: {f['recipient']}")
    if f["dataclass"] is not None:
        out["unrestricted"] = ("Data catalog", f"data class: {f['dataclass']}")
    if f["dpa"] is not None:
        out["dpa"] = ("Contract registry", "signed DPA on file" if f["dpa"] else "no signed DPA on file")
    return out


# deletion -----------------------------------------------------------
def _deletion_params(rng):
    return {}


def _deletion_facts(rng, p, st):
    bal = None
    if st["balance"] is True:
        bal = 0
    elif st["balance"] is False:
        bal = rng.randint(5, 400)
    return {"identity": st["identity"], "balance": bal}


def _deletion_leaves(f):
    return {"identity": f["identity"], "balance": None if f["balance"] is None else f["balance"] == 0}


def _deletion_policy(rng, f):
    return [rng.choice(["An account may be deleted only after the holder's identity is verified.",
                        "Deletion requires a verified identity."]),
            rng.choice(["Accounts with an open balance cannot be deleted.",
                        "Any open balance must be zero before deletion."])]


def _deletion_request(rng, f):
    return rng.choice([f"{_name(rng)} asks to permanently delete their account.",
                       f"Please close and erase the account of {_name(rng)}.",
                       f"Delete my account, signed {_name(rng)}."])


def _deletion_clauses(rng, f):
    out = {}
    if f["identity"] is not None:
        out["identity"] = ("Identity service", "identity verified" if f["identity"] else "identity not verified")
    if f["balance"] is not None:
        out["balance"] = ("Billing system", f"open balance: ${f['balance']}")
    return out


FAMILIES: Dict[str, Fam] = {}


def _reg(fam: Fam):
    FAMILIES[fam.name] = fam


_reg(Fam("refund", ("and", "receipt", ("or", "within", "defective")), ["receipt", "within", "defective"],
         _refund_params, _refund_facts, _refund_leaves, _refund_policy, _refund_request, _refund_clauses,
         "Refund policy", 0.2, False, "Order lookup"))
_reg(Fam("access", ("or", "oncall", ("and", "role", "approval")), ["role", "approval", "oncall"],
         _access_params, _access_facts, _access_leaves, _access_policy, _access_request, _access_clauses,
         "Access policy", 0.2, False, "Directory lookup"))
_reg(Fam("expense", ("and", "amount", "receipt", "category"), ["amount", "receipt", "category"],
         _expense_params, _expense_facts, _expense_leaves, _expense_policy, _expense_request, _expense_clauses,
         "Expense policy", 0.5, False, "Expense report"))
_reg(Fam("shell", ("and", "inpath", ("or", "benign", "confirm")), ["benign", "inpath", "confirm"],
         _shell_params, _shell_facts, _shell_leaves, _shell_policy, _shell_request, _shell_clauses,
         "Command gating policy", 0.0, True, "Pending tool call"))
_reg(Fam("sharing", ("or", "dpa", ("and", "internal", "unrestricted")), ["internal", "unrestricted", "dpa"],
         _sharing_params, _sharing_facts, _sharing_leaves, _sharing_policy, _sharing_request, _sharing_clauses,
         "Data sharing policy", 0.5, False, "Recipient directory"))
_reg(Fam("deletion", ("and", "identity", "balance"), ["identity", "balance"],
         _deletion_params, _deletion_facts, _deletion_leaves, _deletion_policy, _deletion_request,
         _deletion_clauses, "Account deletion policy", 0.2, True, "Identity service"))
FAMILY_ORDER = list(FAMILIES)


def decide(family: str, facts: Dict[str, Any]) -> str:
    """Answer key from raw facts: 'allow', 'deny' or 'escalate' (a needed fact is None)."""
    fam = FAMILIES[family]
    v, _ = tri(fam.formula, fam.leafvals(facts))
    return "allow" if v is True else "deny" if v is False else "escalate"


def _decide_full(fam: Fam, facts):
    st = fam.leafvals(facts)
    v, support = tri(fam.formula, st)
    return v, _dedup(support), [k for k in fam.keys if st[k] is None]


# ---------------------------------------------------------------- items

NOUL_INSTR = ["Under the policy, is the requested action permitted?",
              "Does the policy permit the request described above?",
              "Based on the policy and the lookup results, is the request allowed?"]
NOUL_TRUE = ["The requested action is permitted under the policy.", "The policy allows the request."]
NOUL_FALSE = ["The requested action is not permitted under the policy.", "The policy does not allow the request."]
CHOICE_INSTR = ["Decide how the request should be handled under the policy. Choose escalate only if a required fact is missing.",
                "Pick the outcome the policy requires for this request. Use escalate when information needed to decide is absent.",
                "Given the policy and the lookup results, choose allow, deny or escalate (escalate means a needed fact is missing)."]
CHOICE_DESC = {
    "allow": ["The policy permits the request and the facts show every requirement is met.",
              "Approve the request."],
    "deny": ["A policy requirement is not met by the facts.",
             "Reject the request because a condition fails."],
    "escalate": ["A fact needed to decide is missing, so a human must review.",
                 "Hand the request to a reviewer because required information is absent."],
}


def _gen_one(fam: Fam, rng: random.Random, form: str, idx: int, seed: int) -> Dict[str, Any]:
    params = fam.params(rng)
    if form == "noul":
        target = rng.choice(["allow", "deny"])
    else:
        target = rng.choices(["allow", "deny", "escalate"], [0.35, 0.35, 0.30])[0]
    facts = verdict = support = missing = None
    for _ in range(2000):
        if form == "noul":
            st = {k: rng.random() < 0.5 for k in fam.keys}
        else:
            st = {k: rng.choices([True, False, None], [0.5, 0.3, 0.2])[0] for k in fam.keys}
        facts = fam.facts(rng, params, st)
        v, support, missing = _decide_full(fam, facts)
        verdict = "allow" if v is True else "deny" if v is False else "escalate"
        if verdict == target:
            break
    else:
        raise RuntimeError(f"could not sample {fam.name} item with outcome {target}")

    clauses = fam.clauses(rng, facts)
    decided = verdict != "escalate"
    by_source: Dict[str, List[str]] = {}
    items = list(clauses.items())
    rng.shuffle(items)
    for key, (src, text) in items:
        by_source.setdefault(src, []).append(mark(text) if decided and key in support else text)
    if rng.random() < 0.5:
        filler = rng.choice(FILLERS)
        src = rng.choice(list(by_source) + ["Account note"])
        by_source.setdefault(src, []).append(filler)
        rng.shuffle(by_source[src])
    if not by_source:
        by_source[fam.default_source] = ["no matching records"]
    srcs = list(by_source)
    rng.shuffle(srcs)
    fact_segs = []
    for src in srcs:
        head = rng.choice([f"{src}: ", f"[{src}] ", f"{src} returned: "])
        fact_segs.append(("tool_output", head + "; ".join(by_source[src]) + "."))

    policy_text = f"{fam.title}: " + " ".join(fam.policy(rng, facts))
    ptype = "retrieved_doc" if rng.random() < fam.doc_prob else "system"
    pol = (ptype, policy_text)
    req = ("user", fam.request(rng, facts))
    layout = rng.choice([0, 1, 2])
    if layout == 0:
        segs = [pol, req] + fact_segs
    elif layout == 1:
        segs = [pol] + fact_segs + [req]
    else:
        segs = [req, pol] + fact_segs
    state, prov, wit = assemble(segs)

    permit_class = "destructive" if fam.destructive_permit else "benign"
    if form == "noul":
        criteria = {"true": rng.choice(NOUL_TRUE), "false": rng.choice(NOUL_FALSE)}
        gold = "yes" if verdict == "allow" else "no"
        vclass = {"yes": permit_class, "no": "benign"}
        instr = rng.choice(NOUL_INSTR)
    else:
        names = ["allow", "deny", "escalate"]
        rng.shuffle(names)
        criteria = {n: rng.choice(CHOICE_DESC[n]) for n in names}
        gold = verdict
        vclass = {"allow": permit_class, "deny": "benign", "escalate": "benign"}
        instr = rng.choice(CHOICE_INSTR)
    qtype = form
    rec = {
        "id": f"synth-{fam.name}-{seed}-{idx:06d}",
        "source": SOURCE,
        "split": "train",
        "state": state,
        "provenance": prov,
        "question": {"type": qtype, "instructions": instr, "criteria": criteria},
        "options": options_from_criteria(qtype, criteria),
        "gold": gold,
        "null": False,
        "witness": (wit or None) if decided else None,
        "domain": DOMAIN,
        "meta": {"family": fam.name, "form": form, "outcome": verdict,
                 "decisive_conditions": support if decided else missing,
                 "verdict_class": vclass},
    }
    return rec


def generate(n: int, seed: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        fam = FAMILIES[FAMILY_ORDER[i % len(FAMILY_ORDER)]]
        rng = random.Random(f"{seed}:{i}")
        form = "noul" if rng.random() < 0.5 else "choice"
        rec = _gen_one(fam, rng, form, i, seed)
        validate(rec)
        out.append(rec)
    return out
