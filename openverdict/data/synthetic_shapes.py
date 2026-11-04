"""Programmatic task shapes with code-computed answer keys.

Six families: routing, intent_negation, policy_exemption, adequacy,
long_policy, conflict. Facts are sampled as raw values, the gold label is
computed from them by code, and witness spans come from marker bookkeeping
(see `synthetic.assemble`). Witness is None when gold is a catch-all or
cannot-determine answer with nothing to point at.
"""
from __future__ import annotations

import datetime as _dt
import json
import random
import re
from typing import Any, Dict, List, Optional

from .schema import options_from_criteria, validate
from .synthetic import M0, M1, assemble, mark

SOURCE = "synthetic_shapes"
DOMAIN = "policy"
SPLIT = "train"
FAMILIES = ("routing", "intent_negation", "policy_exemption", "adequacy", "long_policy", "conflict")

PEOPLE = ["Odalys Brennan", "Tomas Kerwin", "Priya Anand", "Lucan Vessey", "Marit Solberg", "Ikenna Dube",
          "Sorcha Whelan", "Yusuf Demir", "Calla Ferreira", "Anders Lindqvist", "Noor Haddad", "Wren Castellan",
          "Bastien Roche", "Ines Varga", "Kofi Mensah", "Tamsin Hallow"]
ORGS = ["Brannock Labs", "Quillon Logistics", "Tessaly Outfitters", "Ostrava Pine Cooperative",
        "Marlowe & Finch Freight", "Veridian Tidewater Group", "Hollin Dock Services", "Corvane Instruments"]


def _rec(family: str, seed: int, j: int, segs, qtype: str, instructions: str, criteria, gold,
         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state, prov, wit = assemble(segs)
    options = options_from_criteria(qtype, criteria)
    names = [o["name"] for o in options]
    rec = {
        "id": f"{SOURCE}-{family}-{seed}-{j:06d}",
        "source": SOURCE,
        "split": SPLIT,
        "state": state,
        "provenance": prov,
        "question": {"type": qtype, "instructions": instructions, "criteria": criteria},
        "options": options,
        "gold": gold,
        "null": False,
        "witness": [list(w) for w in wit] or None,
        "domain": DOMAIN,
        "meta": {"family": family, "verdict_class": {n: "benign" for n in names}, **(extra or {})},
    }
    validate(rec)
    return rec


def _strip(s: str) -> str:
    return s.replace(M0, "").replace(M1, "")


def _fill(tpl: str, slots: Dict[str, List[str]], rng: random.Random) -> str:
    for k, vals in slots.items():
        if "{" + k + "}" in tpl:
            tpl = tpl.replace("{" + k + "}", rng.choice(vals))
    return tpl.replace("[[", M0).replace("]]", M1)


_NUMS = [str(i) for i in range(3, 60)]

# =========================================================== routing

HANDLERS: Dict[str, Dict[str, Any]] = {
    "calc_engine": {
        "group": "math",
        "desc": "Solves a fully stated numeric or algebra problem with no outside material.",
        "slots": {"a": _NUMS, "b": _NUMS, "c": _NUMS},
        "tpl": ["Work out [[the value of x if {a}x + {b} = {c}0]] for me, nothing else attached.",
                "[[Solve this word problem using only the numbers given]]: a tank fills at {a} liters a minute and drains {b} a minute, how long to hold {c}0 liters?",
                "Can you [[compute the derivative of x^{a} + {b}x]] and simplify it?"]},
    "snippet_smith": {
        "group": "code",
        "desc": "Writes a new self-contained function or script from a plain description.",
        "slots": {"task": ["reverses the words in a sentence", "checks whether a string is a palindrome",
                           "merges two sorted lists", "counts the vowels in a string",
                           "converts Celsius to Fahrenheit", "removes duplicates from a list while keeping order"]},
        "tpl": ["[[Write me a standalone Python function]] that {task}. No existing project is involved.",
                "I need [[a short self-contained script from scratch]] that {task}.",
                "[[Draft a small function]] that {task}; it will not live in any repo."]},
    "repo_patcher": {
        "group": "code",
        "desc": "Edits files inside an existing code repository and runs its tests.",
        "slots": {"mod": ["billing", "session", "export", "scheduler", "auth", "notifier"]},
        "tpl": ["In our checkout, [[change the {mod} module and then run the test suite]] to confirm nothing broke.",
                "[[Edit the files under src/{mod}/ in the repo and run the tests]] until they pass.",
                "Please [[patch the existing {mod} code in the repository, then execute its unit tests]]."]},
    "doc_reader": {
        "group": "doc",
        "desc": "Answers questions using the contents of a document the user attached.",
        "slots": {"topic": ["the refund window", "who owns the escalation process", "the renewal date",
                            "the data retention period"]},
        "tpl": ["Using [[the PDF I attached]], what does it say about {topic}?",
                "[[According to the document I uploaded]], what is stated about {topic}?",
                "Check [[the attached report only]] and tell me what it says on {topic}."]},
    "app_driver": {
        "group": "app",
        "desc": "Carries out actions inside an external application or online service.",
        "slots": {"app": ["ticket portal", "booking dashboard", "CRM", "HR leave tool"],
                  "act": ["close ticket 4412", "move Friday's demo to Monday", "mark the Hartley lead as won",
                          "submit my leave for next week"]},
        "tpl": ["[[Open the {app} and {act}]] for me.",
                "[[Log into our {app} account and {act}]].",
                "Please [[operate the {app} on my behalf to {act}]]."]},
    "linguist": {
        "group": "lang",
        "desc": "Converts text from one human language into another.",
        "slots": {"lang": ["French", "Portuguese", "Korean", "Swahili", "Polish"],
                  "text": ["The meeting starts at nine tomorrow.", "Please keep the receipt for your records.",
                           "We are sorry for the delay."]},
        "tpl": ["[[Put this paragraph into {lang}]]: \"{text}\"",
                "How would [[this sentence read in {lang}]]: \"{text}\"",
                "Please [[convert the following text from English into {lang}]]: \"{text}\""]},
    "contract_check": {
        "group": "legal",
        "desc": "Assesses agreements and clauses for legal exposure.",
        "slots": {"clause": ["The supplier may change prices at any time without notice.",
                             "Liability is unlimited for the customer but capped at $100 for the vendor.",
                             "Either party may terminate with no notice and keep all fees paid."]},
        "tpl": ["[[Go through this agreement and flag legal risks]]: {clause}",
                "Does [[the wording of this clause expose us to liability]]? {clause}",
                "[[Check this term for legal problems]] before we sign: {clause}"]},
    "condenser": {
        "group": "summ",
        "desc": "Shortens pasted text into a brief digest.",
        "slots": {"src": ["meeting transcript", "email chain", "forum discussion"]},
        "tpl": ["[[Condense the pasted {src} into a few bullet points]].",
                "Here is a long {src}; [[boil it down to three sentences]].",
                "Give me [[a short digest of the {src} below]]."]},
    "table_cruncher": {
        "group": "data",
        "desc": "Computes statistics and groupings over spreadsheet or CSV data.",
        "slots": {"col": ["revenue", "wait_time", "age", "units_sold"]},
        "tpl": ["From [[the uploaded spreadsheet, work out the mean and spread of {col}]].",
                "[[Crunch the CSV I attached: group by region and total {col}]].",
                "[[Using the data table I gave you, compute the median {col}]]."]},
    "web_scout": {
        "group": "web",
        "desc": "Retrieves current facts from the live internet.",
        "slots": {"topic": ["the new ferry timetable", "the exchange rate for the Norwegian krone today",
                            "the latest release notes of the Ember browser"]},
        "tpl": ["[[Look up today's latest news about]] {topic}.",
                "I need [[current information from the web on]] {topic}.",
                "[[Search online for the freshest details on]] {topic}."]},
    "image_maker": {
        "group": "img",
        "desc": "Produces new pictures or artwork from a text prompt.",
        "slots": {"scene": ["a lighthouse at dusk", "a fox reading a map", "a market street in the rain"]},
        "tpl": ["[[Create an illustration of]] {scene}.",
                "[[Generate a picture]] of {scene}.",
                "[[Draw me a logo concept]] showing {scene}."]},
    "query_writer": {
        "group": "sql",
        "desc": "Writes database queries for a described schema.",
        "slots": {"q": ["the customers who ordered twice in March", "the ten largest invoices"],
                  "t": ["orders", "invoices", "shipments"]},
        "tpl": ["[[Write a SQL query]] that returns {q} from the {t} table.",
                "[[Give me the SELECT statement]] for {q} over our {t} table.",
                "I need [[a database query]] listing {q}; the table is {t}."]},
    "copy_fixer": {
        "group": "proof",
        "desc": "Corrects spelling, grammar and punctuation in supplied text.",
        "slots": {"text": ["Their going to the market tomorow and we was late.",
                           "Please send me the the report by fridy.", "Its a long way, but alot of us goes."]},
        "tpl": ["[[Fix the spelling and grammar in]] this line: \"{text}\"",
                "[[Proofread my note for typos]]: \"{text}\"",
                "[[Clean up the mistakes in this sentence]]: \"{text}\""]},
    "voice_reader": {
        "group": "tts",
        "desc": "Turns written text into spoken audio.",
        "slots": {"text": ["Doors open at seven, please have your tickets ready.",
                           "The warehouse will close early on Friday."]},
        "tpl": ["[[Turn this announcement into spoken audio]]: \"{text}\"",
                "[[Read this aloud and give me an audio file]]: \"{text}\"",
                "I want [[a voice recording of this line]]: \"{text}\""]},
}
_PARTNER = {"snippet_smith": "repo_patcher", "repo_patcher": "snippet_smith"}
_CATCH_GENERIC = [
    ("Any advice on staying motivated at work?", False),
    ("What would be a good name for a golden retriever puppy?", False),
    ("Tell me a joke about penguins.", False),
    ("How many players are on a soccer team on the field?", True),
    ("Roughly how tall is Mount Kilimanjaro in meters?", True),
    ("How many strings does a standard violin have?", True),
    ("What is your opinion on pineapple on pizza?", False),
    ("Why do cats purr, in general terms?", False),
    ("How many legs does a spider have?", True),
    ("Give me three ideas for a weekend with friends.", False),
]
_CATCH_NAMES = ["general", "other"]
_CATCH_DESC = "None of the specialist categories fits."


def _gen_routing(rng, seed, j):
    names = list(HANDLERS)
    ids = rng.sample(names, rng.randint(5, 8))
    catch = rng.random() < 0.15
    target = None
    wit_req = None
    if catch:
        groups = {HANDLERS[i]["group"] for i in ids}
        absent = [n for n in names if n not in ids and HANDLERS[n]["group"] not in groups]
        if absent and rng.random() < 0.5:
            a = rng.choice(absent)
            h = HANDLERS[a]
            req = _strip(_fill(rng.choice(h["tpl"]), h["slots"], rng))
        else:
            req, numeric = rng.choice(_CATCH_GENERIC)
            if numeric and "calc_engine" not in ids and rng.random() < 0.5:
                ids[rng.randrange(len(ids))] = "calc_engine"
        wit_req = req
    else:
        target = rng.choice(ids)
        p = _PARTNER.get(target)
        if p and p not in ids and rng.random() < 0.7:
            ids[rng.choice([i for i in range(len(ids)) if ids[i] != target])] = p
        h = HANDLERS[target]
        wit_req = _fill(rng.choice(h["tpl"]), h["slots"], rng)
    rng.shuffle(ids)
    cname = rng.choice(_CATCH_NAMES)
    order = list(ids)
    order.insert(rng.randint(0, len(order)), cname)
    criteria = {n: (_CATCH_DESC if n == cname else HANDLERS[n]["desc"]) for n in order}
    segs = []
    if rng.random() < 0.4:
        cfg = {"handlers": dict(criteria)}
        segs.append(("system", "Router configuration:\n" + json.dumps(cfg, indent=2)))
        listed = True
    else:
        listed = False
    segs.append(("user", "User request: " + wit_req))
    gold = cname if catch else target
    return _rec("routing", seed, j, segs, "choice",
                "Choose the handler that should take the user's request.", criteria, gold,
                {"handlers_in_state": listed, "catch_all": cname, "is_catch_all": catch})


# =========================================================== intent_negation

INTENTS: Dict[str, Dict[str, str]] = {
    "cancel_order": {"imp": "cancel my order", "past": "I already cancelled my order",
                     "noun": "your cancellation fee policy", "desc": "The customer wants an order cancelled."},
    "refund_request": {"imp": "refund my last purchase", "past": "I already received a refund for my last purchase",
                       "noun": "your refund policy", "desc": "The customer wants money returned for a purchase."},
    "change_address": {"imp": "change my delivery address", "past": "I already changed my delivery address",
                       "noun": "your address verification step", "desc": "The customer wants a delivery address changed."},
    "reset_password": {"imp": "reset my password", "past": "I already reset my password",
                       "noun": "your password rules", "desc": "The customer wants a password reset."},
    "track_package": {"imp": "tell me where my parcel is", "past": "I already found out where my parcel is",
                      "noun": "your tracking page", "desc": "The customer wants the location of a parcel."},
    "update_card": {"imp": "update the card on my account", "past": "I already updated the card on my account",
                    "noun": "your card storage practices", "desc": "The customer wants the stored payment card changed."},
    "upgrade_plan": {"imp": "upgrade me to the Premium plan", "past": "I already upgraded to the Premium plan",
                     "noun": "your plan pricing", "desc": "The customer wants a higher service plan."},
    "close_account": {"imp": "close my account", "past": "I already closed my old account",
                      "noun": "your account closure rules", "desc": "The customer wants an account closed."},
    "report_fraud": {"imp": "report a fraudulent charge", "past": "I already reported the fraudulent charge",
                     "noun": "your fraud protection", "desc": "The customer wants a suspicious charge reported."},
    "book_callback": {"imp": "book a callback from an agent", "past": "I already booked a callback",
                      "noun": "your callback hours", "desc": "The customer wants a phone callback scheduled."},
}
_POS = ["Please {imp}.", "I need you to {imp}.", "Could you {imp}?", "What I want is for you to {imp}."]
_NEG = ["Please do not {imp}.", "Do not {imp}, I changed my mind about that.", "I no longer want you to {imp}."]
_DONE = ["{past}, so no action is needed there.", "{past}.", "{past}, thanks anyway."]
_PRAISE = ["Thanks, {noun} makes sense to me now.", "I read {noun} and it seems fair.",
           "To be clear, {noun} is fine with me."]


def _nonreq(rng, key):
    it = INTENTS[key]
    kind = rng.choice(["neg", "done", "praise"])
    tpl = rng.choice({"neg": _NEG, "done": _DONE, "praise": _PRAISE}[kind])
    return tpl.format(imp=it["imp"], past=it["past"], noun=it["noun"]), kind


def _gen_intent(rng, seed, j):
    allk = list(INTENTS)
    opts = rng.sample(allk, rng.randint(4, 6))
    other = rng.random() < 0.3
    pieces = []
    if other:
        real = None
        pool = [k for k in allk]
        for k in rng.sample(pool, rng.randint(1, 3)):
            pieces.append(_nonreq(rng, k)[0])
        witness_free = True
    else:
        real = rng.choice(opts)
        pos = rng.choice(_POS).format(imp=INTENTS[real]["imp"])
        pos = pos.replace(INTENTS[real]["imp"], mark(INTENTS[real]["imp"]))
        pieces.append(pos)
        for k in rng.sample([x for x in allk if x != real], rng.randint(1, 2)):
            pieces.append(_nonreq(rng, k)[0])
        witness_free = False
    rng.shuffle(pieces)
    if not other and pieces[0] == pieces[-1]:
        pass
    text = " ".join(pieces)
    if rng.random() < 0.4:
        text = rng.choice(["Hello, ", "Hi there. ", "Good morning. "]) + text
    criteria = {k: INTENTS[k]["desc"] for k in opts}
    criteria["other"] = "None of these actions is requested."
    gold = "other" if other else real
    return _rec("intent_negation", seed, j, [("user", "Customer message: " + text)], "choice",
                "Pick the one action the customer is actually asking for now. Mentions of an action that is "
                "refused, already done or only discussed are not requests.", criteria, gold,
                {"real_intent": real})


# =========================================================== policy_exemption

_CRIT = [
    {"true": "Every required condition is established and no prohibition applies.",
     "false": "A condition is missing or a prohibition applies."},
    {"true": "All requirements are satisfied or exempted and nothing prohibits the action.",
     "false": "A requirement is unmet without an exemption, or a prohibition applies."},
    {"true": "The policy permits the action for this case.",
     "false": "The policy does not permit the action for this case."},
]
_S1 = [
    ("Brannock Labs", "submit a signed access form", "enter the cleanroom", "visiting auditors",
     "a visiting auditor", "a staff technician", "submitted the signed access form",
     "has not submitted the signed access form"),
    ("Quillon Logistics", "complete the forklift safety module", "operate a forklift in Bay 4", "certified trainers",
     "a certified trainer", "a seasonal packer", "completed the forklift safety module",
     "has not completed the forklift safety module"),
    ("Tessaly Outfitters", "show a manager's countersignature", "take stock home for repair",
     "members of the repair guild", "a repair guild member", "a sales associate",
     "has a manager's countersignature", "has no manager's countersignature"),
    ("Hollin Dock Services", "hold a valid berth permit", "moor overnight at Pier 9", "emergency vessels",
     "an emergency vessel", "a private yacht", "holds a valid berth permit", "holds no berth permit"),
]
_S2 = [
    ("Ostrava Pine Cooperative", "book the rooftop room", "number of guests", "members of the events committee",
     "a member of the events committee", "an ordinary member"),
    ("Corvane Instruments", "borrow a calibration unit", "number of days requested", "lab supervisors",
     "a lab supervisor", "a junior technician"),
    ("Veridian Tidewater Group", "approve a team outing", "number of attendees", "directors",
     "a director", "a project lead"),
    ("Marlowe & Finch Freight", "send a rush shipment", "number of pallets", "premium accounts",
     "a premium account holder", "a standard account holder"),
]
_S3 = [
    ("Quillon Logistics", "A driver may take the company van home when the driver lives more than 40 km from the depot.",
     "take the company van home", "drivers on probation", "lives more than 40 km from the depot",
     "lives 12 km from the depot", "is on probation", "is not on probation"),
    ("Brannock Labs", "A guest may use the instrument room when a staff sponsor is present.",
     "use the instrument room", "guests from competing firms", "has a staff sponsor present",
     "has no staff sponsor present", "works for a competing firm", "does not work for a competing firm"),
]


def _gen_exemption(rng, seed, j):
    target = rng.random() < 0.5
    for _ in range(200):
        out = _exemption_once(rng)
        if out["ans"] == target:
            break
    crit = rng.choice(_CRIT)
    ptype_pol, ptype_case = "retrieved_doc", "tool_output"
    segs = [(ptype_pol, out["policy"]), (ptype_case, out["case"])]
    gold = "yes" if out["ans"] else "no"
    return _rec("policy_exemption", seed, j, segs, "noul",
                "Apply the policy to the case. Is the requested action permitted?", crit, gold,
                {"structure": out["structure"], "facts": out["facts"]})


def _exemption_once(rng):
    st = rng.choice(["s1", "s2", "s3"])
    name = rng.choice(PEOPLE)
    if st == "s1":
        org, req, act, ex, exm, exn, ok, no = rng.choice(_S1)
        exempt, met = rng.random() < 0.4, rng.random() < 0.5
        req_s = mark(rng.choice([f"At {org}, a person must {req} before they {act}.",
                                 f"Anyone who wants to {act} at {org} is required to {req} first."]))
        ex_s = mark(rng.choice([f"This requirement does not apply to {ex}.", f"{ex[0].upper()+ex[1:]} are exempt.",
                                f"The requirement is waived for {ex}."]))
        pol = " ".join([req_s, ex_s])
        c1 = f"{name} is {exm}." if exempt else f"{name} is {exn}."
        c1 = c1.replace(exm if exempt else exn, mark(exm if exempt else exn), 1)
        r_txt = ok if met else no
        c2 = f"{name} {mark(r_txt)}."
        facts = [c1, c2]
        rng.shuffle(facts)
        case = " ".join(facts) + f" Question: may {name} {act}?"
        return {"structure": "requirement_exemption", "policy": pol, "case": "Case: " + case,
                "ans": exempt or met, "facts": {"exempt": exempt, "met": met}}
    if st == "s2":
        org, act, metric, ex, exm, exn = rng.choice(_S2)
        n = rng.randint(3, 30)
        cmp = rng.choice(["up to and including", "fewer than", "at least"])
        v = rng.choice([n - 2, n - 1, n, n, n + 1, n + 2])
        v = max(v, 1)
        ok = {"up to and including": v <= n, "fewer than": v < n, "at least": v >= n}[cmp]
        exempt = rng.random() < 0.3
        b_s = mark(rng.choice([f"At {org}, anyone may {act} only if the {metric} is {cmp} {n}.",
                               f"Permission to {act} at {org} requires the {metric} to be {cmp} {n}."]))
        ex_s = mark(rng.choice([f"The limit does not apply to {ex}.", f"{ex[0].upper()+ex[1:]} are exempt from the limit.",
                                f"Except for {ex}, who are not bound by it."]))
        c1 = f"{name} is {exm if exempt else exn}."
        t = exm if exempt else exn
        c1 = c1.replace(t, mark(t), 1)
        c2 = f"The {metric} is {mark(str(v))}."
        facts = [c1, c2]
        rng.shuffle(facts)
        case = " ".join(facts) + f" Question: may {name} {act}?"
        return {"structure": "boundary_exemption", "policy": b_s + " " + ex_s, "case": "Case: " + case,
                "ans": exempt or ok, "facts": {"n": n, "cmp": cmp, "v": v, "exempt": exempt}}
    org, pol_s, act, ban, okt, nokt, bant, nbant = rng.choice(_S3)
    met, banned = rng.random() < 0.6, rng.random() < 0.4
    pol = mark(pol_s) + " " + mark(rng.choice([f"It is never allowed for {ban}.", f"This is prohibited for {ban}."]))
    t1 = okt if met else nokt
    t2 = bant if banned else nbant
    f1 = f"{name} {mark(t1)}."
    f2 = f"{name} {mark(t2)}."
    facts = [f1, f2]
    rng.shuffle(facts)
    case = " ".join(facts) + f" Question: may {name} {act}?"
    return {"structure": "prohibition", "policy": pol, "case": "Case: " + case, "ans": met and not banned,
            "facts": {"met": met, "banned": banned}}


# =========================================================== adequacy

_CATALOGS = [
    ("department", "Kestrel Mart store departments", ["grocery", "garden", "pharmacy", "electronics", "bakery", "pet supplies"]),
    ("workshop", "Ostrava Pine Cooperative workshops", ["joinery", "weaving", "pottery", "glassblowing", "tanning"]),
    ("route", "Marlowe ferry routes", ["Dunmore", "Skerry", "Tallow Bay", "Pellin", "Oarsby"]),
]
_WRONG_ITEMS = ["florist", "archery", "bowling", "chandlery", "Ravenmoor", "locksmith", "tailoring"]
_ARRAYS = [("the three primary colours of additive light", ["red", "green", "blue"]),
           ("the four seasons of the year", ["spring", "summer", "autumn", "winter"]),
           ("the two weekend days", ["Saturday", "Sunday"]),
           ("the first three letters of the Greek alphabet", ["alpha", "beta", "gamma"])]
_OBJECTS = [("where Hollin Dock is located, with keys \"city\" and \"country\"", {"city": "Valmere", "country": "Norland"}),
            ("the Corvane Model 7 meter, with keys \"name\" and \"year\"", {"name": "Model 7", "year": 2019}),
            ("the Tessaly parka, with keys \"colour\", \"size\" and \"price\"", {"colour": "slate", "size": "M", "price": 89})]
_WC_THINGS = ["a quiet morning", "the sea at night", "a busy market", "an old library", "a mountain trail"]
_WC_WORDS = ["calm", "bright", "weathered", "silver", "crowded", "slow", "golden", "hushed", "damp", "distant",
             "cold", "restless", "fragrant", "dim", "windy"]
_PLANS = ["Lumen", "Harbor", "Ember", "Quarry", "Tallis"]
_FEATS = ["priority support", "offline mode", "audit logs", "custom domains", "api access", "team folders"]
_PRODUCTS = ["Corvane meter", "Kestrel drill", "Veridian pump", "Quillon scanner"]


def _parse_items(resp: str) -> List[str]:
    s = resp.strip()
    if ":" in s:
        s = s.rsplit(":", 1)[1]
    s = s.strip().rstrip(".")
    return sorted(x.strip().lower() for x in re.split(r"[,;]", s) if x.strip())


def _check_format(kind: str, resp: str, keys=None) -> bool:
    try:
        v = json.loads(resp)
    except ValueError:
        return False
    if kind == "format_array":
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    if kind == "format_object":
        return isinstance(v, dict) and set(v) == set(keys)
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _gen_adequacy(rng, seed, j):
    kind = rng.choice(["list_all", "format", "word_count", "yes_no"])
    good = rng.random() < 0.5
    extra: Dict[str, Any] = {}
    ref = None
    if kind == "list_all":
        noun, title, pool = rng.choice(_CATALOGS)
        items = rng.sample(pool, rng.randint(2, 4))
        ref = f"{title}: " + ", ".join(items) + "."
        req_s = f"Using the reference, list {mark('every ' + noun)} it names."
        resp_items = list(items)
        if not good:
            if rng.random() < 0.5 and len(items) > 2 or len(items) == 2 and rng.random() < 0.5:
                resp_items.remove(rng.choice(resp_items))
            else:
                resp_items.append(rng.choice([w for w in _WRONG_ITEMS if w not in items]))
        rng.shuffle(resp_items)
        resp = rng.choice(["", "Here they are: "]) + ", ".join(resp_items) + "."
        gold_b = _parse_items(resp) == sorted(x.lower() for x in items)
        extra = {"check": "list_all", "reference_items": items, "response": resp}
        req_segs = [("user", "Request: " + req_s)]
    elif kind == "format":
        sub = rng.choice(["format_array", "format_object", "format_number"])
        keys = None
        if sub == "format_array":
            desc, val = rng.choice(_ARRAYS)
            req_s = f"Give {desc} as {mark('a JSON array of strings')}, with no other text."
            good_r = json.dumps(val)
            bads = ["['%s']" % "', '".join(val), ", ".join(val), "The answer is " + json.dumps(val),
                    json.dumps({"items": val}), json.dumps(", ".join(val))]
        elif sub == "format_object":
            desc, val = rng.choice(_OBJECTS)
            keys = list(val)
            req_s = f"Describe {desc}. Reply with {mark('a JSON object')} only."
            good_r = json.dumps(val)
            bads = [json.dumps(list(val.items())), "{" + ", ".join(f"{k}: {v}" for k, v in val.items()) + "}",
                    json.dumps({k: v for k, v in list(val.items())[:-1]}),
                    "Sure! " + json.dumps(val)]
        else:
            a, b = rng.randint(3, 12), rng.randint(3, 12)
            req_s = f"What is {a} times {b}? Reply with {mark('a plain number')} only."
            good_r = str(a * b)
            bads = [f"about {a*b}", json.dumps(str(a * b)), f"{a*b} in total", f"{a*b}.", f"The result is {a*b}"]
        resp = good_r if good else rng.choice(bads)
        gold_b = _check_format(sub, resp, keys)
        extra = {"check": sub, "response": resp, "required_keys": keys}
        req_segs = [("user", "Request: " + req_s)]
    elif kind == "word_count":
        n = rng.randint(1, 5)
        thing = rng.choice(_WC_THINGS)
        req_s = f"Describe {thing} in {mark('exactly ' + str(n) + (' word' if n == 1 else ' words'))}."
        m = n if good else rng.choice([x for x in range(1, 8) if x != n])
        words = rng.sample(_WC_WORDS, m)
        resp = " ".join(words).capitalize() + rng.choice([".", ""])
        gold_b = len(resp.split()) == n
        extra = {"check": "word_count", "target_words": n, "response": resp}
        req_segs = [("user", "Request: " + req_s)]
    else:
        if rng.random() < 0.5:
            plan = rng.choice(_PLANS)
            feats = rng.sample(_FEATS, 3)
            ref = f"The {plan} plan includes: " + ", ".join(feats) + "."
            ask = rng.choice(feats) if rng.random() < 0.5 else rng.choice([f for f in _FEATS if f not in feats])
            truth = ask in feats
            req_s = f"Answer yes or no, using the reference: {mark('Does the ' + plan + ' plan include ' + ask + '?')}"
            extra = {"check": "yes_no", "truth": truth}
        else:
            prod = rng.choice(_PRODUCTS)
            yrs = rng.randint(2, 9)
            m = max(1, yrs + rng.choice([-1, 0, 1]))
            ref = f"The warranty on the {prod} lasts {yrs} years."
            truth = yrs > m
            req_s = f"Answer yes or no, using the reference: {mark('Is the warranty on the ' + prod + ' longer than ' + str(m) + ' years?')}"
            extra = {"check": "yes_no", "truth": truth}
        said_yes = truth if good else (not truth)
        resp = rng.choice(["Yes.", "Yes, it does.", "Yes"]) if said_yes else rng.choice(["No.", "No, it does not.", "No"])
        gold_b = resp.strip().lower().startswith("yes") == truth
        extra["response"] = resp
        req_segs = [("user", "Request: " + req_s)]
    segs = list(req_segs)
    if ref:
        segs.append(("retrieved_doc", "Reference: " + ref))
    segs.append(("tool_output", "Response: " + mark(resp)))
    extra["kind"] = kind
    crit = {"true": "The response is correct, complete, and follows every constraint in the request.",
            "false": "The response is wrong, incomplete, or breaks a constraint in the request."}
    return _rec("adequacy", seed, j, segs, "noul",
                "Does the response fully satisfy the request, using the reference when present?", crit,
                "yes" if gold_b else "no", extra)


# =========================================================== long_policy

_TITLES = ["Team Lead", "Department Head", "Director", "Finance Committee", "Executive Board"]
_CATS = ["data hosting", "air travel", "legal services", "consulting", "software licences", "security equipment",
         "facility construction", "marketing agencies", "vehicle leasing", "laboratory chemicals"]


def _dstr(d: _dt.date) -> str:
    return d.isoformat()


def _distractors(rng, org):
    n = lambda a, b: rng.randint(a, b)  # noqa: E731
    return [
        ("Records Retention", f"Procurement files, including quotes, purchase orders and delivery notes, are kept for {n(5,9)} years after the final payment. Electronic copies are stored in the contract repository and paper originals may be destroyed after {n(12,36)} months once scanned. Records subject to a dispute are kept until the dispute is closed, and the retention clock restarts when the dispute ends. Requests to retrieve archived files must be filed with the records desk at least {n(2,5)} business days ahead."),
        ("Vendor Onboarding", f"New vendors must complete the onboarding form, supply a tax certificate and accept the {org} supplier code. The vendor team reviews each application within {n(5,15)} business days. Vendors that supply goods valued above ${n(2,9)*1000:,} per year are reviewed again every {n(1,3)} years. Banking details may only be changed after a call-back to a number already on file, and a change takes effect {n(2,6)} days after confirmation."),
        ("Travel Booking Windows", f"Staff should book flights at least {n(10,21)} days before departure and rail tickets at least {n(3,7)} days before. Hotel stays are limited to ${n(120,220)} per night in standard cities and ${n(240,320)} in listed high-cost cities. Rebooking fees are reimbursed only when the change was caused by the business. Car rental is available for trips longer than {n(80,160)} km when no rail option exists."),
        ("Conflict of Interest", f"Anyone who has a personal or financial relationship with a bidding vendor must disclose it in writing before the evaluation starts. The ethics officer decides within {n(3,8)} business days whether the person must step back from the decision. Gifts worth more than ${n(40,90)} must be declined or handed to the ethics officer. Disclosures are renewed every {n(12,24)} months, and false disclosures may lead to disciplinary measures."),
        ("Purchase Order Numbering", f"Every purchase order carries a number made of the cost centre code, the year and a running counter of {n(4,6)} digits. Numbers are never reused, even when an order is cancelled. A cancelled order keeps its number and is marked void in the register. Orders raised outside the system are assigned a number by the procurement desk within {n(1,3)} business days and must be keyed in by the requester."),
        ("Receiving and Inspection", f"Goods are inspected at the delivery site on the day of arrival. Discrepancies in quantity of more than {n(2,6)} percent must be reported to the buyer within {n(2,5)} business days. Damaged goods are photographed and held until the vendor confirms collection. Services are accepted only after the requester signs the completion note, and acceptance of partial deliveries does not waive any claim for the remainder."),
        ("Invoice Payment Terms", f"Standard payment terms are net {n(20,60)} days from the date the invoice is received in the accounts mailbox. Early payment discounts of up to {n(1,3)} percent are taken only when the discount is at least equal to the cost of funds. Invoices missing a purchase order number are returned unpaid. Disputed invoices are paid in part when the undisputed portion can be identified, and the remainder waits for resolution."),
        ("Training and Certification", f"All staff who can approve purchases must complete the procurement refresher every {n(12,24)} months. The course takes about {n(2,4)} hours and ends with a quiz that requires {n(70,90)} percent to pass. New approvers have {n(30,60)} days from appointment to finish the course. Missed deadlines suspend approval rights until the course is completed, and the suspension is recorded in the approver register."),
        ("Data Protection in Procurement Files", f"Personal data in quotes and contracts, such as names and phone numbers of vendor staff, is used only for managing the contract. Access to procurement files is limited to the buying team and the audit group. Exports to personal devices are not allowed. A suspected leak must be reported to the privacy desk within {n(1,3)} business days, and the desk logs each report with a reference number."),
        ("Sustainability Preference", f"When two offers are within {n(2,6)} percent of each other on price, the buyer should prefer the supplier with the stronger environmental record. Evidence includes recognised certifications and published emission figures. Buyers record the reason for choosing a higher-priced offer. The sustainability team publishes a short summary of such decisions every {n(3,6)} months for internal review."),
        ("Emergency Procurement", f"When a failure threatens safety or continuity, a buyer may place an order by phone and must document it within {n(1,3)} business days. The emergency route does not change the approval level required by Section 3, and the approver must be told as soon as possible. Emergency orders are listed in a separate register that is reviewed monthly by the finance team, and repeated use by one team triggers a planning review."),
        ("Audit Sampling", f"Internal audit tests a sample of about {n(3,8)} percent of purchase orders each quarter, chosen at random and by risk. Auditors may request quotes, approval records and delivery notes. Teams have {n(5,10)} business days to answer each request. Findings are graded minor, moderate or major, and major findings must be answered with a written action plan within {n(20,40)} days."),
        ("Document Templates", f"The latest request forms, quote comparison sheets and vendor evaluation templates are kept on the procurement intranet page. Local copies must be refreshed at least every {n(3,6)} months because templates change. Forms must be submitted as PDF or in the web tool, and handwritten changes to a signed form invalidate it. Questions about templates go to the help desk, which replies within {n(1,3)} business days."),
        ("Complaints and Disputes", f"Vendors that wish to challenge an award can write to the procurement director within {n(5,15)} business days of the decision. The director replies within {n(10,20)} business days. A challenge does not pause an award unless the director says so in writing. Internal complaints about the process are handled by the compliance office and are tracked in the same register."),
    ]


def _gen_long_policy(rng, seed, j):
    org = rng.choice(ORGS)
    K = rng.choice([3, 4, 5])
    b0 = rng.choice([200, 500, 1000, 2000])
    bounds = []
    v = b0
    for _ in range(K - 1):
        bounds.append(v)
        v *= rng.choice([2, 3, 4, 5])

    def level_for(a):
        for i, b in enumerate(bounds, 1):
            if a <= b:
                return i
        return K

    titles = _TITLES[:K]
    restricted = rng.sample(_CATS, 3)
    unrestricted = [c for c in _CATS if c not in restricted]
    # case facts
    lvl_idx = rng.randint(1, K)
    lo = 1 if lvl_idx == 1 else bounds[lvl_idx - 2] + 1
    hi = bounds[lvl_idx - 1] if lvl_idx <= K - 1 else bounds[-1] * 2
    amount = rng.choice([lo, hi]) if rng.random() < 0.3 else rng.randint(lo, hi)
    base = level_for(amount)
    cat = rng.choice(restricted) if rng.random() < 0.4 else rng.choice(unrestricted)
    quotes = rng.randint(0, 5)
    framework = rng.random() < 0.35
    suspended = rng.random() < 0.12
    # amendments
    kinds = rng.sample(["raise", "floor", "cap", "prohibit"], rng.randint(2, 3))
    d0 = _dt.date(2024, 2, 1).toordinal() + rng.randint(0, 200)
    dates = sorted(d0 + rng.randint(0, 560) for _ in kinds)
    adates = [_dt.date.fromordinal(o) for o in dates]
    lo_d, hi_d = adates[0] - _dt.timedelta(days=60), adates[-1] + _dt.timedelta(days=240)
    po = lo_d + _dt.timedelta(days=rng.randint(0, (hi_d - lo_d).days))
    floor_lvl = min(K, rng.choice([2, 3]))
    cap_lvl = rng.choice([1, 2])
    conds = {"raise": cat in restricted, "floor": quotes < 3, "cap": framework, "prohibit": suspended}
    level, prohibited, applied = base, False, []
    for idx, (k, d) in enumerate(zip(kinds, adates)):
        if po >= d and conds[k]:
            applied.append(idx)
            if k == "raise":
                level = min(K, level + 1)
            elif k == "floor":
                level = max(level, floor_lvl)
            elif k == "cap":
                level = min(level, cap_lvl)
            else:
                prohibited = True
    gold = "prohibited" if prohibited else f"level_{level}"
    trig_fact = {"raise": "cat", "floor": "quotes", "cap": "framework", "prohibit": "vendor"}
    marked_facts = {"amount"} | ({"po"} if applied else set()) | {trig_fact[kinds[i]] for i in applied}

    # document body
    def band_clause(i):
        t = titles[i - 1]
        if i == 1:
            s = f"Level 1 ({t}) may approve a purchase of up to and including ${bounds[0]:,}."
        elif i < K:
            s = f"Level {i} ({t}) approves a purchase of more than ${bounds[i-2]:,} and up to and including ${bounds[i-1]:,}."
        else:
            s = f"Level {K} ({t}) approves any purchase of more than ${bounds[-1]:,}."
        return s

    pool = _distractors(rng, org)
    rng.shuffle(pool)
    sec1 = f"1. Purpose and Scope\n1.1 This policy sets out who may approve spending at {org} and how a purchase request is handled. " \
           f"{rng.choice(['It applies to every employee, contractor and temporary worker who raises or approves purchases.', 'It covers all buying done in the name of the company, whatever the budget source.'])} " \
           f"1.2 Where this policy and a team guideline differ, this policy prevails unless an amendment in the last section says otherwise."
    sec2 = ("2. Definitions\n2.1 \"Purchase\" means any commitment to pay a third party for goods or services.\n"
            f"2.2 \"Restricted Category\" means a purchase in any of the following categories: {restricted[0]}, {restricted[1]} or {restricted[2]}.\n"
            "2.3 \"Framework Agreement\" means a signed supply contract held by procurement that has not expired.\n"
            "2.4 \"Suspended Vendor\" means a vendor shown as suspended in the vendor register on the purchase order date.\n"
            "2.5 \"Single-Source Purchase\" means a purchase for which fewer than three written quotes were obtained.\n"
            "2.6 \"Purchase Order Date\" means the date printed on the purchase order.")
    bands = []
    for i in range(1, K + 1):
        c = band_clause(i)
        bands.append((i, f"3.{i} " + c))
    band_lines = []
    for i, line in bands:
        if i == base:
            band_lines.append(f"3.{i} " + mark(band_clause(i)))
        else:
            band_lines.append(line)
    sec3_head = "3. Approval Levels\n"
    sec3_tail = f"\n3.{K+1} The level required is the lowest level whose band contains the amount, unless a later section or an amendment says otherwise."
    sec3 = sec3_head + "\n".join(band_lines) + sec3_tail
    fixed = [sec1, sec2, sec3]

    amend_texts = []
    for idx, (k, d) in enumerate(zip(kinds, adates), 1):
        ds = _dstr(d)
        if k == "raise":
            t = f"Amendment {idx} (effective {ds}). For any purchase order dated on or after {ds}, a purchase in a Restricted Category requires approval one level higher than Section 3 would otherwise require, but never above Level {K}."
        elif k == "floor":
            t = f"Amendment {idx} (effective {ds}). For any purchase order dated on or after {ds}, a Single-Source Purchase requires approval at Level {floor_lvl} or higher, whatever its amount."
        elif k == "cap":
            t = f"Amendment {idx} (effective {ds}). For any purchase order dated on or after {ds}, a purchase under a Framework Agreement requires approval at no higher than Level {cap_lvl}, whatever its amount."
        else:
            t = f"Amendment {idx} (effective {ds}). For any purchase order dated on or after {ds}, no purchase may be made from a Suspended Vendor."
        amend_texts.append(t)

    case_lines_plain = {
        "po": f"Purchase order date: {_dstr(po)}",
        "amount": f"Amount: ${amount:,}",
        "cat": f"Category: {cat}",
        "quotes": f"Quotes obtained: {quotes}",
        "framework": f"Framework Agreement on file: {'yes' if framework else 'no'}",
        "vendor": f"Vendor register status: {'suspended' if suspended else 'active'}",
        "req": f"Requested by: {rng.choice(PEOPLE)}",
        "cc": f"Cost centre: CC-{rng.randint(100, 999)}",
        "site": f"Delivery site: {rng.choice(['Dock 2', 'Main office', 'North depot', 'Lab annex'])}",
        "need": f"Needed within: {rng.randint(5, 60)} days of approval",
    }
    keys = list(case_lines_plain)
    rng.shuffle(keys)
    case_lines = ["- " + (mark(case_lines_plain[k]) if k in marked_facts else case_lines_plain[k]) for k in keys]
    case = f"CASE FILE PR-{rng.randint(1000, 9999)}\n" + "\n".join(case_lines)

    # choose distractors to reach target length
    target = rng.randint(4400, 8800)
    amend_len = sum(len(t) for t in amend_texts) + 100
    cur = sum(len(s) for s in fixed) + amend_len + len(case) + 120
    chosen = []
    for title, body in pool:
        if cur + len(body) + len(title) + 12 > min(target, 8900):
            continue
        chosen.append((title, body))
        cur += len(body) + len(title) + 12
    secs = list(fixed)
    for i, (title, body) in enumerate(chosen, 4):
        secs.append(f"{i}. {title}\n{i}.1 " + body)
    asec = len(secs) + 1
    am_parts = []
    for idx, t in enumerate(amend_texts):
        am_parts.append(f"{asec}.{idx+1} " + (mark(t) if idx in applied else t))
    secs.append(f"{asec}. Amendments and Endorsements\n"
                "Amendments are applied in numerical order after the level has been set under Section 3, and each one operates on the result of the one before it.\n"
                + "\n".join(am_parts))
    title_line = f"{org.upper()} PURCHASE AUTHORITY POLICY"
    doc = title_line + "\n\n" + "\n\n".join(secs)

    criteria = {f"level_{i}": f"The highest approval level required is Level {i} ({titles[i-1]})." for i in range(1, K + 1)}
    criteria["prohibited"] = "The purchase may not proceed under this policy."
    decisive = amend_texts[applied[-1]] if applied else None
    segs = [("retrieved_doc", doc), ("tool_output", case)]
    return _rec("long_policy", seed, j, segs, "choice",
                "Using the policy and the case file, what is the highest approval level required for this purchase? "
                "Apply the base rule first, then every amendment whose condition the case meets.", criteria, gold,
                {"base_level": base, "applied": applied, "decisive_amendment": decisive,
                 "amendment_texts": amend_texts, "kinds": kinds})


# =========================================================== conflict

_TOPICS = [
    {"subj": "the nightly parcel export", "q": "Did the nightly parcel export complete?", "svc": "export-runner",
     "tile": "Nightly Export", "A": "export_completed", "B": "export_failed",
     "dA": "The nightly parcel export completed.", "dB": "The nightly parcel export did not complete.",
     "pA": ["job finished with exit status 0", "export marked COMPLETE"], "pB": ["job aborted with exit status 3", "export marked FAILED"],
     "cA": ["the export went through fine", "export looks done to me"], "cB": ["the export blew up again", "export is dead, nothing was written"],
     "other": "whether the job was queued", "event": "the 02:00 job restart"},
    {"subj": "the Kestrel gate position", "q": "Is the Kestrel loading gate open?", "svc": "gate-controller",
     "tile": "Kestrel Gate", "A": "gate_open", "B": "gate_closed",
     "dA": "The Kestrel loading gate is open.", "dB": "The Kestrel loading gate is closed.",
     "pA": ["sensor reads OPEN", "position 100 percent"], "pB": ["sensor reads CLOSED", "position 0 percent"],
     "cA": ["I can see the gate standing open", "gate is open, trucks rolling in"], "cB": ["the gate is shut, I just walked past", "gate is closed"],
     "other": "the last command sent to the gate", "event": "the sensor firmware update"},
    {"subj": "the quickpay feature flag", "q": "Is the quickpay flag enabled in production?", "svc": "flag-service",
     "tile": "Feature Flags", "A": "flag_on", "B": "flag_off",
     "dA": "The quickpay flag is enabled in production.", "dB": "The quickpay flag is disabled in production.",
     "pA": ["quickpay=true", "quickpay enabled for 100 percent of traffic"], "pB": ["quickpay=false", "quickpay disabled for all traffic"],
     "cA": ["quickpay is live now", "I switched quickpay on"], "cB": ["quickpay is still off", "we rolled quickpay back"],
     "other": "the flag value in the staging environment", "event": "last week's config rollback"},
    {"subj": "invoice batch 7731", "q": "Has invoice batch 7731 been paid?", "svc": "ledger-sync",
     "tile": "Invoice Batches", "A": "batch_paid", "B": "batch_unpaid",
     "dA": "Invoice batch 7731 has been paid.", "dB": "Invoice batch 7731 has not been paid.",
     "pA": ["batch 7731 settled", "payment confirmed for batch 7731"], "pB": ["batch 7731 pending", "no payment recorded for batch 7731"],
     "cA": ["7731 went out this morning", "the money for 7731 is gone"], "cB": ["7731 is still waiting on approval", "nothing has been sent for 7731"],
     "other": "whether the batch was submitted for payment", "event": "the month-end close"},
    {"subj": "node orchard-3", "q": "Is node orchard-3 healthy?", "svc": "health-probe",
     "tile": "Cluster Nodes", "A": "node_healthy", "B": "node_down",
     "dA": "Node orchard-3 is healthy.", "dB": "Node orchard-3 is down.",
     "pA": ["orchard-3 responded in 40 ms", "orchard-3 status OK"], "pB": ["orchard-3 unreachable", "orchard-3 status DOWN"],
     "cA": ["orchard-3 is answering fine", "orchard-3 looks healthy from here"], "cB": ["orchard-3 is not answering", "orchard-3 fell over"],
     "other": "whether the node is registered in the cluster", "event": "the network maintenance window"},
    {"subj": "the Lumen pallet shipment", "q": "Was the Lumen pallet delivered?", "svc": "carrier-feed",
     "tile": "Shipments", "A": "delivered", "B": "not_delivered",
     "dA": "The Lumen pallet was delivered.", "dB": "The Lumen pallet was not delivered.",
     "pA": ["status DELIVERED, signed at dock", "pallet delivered"], "pB": ["status IN TRANSIT", "delivery attempt failed"],
     "cA": ["the pallet arrived at the dock", "got the Lumen pallet this morning"], "cB": ["no pallet yet", "the Lumen pallet never showed up"],
     "other": "whether the pallet left the depot", "event": "the carrier's system migration"},
    {"subj": "database migration 0412", "q": "Has migration 0412 been applied?", "svc": "migrator",
     "tile": "Migrations", "A": "migration_applied", "B": "migration_pending",
     "dA": "Migration 0412 has been applied.", "dB": "Migration 0412 is still pending.",
     "pA": ["0412 applied", "migration 0412 recorded in schema table"], "pB": ["0412 pending", "migration 0412 not found in schema table"],
     "cA": ["0412 ran last night", "I applied 0412 already"], "cB": ["0412 has not run yet", "still waiting to run 0412"],
     "other": "whether the migration file exists", "event": "the schema table restore"},
    {"subj": "the Tallis certificate", "q": "Has the Tallis TLS certificate been renewed?", "svc": "cert-watch",
     "tile": "Certificates", "A": "cert_renewed", "B": "cert_expired",
     "dA": "The Tallis certificate has been renewed.", "dB": "The Tallis certificate has expired.",
     "pA": ["new certificate valid until next year", "renewal succeeded"], "pB": ["certificate expired", "renewal failed, certificate expired"],
     "cA": ["the cert is renewed", "renewed it this afternoon"], "cB": ["browsers are complaining, the cert expired", "cert is expired"],
     "other": "whether a renewal was requested", "event": "the certificate authority outage"},
]
_SRC_KINDS = {"log": ("the system log", "tool_output"), "dashboard": ("the dashboard", "tool_output"),
              "chat": ("the chat message", "user"), "ticket": ("the ticket", "retrieved_doc")}


def _evidence(rng, kind, topic, reading):
    ph = rng.choice(topic["p" + reading])
    ca = rng.choice(topic["c" + reading])
    if kind == "log":
        return f"[{rng.randint(0,23):02d}:{rng.randint(0,59):02d}] {topic['svc']}: ", ph, ""
    if kind == "dashboard":
        return f"Dashboard tile \"{topic['tile']}\": ", ph, ""
    if kind == "chat":
        return f"{rng.choice(PEOPLE)} in #ops: \"", ca, "\""
    return f"Ticket OPS-{rng.randint(100, 999)} ({rng.choice(PEOPLE)}): ", ph, ""


def _gen_conflict(rng, seed, j):
    t = rng.choice(_TOPICS)
    has_rule = rng.random() < 0.6
    nsrc = rng.choice([2, 3]) if has_rule else 2
    kinds = rng.sample(list(_SRC_KINDS), nsrc)
    readings = ["A", rng.choice(["A", "B"]) if nsrc == 3 else "B"]
    readings = ["A", "B"] + ([rng.choice(["A", "B"])] if nsrc == 3 else [])
    rng.shuffle(readings)
    if readings[0] == readings[1]:
        readings[1] = "B" if readings[0] == "A" else "A"
    swap = rng.random() < 0.5  # which reading is called A in the option list
    valid = list(range(nsrc))
    rule_seg = None
    rule_kind = None
    rule_idx = None
    if has_rule:
        rule_idx = rng.randrange(nsrc)
        name = _SRC_KINDS[kinds[rule_idx]][0]
        rule_kind = rng.choice(["authoritative", "stale", "other_measure"])
        if rule_kind == "authoritative":
            rule = f"Runbook note: for {t['subj']}, {name} is the system of record; where sources disagree, follow it."
            valid = [rule_idx]
        elif rule_kind == "stale":
            rule = f"Runbook note: {name} was last refreshed before {t['event']} and should be disregarded."
            valid = [i for i in range(nsrc) if i != rule_idx]
        else:
            rule = f"Runbook note: {name} reports {t['other']}, not {t['subj']}, so it does not settle this question."
            valid = [i for i in range(nsrc) if i != rule_idx]
    vals = {readings[i] for i in valid}
    gold_reading = vals.pop() if len(vals) == 1 else None
    # build segments
    parts = []
    for i, k in enumerate(kinds):
        pre, body, post = _evidence(rng, k, t, readings[i])
        witnessed = gold_reading is not None and i in valid
        parts.append((_SRC_KINDS[k][1], pre + (mark(body) if witnessed else body) + post))
    if not has_rule and rng.random() < 0.3:
        parts.append(("user", f"{rng.choice(PEOPLE)} in #ops: \"anyone seen the new badge printer?\""))
    rng.shuffle(parts)
    segs = [("user", "Question: " + t["q"])] + parts
    if has_rule:
        rs = mark(rule) if gold_reading is not None else rule
        pos = rng.choice([1, len(segs)])
        segs.insert(pos, ("system", rs))
    a, b = (t["A"], t["B"]) if not swap else (t["B"], t["A"])
    da, db = (t["dA"], t["dB"]) if not swap else (t["dB"], t["dA"])
    criteria = {a: da, b: db,
                "cannot_determine": "The evidence genuinely conflicts and nothing resolves it."}
    if gold_reading is None:
        gold = "cannot_determine"
    else:
        gold = t[gold_reading]
    return _rec("conflict", seed, j, segs, "choice",
                "Decide which reading the evidence supports, following any stated rule about which source "
                "to trust. If the evidence conflicts and nothing resolves it, say so.", criteria, gold,
                {"has_rule": has_rule, "rule_kind": rule_kind, "valid_sources": [kinds[i] for i in valid]})


# =========================================================== api

_GENS = {"routing": _gen_routing, "intent_negation": _gen_intent, "policy_exemption": _gen_exemption,
         "adequacy": _gen_adequacy, "long_policy": _gen_long_policy, "conflict": _gen_conflict}


def _item(family: str, j: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(f"{SOURCE}|{family}|{seed}|{j}")
    return _GENS[family](rng, seed, j)


def generate_family(name: str, n: int, seed: int) -> List[Dict[str, Any]]:
    if name not in _GENS:
        raise ValueError(f"unknown family {name}")
    return [_item(name, j, seed) for j in range(n)]


def generate(n: int, seed: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        out.append(_item(FAMILIES[i % len(FAMILIES)], i // len(FAMILIES), seed))
    return out
