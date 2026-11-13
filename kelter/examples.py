"""Quick-start examples, sent to a running server and saved with the real responses.

  python -m kelter.server --port 8765          (in another shell; installed: kelter serve)
  python -m kelter.examples --url http://127.0.0.1:8765   (installed: kelter examples)

Writes results/examples/<name>.json under the writable folder (kelter.paths.home: the clone, or a per-user folder for
an installed package) for each example: the request, the full response and the command.
`act` is a support ticket the shipped engine routes and acts on (witness basis: the evidence spans pass the
sufficiency and necessity checks); `defer` is a ticket too vague to route, where it finds no evidence span and defers.
"""
from __future__ import annotations

import argparse
import json
import urllib.request


QUESTION = {"type": "choice",
            "instructions": "Route the support ticket to the team that should handle it.",
            "criteria": {"billing": "Charges, refunds, invoices and payment problems.",
                         "shipping": "Delivery status, lost or late packages.",
                         "account": "Login, password and profile problems.",
                         "technical": "Bugs, crashes and errors in the app.",
                         "other": "Anything that fits none of the teams above."}}


def request(parts):
    """Join (text, provenance type) parts with newlines and return the request body with character spans."""
    state, prov = "", []
    for i, (text, kind) in enumerate(parts):
        if i:
            state += "\n"
        prov.append({"start": len(state), "end": len(state) + len(text), "type": kind})
        state += text
    return {"model": "kelter", "state": state, "provenance": prov, "questions": {"route": QUESTION}}


EXAMPLES = {
    "act": request([("Customer: I forgot my password and cannot log in to my account.", "user")]),
    "defer": request([("Customer: Something is wrong, please help.", "user")]),
}


def post(url, body):
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kelter examples")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    from kelter import paths
    d = paths.home() / "results" / "examples"
    d.mkdir(parents=True, exist_ok=True)
    for name, body in EXAMPLES.items():
        if a.only and name not in a.only:
            continue
        resp = post(a.url, body)
        ans = resp["answers"]["route"]
        out = {"name": name, "request": body, "response": resp,
               "command": "python -m kelter.server --port 8765, then python -m kelter.examples"}
        (d / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(name, ans.get("action"), (ans.get("verifier") or {}).get("basis"), ans.get("choice"))
    print(f"kelter: responses written to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
