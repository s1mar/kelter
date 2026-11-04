# TypeSafe `/v1/systemone` wire contract

Pinned for `openverdict/server.py`. Source keys used below:

- **OAS** = https://api.typesafe.ai/openapi.json (version 0.2.0; the file docs.typesafe.ai renders its API reference from). Fetched raw.
- **API** = https://docs.typesafe.ai/api/system-one.md, `/api/list-models.md`, `/api/errors.md`. Fetched raw.
- **CONF** = https://docs.typesafe.ai/confidence.md. Fetched raw (interactive widgets are stripped from the markdown, formulas survive).
- **PRIM** = https://docs.typesafe.ai/primitives/{choice,score,noul}.md. Fetched raw.
- **SDK** = https://docs.typesafe.ai/sdk/python.md and the `typesafe-sdk` 0.7.5 package itself.
- **MODELS** = https://docs.typesafe.ai/models.md.
- **LITELLM** = https://docs.litellm.ai/docs/pass_through/typesafe. Fetched through the summarizing fetch tool only; it has no JSON examples and defers to the TypeSafe format page, so it adds nothing beyond OAS.
- **JB** = `third_party/jevbench/jevbench/adapters/typesafe.py` and `adapters/base.py` (read in full). This is what our benchmark client sends and parses.
- **KEV** = github.com/jaredpalmer/kev (Apache-2.0 per its README and LICENSE), files `kev/api.py` and `kev/serve.py` read raw. No code was copied; only shapes were compared.

Where sources disagree, docs.typesafe.ai (OAS, API, PRIM, CONF) wins.

## Endpoints

| Route | Notes | Source |
| - | - | - |
| `POST /v1/systemone` | JSON body. Also accepts multipart with images (not supported here). | OAS |
| `GET /v1/models` | `{"models": [{"name", "description", "release_date"}]}`. Not the OpenAI list shape. | OAS, API |
| Auth | `Authorization: Bearer <key>`. We accept and ignore it. | OAS, JB |
| Optional headers `TypeSafe-Danger-Zone-Images`, `TypeSafe-Danger-Zone-Long-Context` | Ignored by us. | OAS, MODELS |
| Response header `x-typesafe-request-id` | Only seen in KEV, not in OAS. We send a random hex id; harmless if unused. | KEV |

## Request

| Field | Type | Required | Meaning | Source |
| - | - | - | - | - |
| `model` | string | yes | Model name or alias (`jev-latest`, `jev-1.14.0`). | OAS |
| `state` | string, object or array | yes | Content all questions refer to. We render non-strings with `openverdict.state.render_state` (indented JSON). | OAS |
| `questions` | object, at least 1 entry | yes | Keyed by a question id the caller chooses. The id is not sent to the model. | OAS, PRIM |
| `questions.<id>.type` | `"noul"`, `"choice"`, `"score"` (also `"box"`, unsupported here) | yes | Discriminator. | OAS |
| `questions.<id>.instructions` | string, object or array (nullable in OAS) | optional in OAS | What to decide. | OAS |
| choice `criteria` | object, 1 to 255 entries, name to description (string, object, array or null) | yes | Option names are what the answer returns. | OAS |
| score `criteria` | array of descriptions (string, object, array), position is the level number from 0 | yes | OAS: 1 to 10 items. PRIM: "at least two levels; the API accepts up to 10". | OAS, PRIM |
| noul `criteria` | `{"true": ..., "false": ...}` or null | optional | What counts as yes and no. | OAS, PRIM |
| `assets` | object | optional | Image assets. We return 422. | OAS |
| `provenance` | list of `{start, end, type}` | extension, optional | Not TypeSafe. Char ranges over the rendered state. | this repo |

## Response (200)

| Field | Type | Meaning | Source |
| - | - | - | - |
| `model` | string | Model that answered. "May differ from the alias supplied in the request." We always return `engine.model_id`. | OAS |
| `answers` | object, keyed by the request question ids | One answer per question, same `type`. | OAS |
| `usage.input_tokens` | integer | Billable input tokens. | OAS |
| `usage.output_tokens` | integer | Output tokens (we return 0). | OAS |

### Choice answer (all four required)

- `type`: `"choice"`.
- `choice`: name of the highest probability option.
- `probabilities`: `{option name: p}`, each 0 to 1, "sum to approximately 1".
- `confidence`: number 0 to 1. Formula (CONF): `(p_max - 1/n) / (1 - 1/n)`, n options. Verified on the page, which also gives the worked case (0.6, 0.3, 0.1) and (0.6, 0.2, 0.2) both giving 0.4, and the n=3 form `(3 p_max - 1) / 2`.

### Score answer (all five required)

- `type`: `"score"`.
- `score`: expected level, `sum_i i * p_i`. May fall between integers (OAS, PRIM: 0*0.0 + 1*0.57 + 2*0.43 = 1.43).
- `probabilities`: keyed by level number as a string `"0"`..`"k-1"`; sums to 1 (PRIM).
- `legend`: level number string to the criteria entry (same value type as sent).
- `confidence`: CONF: `max(0, 1 - sum_i p_i |i - m| / MAD_unif)`, m the most likely level, `MAD_unif = (1/n) sum_i |i - (n-1)/2|`. CONF worked case (0, 0.57, 0.43) gives about 0.35 (PRIM prints 0.35). Both are tested.

### Noul answer (both required)

- `type`: `"noul"`.
- `noul`: P(yes), 0 to 1.
- No `confidence` (CONF and PRIM say so explicitly; they suggest `|2p - 1|` for callers who want one, but it is not a response field). We follow the docs and omit it.

### Extension fields we add to every answer (Jev clients ignore them)

`null_probability`, `witness` (list of `{start, end, provenance, text}`), `verifier`, `set`, `coverage_level`, `action` (`act`, `escalate`, `defer`), `receipt`. Passed through from the engine result. The official Python SDK parses answers carrying these (test `test_typesafe_python_sdk_against_server`).

## Errors

- 422 body (OAS `HTTPValidationError`): `{"detail": [{"loc": [...], "msg": str, "type": str, "input"?, "ctx"?}]}`. We return that shape for every 422, including our own checks (bad provenance, one-level score, missing instructions, assets).
- 400 `{"detail": str | object}` for body-rule breaks (invalid UTF-8, nesting over 64, non-finite numbers, malformed compressed body); 401 bad key; 413 over 32 MiB; 415 bad `Content-Encoding`; 429, 529 (API). **Not implemented**: we never return 401, 413, 415, 429 or 529, and FastAPI does not decode gzip or zstd request bodies. Unparseable JSON gives FastAPI's 422, where TypeSafe gives 400. KEV returns a plain string `detail` for its own 422s; the docs say a list, so we follow the docs.
- Engine failures (bad result shape) return 500 `{"detail": "engine error: ..."}`. Not a TypeSafe status; it never occurs with a healthy engine.

## `GET /v1/models`

OAS `ModelMetadataList`: `{"models": [{"name": str, "description": str, "release_date": "YYYY-MM-DD"}]}`. We return one entry, `engine.model_id`. `release_date` comes from `engine.release_date` if the engine has it, otherwise the date the app was created (the docs give no semantics beyond "release date").

## Our behaviour choices (not in TypeSafe's docs)

- Unknown `model` values are served anyway; the response `model` is our id (the docs allow the answering model to differ from the alias).
- Probabilities are renormalized over the options, so they sum to 1 even when the engine reserves mass for the null. The raw null mass is in `null_probability`.
- Ties: argmax and the score mode pick the first option in criteria order (docs are silent).
- One option: confidence 1.0 (docs are silent; KEV does the same).
- Probabilities are not rounded (KEV rounds to 4 decimals; docs do not require it).
- `instructions` must be non-empty after rendering (OAS lets it be null). The repo's record schema rejects empty instructions, so we return 422 rather than invent text.
- Score needs at least 2 levels (PRIM wording; OAS says minimum 1, KEV allows 1). Max 10 (OAS, PRIM; KEV allows 255).
- Non-string criteria and instruction values are rendered as JSON text for the model; the original values are echoed in `legend`.
- Record schema quirk: `schema.validate` requires a gold unless `null` is true, but requests carry none. The server validates a copy with a placeholder gold and hands the engine the record with `gold` None as specified.

## Disagreements between sources

| Topic | Sources | Followed |
| - | - | - |
| Score levels minimum | PRIM "at least two" vs OAS minItems 1 vs KEV 1 | PRIM (2) |
| Score levels maximum | OAS and PRIM 10 vs KEV 255 | 10 |
| Default model name | SDK and OAS `jev-latest` vs KEV `kev-latest` | not applicable (model is required) |
| Response extras | KEV adds `latency_ms`, `usage.state_tokens`; OAS has none | OAS plus our documented extensions |
| `model` in examples | OAS examples `jev-latest`; PRIM example `jev-1.13.0` | consistent with "may differ from alias" |
| 422 `detail` | OAS list of objects vs KEV plain string | OAS |

## Python SDK

PyPI has the official client: `typesafe-sdk` (0.7.5 at the time of writing; docs link github.com/typesafe-ai/typesafe-sdk-python). Before installing, `pip install --dry-run typesafe-sdk` showed it would add only `tenacity 9.2.1` and `typesafe-sdk 0.7.5`; everything else it needs was already present. After the install, `pip list` still shows torch 2.11.0+cu128, transformers 5.20.2, fastapi 0.162.0, pydantic 2.14.1, httpx 0.28.1. Base URL is `TypeSafeClient(base_url=..., api_key=...)` (or env `TYPESAFE_BASE_URL`, `TYPESAFE_API_KEY`). The SDK keys `probabilities` and `legend` by integer level; the wire stays string-keyed. The test starts a real uvicorn server on a free localhost port in a thread with the FakeEngine and skips itself if the SDK is not installed.

## Not fetched or only partly read

- docs.typesafe.ai HTML pages were not fetched; the `.md` twins listed in `llms.txt` were. The state, images, patterns and cookbook pages were not read. The `state` type comes from OAS.
- LITELLM came only through the summarizing fetch tool.
- KEV's README came through the summarizing fetch tool; its `kev/api.py` and `kev/serve.py` were read raw. `tests/test_api.py` was not read.
