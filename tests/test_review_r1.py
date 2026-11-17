"""Review fixes: base revision pin, ensemble, download, server limits, preflight, build, resume, canary, requirements, truncation. No GPU, no network."""
import hashlib
import io
import json
import pickle
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---- 1. the LLM base model is loaded at the revision the config records
def test_llm_loads_base_at_recorded_revision(tmp_path, monkeypatch):
    import transformers

    from kelter import llm
    calls = []

    class Tok:
        padding_side = "right"
        pad_token_id = 0

        def encode(self, ch, add_special_tokens=False):
            return [ord(ch)]

    class Model:
        def to(self, d):
            return self

        def eval(self):
            return self

    def tok_from(base, **kw):
        calls.append(("tok", base, kw))
        return Tok()

    def model_from(base, **kw):
        calls.append(("model", base, kw))
        return Model()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(tok_from))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(model_from))
    for cfg, want in (({"llm": {"base": "o/m"}, "base_revision": "abc123"}, {"revision": "abc123"}),
                      ({"llm": {"base": "o/m"}}, {})):
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        calls.clear()
        llm.LLMPredictor(str(tmp_path), device="cpu", witness=False)
        assert [c[0] for c in calls] == ["tok", "model"]
        assert all(c[2].get("revision") == want.get("revision") for c in calls)
        assert "revision" not in calls[0][2] or calls[0][2]["revision"] == want["revision"]


# ---- 2. ensemble: one read of the stacker, and no options member with a null head
class _StubPredictor:
    def __init__(self, path, device=None):
        self.tok, self.feat = object(), object()

    def predict(self, recs, canonical=True, temperature=True):
        return [{"probs": {"a": 0.5, "b": 0.5}, "null": 0.1, "option_probs": {"a": 0.5, "b": 0.5},
                 "null_threshold": 0.5} for _ in recs]


def _ensemble(tmp_path, monkeypatch, blob, sha):
    from kelter import ensemble, model, paths
    spec = {"name": "T", "options": "m", "null": ["m"], "refs": {"m": [0.1, 0.2]}, "logistic": {"a": 1.0, "b": 0.0},
            "stacker": "stk.pkl", "stacker_sha256": sha}
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "stk.pkl").write_bytes(blob)
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))
    monkeypatch.setattr(paths, "asset", lambda rel: tmp_path / "spec.json")
    monkeypatch.setattr(model, "Predictor", _StubPredictor)
    return ensemble


def test_ensemble_hashes_and_unpickles_the_same_bytes(tmp_path, monkeypatch):
    blob = pickle.dumps({"ok": 1})
    ens = _ensemble(tmp_path, monkeypatch, blob, hashlib.sha256(blob).hexdigest())
    reads = []
    orig = Path.read_bytes

    def counting(self):
        reads.append(self.name)
        return orig(self)
    monkeypatch.setattr(Path, "read_bytes", counting)
    e = ens.Ensemble("T", device="cpu")
    assert e.stacker == {"ok": 1} and reads.count("stk.pkl") == 1


def test_ensemble_rejects_a_hash_mismatch_and_a_null_head_member(tmp_path, monkeypatch):
    blob = pickle.dumps({"ok": 1})
    ens = _ensemble(tmp_path, monkeypatch, blob, "0" * 64)
    with pytest.raises(ValueError, match="stacker_sha256"):
        ens.Ensemble("T", device="cpu")
    ens = _ensemble(tmp_path, monkeypatch, blob, hashlib.sha256(blob).hexdigest())
    e = ens.Ensemble("T", device="cpu")
    with pytest.raises(ValueError, match="null head"):
        e.predict([{"state": "x"}])


# ---- 3. download: a checkpoint folder appears only after a complete extraction; the network read has a timeout
def _tar(name, extra=b""):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for fname, data in (("config.json", b"{}"), ("model.pt", extra)):
            info = tarfile.TarInfo(f"{name}/{fname}")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_extracts_into_a_temporary_folder(tmp_path, monkeypatch):
    from kelter import download
    blob = _tar("r2", b"w" * 10)
    manifest = {"tag": "v0", "repo": "o/r", "variants": {"default": ["r2"]},
                "assets": {"r2": {"file": "r2.tar", "sha256": hashlib.sha256(blob).hexdigest(), "size_mb": 0}}}
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))
    monkeypatch.setattr(download, "load_manifest", lambda: manifest)
    arch = tmp_path / "weights" / "archives"
    arch.mkdir(parents=True)
    (arch / "r2.tar").write_bytes(blob)
    seen = {}
    real = tarfile.TarFile.extractall

    def failing(self, path=".", *a, **k):  # writes config.json, then dies before the rest
        real(self, path, *a, **k)
        seen["config_visible"] = (Path(path) / "r2" / "config.json").exists()
        seen["final_visible"] = (tmp_path / "checkpoints" / "r2" / "config.json").exists()
        raise OSError("disk full")
    monkeypatch.setattr(tarfile.TarFile, "extractall", failing)
    with pytest.raises(OSError):
        download.main([])
    assert seen == {"config_visible": True, "final_visible": False}
    assert not (tmp_path / "checkpoints" / "r2" / "config.json").exists()
    monkeypatch.setattr(tarfile.TarFile, "extractall", real)
    assert download.main([]) == 0
    assert (tmp_path / "checkpoints" / "r2" / "model.pt").read_bytes() == b"w" * 10
    assert not list((tmp_path / "checkpoints").glob(".*"))


def test_download_fetch_has_a_timeout(tmp_path, monkeypatch):
    from kelter import download
    got = {}

    def fake_urlopen(url, timeout=None):
        got["timeout"] = timeout
        raise OSError("stop")
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(OSError):
        download.fetch("http://x/y", tmp_path / "f")
    assert got["timeout"] == download.TIMEOUT_S > 0


# ---- 4. server: an unknown variant is one line; request limits give 422
def test_unknown_variant_from_the_environment_is_one_line(monkeypatch):
    from kelter import server
    monkeypatch.setenv("KELTER_VARIANT", "bogus")
    monkeypatch.delenv("KELTER_CKPT", raising=False)
    with pytest.raises(SystemExit) as e:
        server.main([])
    msg = str(e.value.code)
    assert "unknown variant" in msg and "\n" not in msg and msg.startswith("kelter: ")


def test_request_limits_give_422():
    from fastapi.testclient import TestClient

    from kelter import server
    from tests.fakes import FakeEngine
    c = TestClient(server.create_app(FakeEngine()))
    q = {"type": "noul", "instructions": "Is this about billing?"}

    def post(**kw):
        body = {"model": "m", "state": "hello world", "questions": {"q": q}}
        body.update(kw)
        return c.post("/v1/systemone", json=body)
    assert post().status_code == 200
    assert post(questions={f"q{i}": q for i in range(server.MAX_QUESTIONS + 1)}).status_code == 422
    r = post(state="x" * (server.MAX_STATE_CHARS + 1))
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "state"]
    # non-overlapping spans, so only the count can be what is wrong
    n = server.MAX_PROVENANCE_SPANS
    state = "x" * (n + 1)
    spans = lambda k: [{"start": i, "end": i + 1, "type": "user"} for i in range(k)]
    assert post(state=state, provenance=spans(n)).status_code == 200
    r = post(state=state, provenance=spans(n + 1))
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "provenance"]
    assert "at most" in r.json()["detail"][0]["msg"]
    assert "Request limits" in (ROOT / "docs" / "typesafe-contract.md").read_text(encoding="utf-8")


# ---- 5. preflight: not checked is not clean
def test_preflight_not_checked_and_no_files(tmp_path, monkeypatch, capsys):
    import kelter.eval.preflight as pf
    f = tmp_path / "f.bin"
    f.write_bytes(b"x")
    monkeypatch.setattr(pf, "_hash_unbuffered", lambda p: None)
    with pytest.raises(pf.NotChecked):
        pf.one_pass([f])
    monkeypatch.setattr(pf, "collect", lambda paths, base: [f])
    assert pf.main(["--tag", "t_notchecked", "--base", ""]) == 1
    assert "not checked" in capsys.readouterr().out
    assert not (ROOT / "results" / "eval" / "preflight" / "t_notchecked.json").exists()
    monkeypatch.setattr(pf, "collect", lambda paths, base: [])
    assert pf.main(["--tag", "t_nofiles", "--base", ""]) == 1
    assert "no files" in capsys.readouterr().out


# ---- 7. build refuses to overwrite data/built
def test_build_refuses_small_or_other_recipe_without_out(capsys):
    from kelter.data import build
    assert build.main(["--small"]) == 2
    assert "would overwrite data/built" in capsys.readouterr().err
    assert build.main(["--recipe", "r5"]) == 2
    assert "--out" in capsys.readouterr().err


# ---- 8. llm_train: --resume without a state file
def test_llm_train_resume_without_state_exits_with_one_line(capsys):
    from kelter import llm_train
    with pytest.raises(SystemExit) as e:
        llm_train.main(["--name", "no_such_run_r1", "--resume"])
    msg = str(e.value.code)
    assert "--resume" in msg and "\n" not in msg and "state.pt" in msg
    assert not (ROOT / "notes" / "train_no_such_run_r1.log").exists()


# ---- 9. canary: NaN fails
def test_canary_nan_fails_the_comparison():
    from kelter.eval.canary import compare_rows
    row = lambda p: {"probs": {"a": p, "b": 1 - p}, "argmax": "a"}
    ok = compare_rows([row(0.9), row(0.9)], [row(0.9), row(0.9)], 2, 0.005)
    assert ok["pass"] and not ok["has_nan"]
    nan_last = compare_rows([row(0.9), row(float("nan"))], [row(0.9), row(0.9)], 2, 0.005)
    assert not nan_last["pass"] and nan_last["has_nan"]  # max() alone would skip this NaN
    assert not compare_rows([row(float("nan")), row(0.9)], [row(0.9), row(0.9)], 2, 0.005)["pass"]


# ---- 10. requirements
def test_requirements_is_what_ci_installs():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "laya" not in [l.split("==")[0] for l in req.splitlines() if "==" in l]
    assert "laya==" in (ROOT / "requirements-laya.txt").read_text(encoding="utf-8")
    wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in wf and "grep -v" not in wf
    assert "--no-deps -r requirements-laya.txt" in wf


# ---- 11. truncation is surfaced
def test_llm_prompt_reports_truncation():
    from kelter import llm

    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids):
            return "".join(chr(i) for i in ids)
    rec = {"state": "s" * 400, "question": {"type": "choice", "instructions": "i"},
           "options": [{"name": "a", "description": "A"}, {"name": "b", "description": "B"}]}
    info = {}
    short = llm.build_prompt(Tok(), dict(rec, state="short"), [0, 1], 10_000, info)
    assert info == {} and short
    ids = llm.build_prompt(Tok(), rec, [0, 1], 300, info)
    assert info == {"truncated": True} and len(ids) <= 300


def test_engine_reports_truncated_from_any_pass():
    from kelter.engine import Engine
    from tests.test_engine import APPROVAL, ScriptedPredictor, make_rec

    class Trunc(ScriptedPredictor):
        def __init__(self, flag):
            super().__init__()
            self.flag = flag

        def predict(self, recs, canonical=True):
            return [dict(o, truncated=self.flag) for o in super().predict(recs, canonical)]
    rec = make_rec([("Run it. ", "user"), (APPROVAL, "tool_output")])
    for flag in (True, False):
        res = Engine(Trunc(flag)).decide([rec])[0]
        assert res["verifier"]["truncated"] is flag
    plain = Engine(ScriptedPredictor()).decide([rec])[0]
    assert plain["verifier"]["truncated"] is False


# ---- round 2
def _client(engine=None, variant=None):
    from fastapi.testclient import TestClient

    from kelter import server
    from tests.fakes import FakeEngine
    return TestClient(server.create_app(engine or FakeEngine(), variant))


CHOICE = {"type": "choice", "instructions": "What is the tone?", "criteria": {"angry": "upset", "calm": "neutral"}}


def test_verdict_class_values_and_keys_are_checked():
    c = _client()

    def post(vc):
        return c.post("/v1/systemone", json={"model": "m", "state": "hello", "questions": {"q": CHOICE},
                                             "verdict_classes": vc})
    assert post({"q": {"angry": "destructive"}}).status_code == 200
    r = post({"q": {"angry": "Destructive"}})  # a typo would silently fall back to "default"
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "verdict_classes", "q", "angry"]
    assert "unknown verdict class" in r.json()["detail"][0]["msg"]
    r = post({"nope": {"angry": "destructive"}})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "verdict_classes", "nope"]


def test_option_limit_is_a_422_naming_the_limit():
    from tests.fakes import FakeEngine
    eng = FakeEngine()
    eng.max_options = 2
    c = _client(eng, "default")
    q = {"type": "choice", "instructions": "x", "criteria": {"a": "1", "b": "2", "c": "3"}}
    r = c.post("/v1/systemone", json={"model": "m", "state": "hello", "questions": {"q": q}})
    d = r.json()["detail"][0]
    assert r.status_code == 422 and d["loc"] == ["body", "questions", "q", "criteria"]
    assert "at most 2" in d["msg"] and "default" in d["msg"]
    from kelter.llm import LETTERS, LLMPredictor
    assert LLMPredictor.max_options == len(LETTERS) - 1 == 25
    assert "25" in (ROOT / "docs" / "typesafe-contract.md").read_text(encoding="utf-8")


def test_llm_truncated_includes_the_witness_encoders_cut():
    import torch

    from kelter import llm

    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids):
            return "".join(chr(i) for i in ids)
    p = llm.LLMPredictor.__new__(llm.LLMPredictor)
    p.tok, p.max_tokens, p.temps = Tok(), 10_000, {}
    p.logits_for = lambda prompts: [torch.zeros(26) for _ in prompts]
    rec = {"state": "short", "question": {"type": "choice", "instructions": "i"},
           "options": [{"name": "a", "description": "A"}, {"name": "b", "description": "B"}]}
    for wit_flag, want in ((True, True), (False, False)):
        p.witness_pred = types.SimpleNamespace(predict=lambda recs, canonical=True, f=wit_flag:
                                               [{"witness": [], "truncated": f} for _ in recs])
        assert p.predict([rec])[0]["truncated"] is want
    p.witness_pred = None
    assert p.predict([rec])[0]["truncated"] is False


def test_build_prompt_with_a_state_under_the_floor_is_refused_not_cut():
    from kelter import llm

    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return msgs[0]["content"]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids):
            return "".join(chr(i) for i in ids)
    rec = {"state": "ABCDEFGHIJ", "question": {"type": "choice", "instructions": "i"},
           "options": [{"name": "a", "description": "A"}, {"name": "b", "description": "B"}]}
    from kelter.data.schema import QuestionTooLong
    full = llm.build_prompt(Tok(), rec, [0, 1], 10_000)
    # five tokens short: the question and options need len(full) - 10 tokens, so a state of 10 tokens cannot fit. An older
    # version cut the prompt from the left (header, instructions and the first option lines lost); now it is refused.
    with pytest.raises(QuestionTooLong):
        llm.build_prompt(Tok(), rec, [0, 1], len(full) - 5)
    assert llm.build_prompt(Tok(), rec, [0, 1], len(full)) == full  # exactly fitting is untouched


def test_llm_train_resolves_the_loaded_base_revision(monkeypatch):
    from kelter import llm_train
    N = types.SimpleNamespace
    assert llm_train.resolved_revision(N(config=N(_commit_hash="abc")), "x/y") == "abc"
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda base, local_files_only=False: "/cache/snapshots/def456")
    assert llm_train.resolved_revision(N(config=N()), "x/y") == "def456"

    def none(*a, **k):
        raise OSError("not cached")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", none)
    assert llm_train.resolved_revision(N(config=N()), "x/y") is None


def test_preflight_missing_path_or_base_is_an_error(monkeypatch, capsys):
    import kelter.eval.preflight as pf
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pf.collect(["checkpoints/no_such_checkpoint"], None)
    assert pf.main(["--tag", "t_missing_path", "--paths", "checkpoints/no_such_checkpoint", "--base", ""]) == 1
    assert "does not exist" in capsys.readouterr().out
    monkeypatch.setattr(pf, "base_files", lambda repo: [])
    with pytest.raises(FileNotFoundError, match="not in the Hugging Face cache"):
        pf.collect([], "no/such-repo")
    assert not (ROOT / "results" / "eval" / "preflight" / "t_missing_path.json").exists()


def test_checkpoint_present_means_complete(tmp_path, monkeypatch):
    from kelter import download, paths
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))
    ck = tmp_path / "checkpoints"
    (ck / "enc").mkdir(parents=True)
    (ck / "enc" / "config.json").write_text(json.dumps({"backbone": "b"}))  # an interrupted unpack: config alone
    (ck / "llm").mkdir()
    (ck / "llm" / "config.json").write_text(json.dumps({"llm": {}}))
    assert paths.checkpoint_missing(ck / "enc") == ["model.pt"] and paths.checkpoint_missing(ck / "llm") == ["adapter"]
    manifest = {"variants": {"v": ["enc", "llm"]}}
    assert download.present(manifest, "v", tmp_path) is False
    (ck / "enc" / "model.pt").write_bytes(b"w")
    (ck / "llm" / "adapter").mkdir()
    (ck / "llm" / "adapter" / "adapter_model.safetensors").write_bytes(b"w")
    assert download.present(manifest, "v", tmp_path) is True
    from kelter import server
    assert paths.checkpoint_missing(tmp_path / "nothing") == ["config.json"]
    with pytest.raises(server.MissingWeights):
        (ck / "enc" / "model.pt").unlink()
        server.check_weights(str(ck / "enc"))
