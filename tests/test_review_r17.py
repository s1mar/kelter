"""The status screen's GPU memory check, a body nested past the JSON decoder's limit, and the examples timeout. No GPU, no network."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from kelter import cli, download, examples, server
from tests.fakes import FakeEngine


# ---- 1: the status screen compares the free GPU memory with what the default variant needs
def _status(monkeypatch, gpu, have):
    monkeypatch.delenv("KELTER_DEVICE", raising=False)  # CI sets it
    monkeypatch.setattr(server, "_cuda_free_gb", lambda: gpu)
    monkeypatch.setattr(download, "present", lambda m, v, home: v in have)
    return "\n".join(cli.status())


@pytest.mark.parametrize("gpu,usable", [((3.6, 4.0), False), ((8.9, 12.0), False), ((9.0, 12.0), True), (None, False)])
def test_status_default_variant_needs_llm_gpu_gb_free(monkeypatch, gpu, usable):
    assert server.LLM_GPU_GB == 9.0
    # weights not downloaded
    out = _status(monkeypatch, gpu, have=())
    nxt = [l for l in out.splitlines() if l.startswith("Next:")][0]
    assert ("--variant lower-latency" not in nxt) == usable
    # both downloaded
    out = _status(monkeypatch, gpu, have=("default", "lower-latency"))
    nxt = [l for l in out.splitlines() if l.startswith("Next:")][0]
    assert ("--variant lower-latency" not in nxt) == usable
    if gpu and not usable:
        assert f"only {server.gb(gpu[0])} GB free" in nxt and "9.0 GB" in nxt and "KELTER_DEVICE=cpu" in nxt
    if usable:
        assert "Next: " in nxt and "serve" in nxt
    # only the default downloaded
    nxt = [l for l in _status(monkeypatch, gpu, have=("default",)).splitlines() if l.startswith("Next:")][0]
    assert ("--variant lower-latency" in nxt) == (not usable)


# ---- 2: a body nested past the decoder's limit is a 422, not FastAPI's 400
@pytest.mark.parametrize("depth", [3500, 100000])
def test_deeply_nested_body_is_422(depth):
    c = TestClient(server.create_app(FakeEngine()))
    body = '{"model":"m","state":' + "[" * depth + "]" * depth + ',"questions":{}}'
    r = c.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422  # never FastAPI's bare 400, wherever this platform's decoder gives up
    d = r.json()["detail"]
    if depth >= 100000:  # past every platform's decoder limit: the parse fails and the field is unknown
        assert d[0]["loc"] == ["body"] and d[0]["type"] == "value_error" and d[0]["msg"].startswith("nested deeper than 64 levels")


def test_nesting_between_64_and_the_decoder_limit_still_names_the_field():
    c = TestClient(server.create_app(FakeEngine()))
    body = '{"model":"m","state":' + "[" * 200 + "]" * 200 + ',"questions":{"q":{"type":"choice","instructions":"i","criteria":{"a":"1","b":"2"}}}}'
    r = c.post("/v1/systemone", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "state"]


# ---- 3: the examples command waits as long as asked and says what a timeout means
class Slow(BaseHTTPRequestHandler):
    def do_POST(self):
        import time
        time.sleep(2)
        try:
            self.send_response(200)
            self.end_headers()
        except OSError:
            pass

    def log_message(self, *a):
        pass


def test_examples_timeout_message_and_option(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rc = examples.main(["--url", f"http://127.0.0.1:{srv.server_port}", "--timeout", "0.3", "--only", "act"])
    finally:
        srv.shutdown()
    err = capsys.readouterr().err
    assert rc == 1 and "no answer" in err and "--timeout" in err and "may still be working" in err
    assert "run this again" not in err
    assert examples.DEFAULT_TIMEOUT >= 900
    with pytest.raises(SystemExit):
        examples.main(["--timeout", "notanumber"])
