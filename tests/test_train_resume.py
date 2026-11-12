"""save_state / load_state carry parameters, optimizer, scheduler and RNG,
so an interrupted-and-resumed run continues where it stopped."""
import math

import torch

from kelter.llm_train import load_state, save_state


def _setup():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 3))
    for p in m[0].parameters():
        p.requires_grad_(False)  # frozen base, trainable head, as with LoRA
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / 10)))
    return m, opt, sched


def _steps(m, opt, sched, n):
    g = torch.Generator().manual_seed(1)
    data = [(torch.randn(4, 8, generator=g), torch.randint(0, 3, (4,), generator=g)) for _ in range(10)]
    return data


def _train(m, opt, sched, data, lo, hi):
    m.train()
    for x, y in data[lo:hi]:
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)


def test_resume_matches_uninterrupted(tmp_path):
    m, opt, sched = _setup()
    data = _steps(m, opt, sched, 10)
    torch.manual_seed(5)
    _train(m, opt, sched, data, 0, 10)
    ref = [p.detach().clone() for p in m.parameters()]

    m, opt, sched = _setup()
    torch.manual_seed(5)
    _train(m, opt, sched, data, 0, 4)
    save_state(tmp_path / "state.pt", m, opt, sched, {"next_bi": 4, "step": 4})
    assert not (tmp_path / "state.tmp").exists()

    m2, opt2, sched2 = _setup()  # a fresh process: new model, optimizer and scheduler
    torch.manual_seed(123)        # and a different RNG state, which load_state replaces
    extra = load_state(tmp_path / "state.pt", m2, opt2, sched2)
    assert extra == {"next_bi": 4, "step": 4}
    assert sched2.last_epoch == 4
    _train(m2, opt2, sched2, data, extra["next_bi"], 10)
    for a, b in zip(ref, m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_sentinel_detects_changed_bytes(tmp_path):
    """The sentinel passes on the recorded hash and fails on any changed byte."""
    import hashlib
    from kelter.llm_train import sentinel_check
    f = tmp_path / "model.pt"
    f.write_bytes(bytes(range(256)) * 4096)
    good = hashlib.sha256(f.read_bytes()).hexdigest()
    assert sentinel_check(f, good)
    b = bytearray(f.read_bytes())
    b[123456] ^= 1
    f.write_bytes(bytes(b))
    assert not sentinel_check(f, good)


def test_sentinel_probe_classifies(tmp_path, monkeypatch):
    """On a mismatch the probe compares an unbuffered read and a cached re-read."""
    import hashlib
    import os
    import kelter.llm_train as lt
    f = tmp_path / "model.pt"
    data = bytes(range(256)) * 8191 + b"tail"  # not a multiple of the sector size
    f.write_bytes(data)
    good = hashlib.sha256(data).hexdigest()
    assert lt.sentinel_probe(f, good) == {"cached": good[:12], "ok": True}
    if os.name == "nt":
        assert lt._hash_unbuffered(f) == good
    seq = iter(["bad" * 20, good])
    monkeypatch.setattr(lt, "_hash_cached", lambda p: next(seq))
    monkeypatch.setattr(lt, "_hash_unbuffered", lambda p: good)
    assert lt.sentinel_probe(f, good)["kind"].startswith("stale cache")
    monkeypatch.setattr(lt, "_hash_cached", lambda p: "bad" * 20)
    assert lt.sentinel_probe(f, good)["kind"] == "memory (live flip)"
    monkeypatch.setattr(lt, "_hash_unbuffered", lambda p: "bad" * 20)
    assert lt.sentinel_probe(f, good)["kind"] == "disk"


def test_preflight_pass_clean_and_mismatch(tmp_path, monkeypatch):
    """Preflight reports a file whose cached and disk hashes differ."""
    import kelter.eval.preflight as pf
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 10000)
    assert pf.one_pass([f]) == []
    monkeypatch.setattr(pf, "_hash_cached", lambda p: "a" * 64)
    monkeypatch.setattr(pf, "_hash_unbuffered", lambda p: "b" * 64)
    assert pf.one_pass([f]) == [{"file": str(f), "cached": "a" * 12, "disk": "b" * 12}]
