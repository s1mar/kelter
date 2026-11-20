"""One shared GPU memory test for the status screen and the server, connect and read timeouts in the examples command, the --timeout option's accepted values, and usage lines that name the command typed. No GPU, no network."""
import json
import socket
import urllib.error

import pytest

from kelter import cli, download, examples, paths, server


# ---- 1
@pytest.mark.parametrize("free,fits", [(8.94, False), (8.96, False), (8.999, False), (9.0, True), (9.04, True)])
def test_status_and_check_device_agree_on_the_unrounded_figure(tmp_path, monkeypatch, free, fits):
    monkeypatch.delenv("KELTER_DEVICE", raising=False)
    monkeypatch.setattr(server, "_cuda_free_gb", lambda: (free, 12.0))
    monkeypatch.setattr(download, "present", lambda m, v, home: True)
    nxt = [l for l in cli.status() if l.startswith("Next:")][0]
    assert ("--variant lower-latency" not in nxt) == fits
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"llm": {}}))
    if fits:
        server.check_device(str(ck))
    else:
        with pytest.raises(server.MissingWeights) as e:
            server.check_device(str(ck))
        assert f"{server.gb(free)} GB of 12.00 GB is free" in str(e.value) and "9.00 GB of" not in str(e.value) and "9.0 GB" in str(e.value)  # cannot read "9.0 of 9.0"
        assert f"only {server.gb(free)} GB" in nxt


# ---- 2 and 3
def _run(monkeypatch, tmp_path, exc, argv=()):
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))

    def post(url, body, timeout=None):
        raise exc
    monkeypatch.setattr(examples, "post", post)
    return examples.main(["--only", "act", *argv])


def test_connect_timeout_is_not_reached_and_read_timeout_is_still_working(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, tmp_path, urllib.error.URLError(TimeoutError("timed out"))) == 1  # Windows WinError 10060 is one too
    err = capsys.readouterr().err
    assert "was not reached" in err and "may still be working" not in err and "Check the address" in err
    assert _run(monkeypatch, tmp_path, urllib.error.URLError(socket.timeout("timed out"))) == 1
    assert "was not reached" in capsys.readouterr().err
    assert _run(monkeypatch, tmp_path, TimeoutError("read timed out")) == 1
    err = capsys.readouterr().err
    assert "may still be working" in err and "--timeout" in err and "was not reached" not in err and "run this again" not in err
    assert _run(monkeypatch, tmp_path, urllib.error.URLError("refused")) == 1
    assert "could not reach a server" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", "-1", "0", "86401", "1e999", "abc", ""])
def test_timeout_is_validated_when_parsed(bad):
    with pytest.raises(SystemExit) as e:
        examples.main(["--timeout", bad])
    assert e.value.code == 2


def test_valid_timeouts_pass_and_a_bad_address_is_not_a_bad_reply(monkeypatch, tmp_path, capsys):
    assert examples._timeout("0.5") == 0.5 and examples._timeout("86400") == 86400.0
    monkeypatch.setenv("KELTER_HOME", str(tmp_path))
    assert examples.main(["--url", "not-a-url", "--only", "act"]) == 1  # urlopen raises ValueError: unknown url type
    err = capsys.readouterr().err
    assert "cannot be used as the server address" in err and "not a kelter answer" not in err
    monkeypatch.setattr(examples, "post", lambda url, body, timeout=None: {"answers": []})
    assert examples.main(["--only", "act"]) == 1
    assert "not a kelter answer" in capsys.readouterr().err


# ---- 4
@pytest.mark.parametrize("checkout", [True, False])
def test_every_subcommand_usage_starts_with_the_command_typed(monkeypatch, capsys, checkout):
    monkeypatch.setattr(paths, "is_checkout", lambda: checkout)
    for sub, mod in (("serve", server), ("download", download), ("examples", examples)):
        with pytest.raises(SystemExit) as e:
            mod.main(["--help"])
        assert e.value.code == 0
        first = capsys.readouterr().out.split("\n", 1)[0]
        assert first.startswith(f"usage: {paths.command(sub)} "), (sub, first)
    assert paths.command("serve") == ("python -m kelter.server" if checkout else "kelter serve")


@pytest.mark.parametrize("x,shown", [(2.3, "2.30"), (1.15, "1.15"), (4.35, "4.35"), (8.29, "8.29"), (0.29, "0.29"), (0.0, "0.00"),
                                      (9.0, "9.00"), (8.999999999068677, "8.99"), (8.94, "8.94"), (8.96, "8.96"), (24.0, "24.00"),
                                      (1e6, "1000000.00")])
def test_gb_cuts_down_on_the_decimal_representation(x, shown):
    assert server.gb(x) == shown


def test_gb_never_prints_a_refused_figure_as_the_threshold():
    just_under = 9.0 - 1 / 2**30  # 9 GiB minus one byte
    assert not server.gpu_fits(just_under) and float(server.gb(just_under)) < server.LLM_GPU_GB
    assert server.gpu_fits(9.0) and server.gb(9.0) == "9.00"
