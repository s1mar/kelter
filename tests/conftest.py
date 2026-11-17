"""CI guard: with KELTER_FAIL_ON_SKIP=1 a skipped test is a failure.

The JevBench round trip and the typesafe-sdk test skip when their dependency is missing, which on a developer machine
is fine; in CI both are installed on purpose, so a silent skip there would mean the test no longer runs.
"""
import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("KELTER_FAIL_ON_SKIP") == "1" and report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"skipped in CI, where every test is expected to run: {reason}"
