"""Opt-in collection and terminal/JSON/Markdown reports for synthetic mesh tests."""

import json
import math
from pathlib import Path

import pytest

from tests.synthetic_mesh import (
    CORPUS_VERSION,
    EXPECTED_COUNTS,
    METRICS,
    markdown_summary,
    summarize,
)

RESULTS = pytest.StashKey[list]()
WEIGHTS = pytest.StashKey[dict]()


def pytest_addoption(parser):
    group = parser.getgroup("mesh benchmark")
    group.addoption(
        "--run-mesh-benchmark",
        action="store_true",
        help="Enable synthetic mesh quality tests",
    )
    group.addoption(
        "--mesh-report-dir", help="Write mesh benchmark summary.md and results.json"
    )
    group.addoption(
        "--mesh-weights",
        default=str(Path(__file__).with_name("mesh_weights.json")),
        help="JSON mapping of family names to nonnegative weights",
    )


def pytest_configure(config):
    config.stash[RESULTS] = []
    if not config.getoption("--run-mesh-benchmark"):
        return
    try:
        weights = json.loads(Path(config.getoption("--mesh-weights")).read_text())
        if not isinstance(weights, dict) or weights.keys() != EXPECTED_COUNTS.keys():
            raise ValueError(
                "weights must specify exactly the benchmark's nine families"
            )
        if (
            any(
                isinstance(w, bool)
                or not isinstance(w, (int, float))
                or not math.isfinite(w)
                or w < 0
                for w in weights.values()
            )
            or sum(weights.values()) <= 0
        ):
            raise ValueError(
                "weights must be finite, nonnegative numbers with positive total"
            )
    except (OSError, ValueError) as error:
        raise pytest.UsageError(f"Invalid mesh benchmark weights: {error}") from error
    config.stash[WEIGHTS] = weights


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-mesh-benchmark"):
        return
    deselected = [item for item in items if item.get_closest_marker("mesh_benchmark")]
    items[:] = [item for item in items if not item.get_closest_marker("mesh_benchmark")]
    config.hook.pytest_deselected(items=deselected)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not item.get_closest_marker("mesh_benchmark"):
        return
    if report.when != "call" and not (report.when == "setup" and not report.passed):
        return
    result = next(
        (value for key, value in item.user_properties if key == "mesh_result"), None
    )
    if result is None:
        params = item.callspec.params
        result = dict(
            id=params["case_id"],
            family=params["family"],
            expected_width=params["expected_width"],
        )
    result = dict(result, status=report.outcome)
    if report.failed:
        result["failure"] = report.longreprtext
    item.config.stash[RESULTS].append(result)


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    if not config.getoption("--run-mesh-benchmark"):
        return
    directory = config.getoption("--mesh-report-dir")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        results, weights = config.stash[RESULTS], config.stash[WEIGHTS]
        (path / "summary.md").write_text(markdown_summary(results, weights))
        (path / "results.json").write_text(
            json.dumps(
                dict(
                    corpus_version=CORPUS_VERSION,
                    metrics_version=2,
                    exitstatus=int(exitstatus),
                    weights=weights,
                    **summarize(results, weights),
                    metrics={
                        metric: summarize(results, weights, metric)
                        for metric in METRICS
                    },
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.getoption("--run-mesh-benchmark"):
        terminalreporter.write_sep("=", "Synthetic mesh quality (advisory)")
        terminalreporter.write_line(
            markdown_summary(config.stash[RESULTS], config.stash[WEIGHTS])
        )
