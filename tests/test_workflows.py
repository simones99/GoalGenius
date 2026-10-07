from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / ".github" / "workflows" / name).read_text()


def test_ci_runs_lint_and_tests_on_python_312():
    ci = read("ci.yml")
    for text in ("push", "pull_request", "python-version: \"3.12\"", "ruff check", "pytest"):
        assert text in ci, text


def test_publish_runs_final_not_develop():
    pub = read("publish.yml")
    for text in ("workflow_dispatch", "tags:", "v*", "fetch-depth: 0", "goalline download",
                 "goalline validate", "goalline final", "goalline report", "deploy-pages"):
        assert text in pub, text
    assert "goalline develop" not in pub
