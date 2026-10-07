from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = "pip install -r requirements-lock.txt && pip install -e . --no-deps"


def read(name):
    return (ROOT / ".github" / "workflows" / name).read_text()


def test_ci_runs_lint_and_tests_on_python_313():
    ci = read("ci.yml")
    for text in ("push", "pull_request", "python-version: \"3.13\"", "ruff check", "pytest"):
        assert text in ci, text


def test_both_workflows_install_from_the_lock_file():
    for name in ("ci.yml", "publish.yml"):
        text = read(name)
        assert INSTALL in text, name
        assert "cache-dependency-path: requirements-lock.txt" in text, name
        assert "python-version: \"3.13\"" in text, name


def test_environment_is_pinned_by_one_lock_file():
    assert (ROOT / "requirements-lock.txt").exists()
    assert not (ROOT / "requirements.txt").exists()
    assert not (ROOT / "uv.lock").exists()
    lock = (ROOT / "requirements-lock.txt").read_text().splitlines()
    pins = [line for line in lock if line and not line.startswith("#")]
    assert pins and all("==" in line for line in pins)
    assert not any(line.lower().startswith("goal-line-calibration") for line in pins)


def test_ci_token_is_read_only():
    ci = read("ci.yml")
    assert "permissions:\n  contents: read" in ci


def test_publish_runs_final_not_develop():
    pub = read("publish.yml")
    for text in ("workflow_dispatch", "tags:", "v*", "fetch-depth: 0", "goalline download",
                 "goalline validate", "goalline final", "goalline report", "deploy-pages"):
        assert text in pub, text
    assert "goalline develop" not in pub


def test_publish_uploads_the_output_directory():
    pub = read("publish.yml")
    assert "actions/upload-artifact" in pub
    assert "path: output" in pub
    assert pub.index("goalline report") < pub.index("actions/upload-artifact")


def test_publish_pages_permissions_only_on_the_deploy_job():
    pub = read("publish.yml")
    top, deploy = pub.split("\n  deploy:\n")
    assert "pages: write" not in top and "id-token: write" not in top
    assert "pages: write" in deploy and "id-token: write" in deploy
    assert top.count("contents: read") >= 1
