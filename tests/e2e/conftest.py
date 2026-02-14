"""Fixtures for e2e tests."""

import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import pytest


E2E_WORKSPACES = Path(__file__).resolve().parent.parent.parent / "test-workspaces"


def pytest_addoption(parser):
    parser.addoption(
        "--keep",
        action="store_true",
        help="Keep e2e test workspaces on success (default: delete on success, keep on failure)",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item.rep_call = rep


@pytest.fixture(scope="session", autouse=True)
def verify_llm_reachable():
    """Skip e2e tests with a clear message if the LLM API is not reachable."""
    if not os.environ.get("E2E_RUN"):
        return
    from agents.config import config

    base_url = config["llm"]["base_url"].rstrip("/")
    try:
        urllib.request.urlopen(f"{base_url}/models", timeout=3)
    except Exception as e:
        pytest.skip(
            f"LLM API not reachable at {base_url}. "
            "Start your local LLM (e.g. Tabby) before running e2e tests. "
            f"Override URL with LEMMINGS_LLM_BASE_URL. Error: {e}"
        )


@pytest.fixture
def e2e_project_dir(request):
    """Create a workspace directory in test-workspaces/ for e2e test projects.

    Retained on failure for inspection. Deleted on success unless --keep.
    """
    E2E_WORKSPACES.mkdir(parents=True, exist_ok=True)
    datestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    short_id = uuid.uuid4().hex[:4]
    test_name = request.node.name
    workspace_dir = E2E_WORKSPACES / f"{datestamp}_{test_name}_{short_id}"
    workspace_dir.mkdir(parents=True)

    yield workspace_dir

    rep = getattr(request.node, "rep_call", None)
    passed = rep.passed if rep else False
    keep_flag = request.config.getoption("--keep", False)

    if passed and not keep_flag:
        shutil.rmtree(workspace_dir, ignore_errors=True)


def run_workflow(
    request: str,
    repo_root: str,
    verbose: bool | None = None,
    show_thinking: bool | None = None,
) -> dict:
    """Run the full agent workflow.

    Args:
        request: The user's development request
        repo_root: Path to the project directory
        verbose: Enable verbose output. If None, uses config (LEMMINGS_LOG_LEVEL=DEBUG).
        show_thinking: Stream agent thinking. If None, uses config (LEMMINGS_NO_THINKING).

    Returns:
        Result dict from run_workflow (status, etc.)
    """
    from agents.config import config
    from agents.main import run_workflow as _run_workflow

    if verbose is None:
        verbose = config.get("verbose", False)
    if show_thinking is None:
        show_thinking = config.get("show_thinking", True)

    return _run_workflow(
        user_request=request,
        repo_root=repo_root,
        verbose=verbose,
        show_thinking=show_thinking,
    )


def run_python(project_dir: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a Python script or module in the project directory.

    Args:
        project_dir: Working directory for the subprocess
        *args: Arguments to pass (e.g. "hello.py" or "-m", "math_cli", "3 + 2")

    Returns:
        CompletedProcess with stdout, stderr, returncode
    """
    return subprocess.run(
        [sys.executable] + list(args),
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )
