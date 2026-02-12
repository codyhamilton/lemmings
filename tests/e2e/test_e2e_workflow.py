"""End-to-end tests for the full agent workflow.

These tests run the complete graph (Intent → Milestone → Expander → ... → Assessor)
against isolated temp projects. They require a real LLM API and are skipped by default.

Run with: pytest tests/e2e/ -m e2e -v
"""

import os

import pytest

from .conftest import e2e_project_dir, run_python, run_workflow


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.environ.get("E2E_RUN"),
    reason="Set E2E_RUN=1 to run e2e tests (require LLM API)",
)
def test_hello_world(e2e_project_dir):
    """Create a hello world Python script and validate output.

    Exercises: Full graph flow, minimal expansion, implementor creates file.
    """
    result = run_workflow(
        "Create a Python script named hello.py that prints 'Hello, World!' when run.",
        str(e2e_project_dir),
    )

    assert result.get("status") == "completed", f"Workflow failed: {result.get('error', 'unknown')}"

    proc = run_python(e2e_project_dir, "hello.py")
    assert proc.returncode == 0, f"Script failed: {proc.stderr}"
    assert "Hello" in proc.stdout or "hello" in proc.stdout.lower()


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.environ.get("E2E_RUN"),
    reason="Set E2E_RUN=1 to run e2e tests (require LLM API)",
)
def test_math_cli(e2e_project_dir):
    """Create a Python math CLI module and validate arithmetic.

    Exercises: Intent parsing, milestone breakdown, expander, abstract reasoning,
    planning, implementation.
    """
    result = run_workflow(
        "Create a Python module named math_cli that provides a CLI for basic arithmetic. "
        "When run as `python -m math_cli '3 + 2'` it should output 5. Support +, -, *, /.",
        str(e2e_project_dir),
    )

    assert result.get("status") == "completed", f"Workflow failed: {result.get('error', 'unknown')}"

    proc = run_python(e2e_project_dir, "-m", "math_cli", "3 + 2")
    assert proc.returncode == 0, f"Script failed: {proc.stderr}"
    assert "5" in proc.stdout
