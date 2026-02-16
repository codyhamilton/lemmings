#!/bin/bash
# Run e2e tests. Usage: run_e2e.sh [test_name] [--keep] [--clean] [--debug]
#
# Uses -s (show output live). Log level defaults to INFO; use --debug for DEBUG.
#
# Options:
#   --keep     Keep test workspaces on success (default: delete on success, keep on failure)
#   --clean    Remove all test-workspaces and exit
#   --debug    Set LEMMINGS_LOG_LEVEL=DEBUG (default: INFO)
#
# Examples:
#   scripts/run_e2e.sh                    # Run all e2e tests
#   scripts/run_e2e.sh test_hello_world
#   scripts/run_e2e.sh test_hello_world --keep
#   scripts/run_e2e.sh test_hello_world --debug
#   scripts/run_e2e.sh --clean            # Remove all test workspaces

set -e
script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
venv_python="$project_root/.venv/bin/python"
workspaces="$project_root/test-workspaces"

if [ ! -x "$venv_python" ]; then
  echo "Error: venv not found at $project_root/.venv" >&2
  exit 1
fi

# Parse --clean
for arg in "$@"; do
  if [ "$arg" = "--clean" ]; then
    if [ -d "$workspaces" ]; then
      rm -rf "$workspaces"/*
      echo "Removed all test workspaces"
    fi
    exit 0
  fi
done

# Parse --keep, --no-thinking, --debug; filter from pytest args
pytest_args=()
keep=""
for arg in "$@"; do
  if [ "$arg" = "--keep" ]; then
    keep="--keep"
  elif [ "$arg" = "--no-thinking" ]; then
    export LEMMINGS_NO_THINKING=1
  elif [ "$arg" = "--debug" ]; then
    export LEMMINGS_LOG_LEVEL=DEBUG
  else
    pytest_args+=("$arg")
  fi
done

# Default log level to INFO if not already set (e.g. by --debug)
if [ -z "${LEMMINGS_LOG_LEVEL+x}" ]; then
  export LEMMINGS_LOG_LEVEL=INFO
fi

export E2E_RUN=1
cd "$project_root"

# First non-option arg is test name (for targeted run)
test_name=""
remaining=()
for arg in "${pytest_args[@]}"; do
  if [ -n "$arg" ] && [ "${arg#-}" = "$arg" ] && [ -z "$test_name" ]; then
    test_name="$arg"
  else
    remaining+=("$arg")
  fi
done

if [ -n "$test_name" ]; then
  exec "$venv_python" -m pytest "tests/e2e/test_e2e_workflow.py::$test_name" -m e2e -v -rs -s $keep "${remaining[@]}"
else
  exec "$venv_python" -m pytest tests/e2e/ -m e2e -v -rs -s $keep "${remaining[@]}"
fi
