"""CLI runner for testing individual agents in isolation.

This allows developers to run single agents with test input for debugging
and prompt optimization during development.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.state import WorkflowState
from agents.testing.fixtures import (
    create_test_state,
    create_test_state_from_fixture,
    save_state_fixture,
)
from agents.testing.mock_llm import create_mock_llm, create_mock_llm_from_fixture


# Map agent names to their node functions
AGENT_NODES = {
    "intent": "agents.agents.intent.intent_node",
    "milestone": "agents.agents.milestone.milestone_node",
    "expander": "agents.agents.expander.expander_node",
    "prioritizer": "agents.agents.prioritizer.prioritizer_node",
    "researcher": "agents.agents.researcher.researcher_node",
    "planner": "agents.agents.planner.planner_node",
    "implementor": "agents.agents.implementor.implementor_node",
    "validator": "agents.agents.validator.validator_node",
    "qa": "agents.agents.qa.qa_node",
    "assessor": "agents.agents.assessor.assessor_node",
}


def import_agent_node(agent_name: str):
    """Import and return the agent node function.
    
    Args:
        agent_name: Name of the agent (e.g., "researcher")
    
    Returns:
        Agent node function
    """
    if agent_name not in AGENT_NODES:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(AGENT_NODES.keys())}")
    
    module_path, func_name = AGENT_NODES[agent_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)


def run_agent(
    agent_name: str,
    state: WorkflowState,
    mock_llm: Any | None = None,
    verbose: bool = False,
) -> dict:
    """Run a single agent node with the given state.
    
    Args:
        agent_name: Name of the agent to run
        state: WorkflowState to pass to the agent
        mock_llm: Optional mock LLM to use (patches the agent's LLM)
        verbose: Enable verbose output
    
    Returns:
        State update dict from the agent
    """
    agent_node = import_agent_node(agent_name)
    
    if mock_llm:
        # Determine which LLM to patch based on agent
        llm_imports = {
            "researcher": "agents.agents.researcher.planning_llm",
            "planner": "agents.agents.planner.planning_llm",
            "implementor": "agents.agents.implementor.coding_llm",
            "validator": "agents.agents.validator.planning_llm",
            "qa": "agents.agents.qa.planning_llm",
            "assessor": "agents.agents.assessor.planning_llm",
            "intent": "agents.agents.intent.planning_llm",
            "milestone": "agents.agents.milestone.planning_llm",
            "expander": "agents.agents.expander.planning_llm",
            "prioritizer": "agents.agents.prioritizer.planning_llm",
        }
        
        if agent_name in llm_imports:
            from unittest.mock import patch
            with patch(llm_imports[agent_name], mock_llm):
                return agent_node(state)
    
    return agent_node(state)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run individual agents for testing and debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run researcher with fixture state
  python -m agents.testing.runner researcher --state tests/fixtures/states/researcher_input.json

  # Run with mock LLM
  python -m agents.testing.runner researcher --mock-llm --response tests/fixtures/responses/researcher_gap_exists.json

  # Run with real LLM (for prompt testing)
  python -m agents.testing.runner researcher --state tests/fixtures/states/researcher_input.json --verbose

  # Save current state as fixture
  python -m agents.testing.runner researcher --state input.json --save-output output.json
        """
    )
    
    parser.add_argument(
        "agent",
        choices=list(AGENT_NODES.keys()),
        help="Agent to run",
    )
    
    parser.add_argument(
        "--state",
        type=str,
        help="Path to JSON state fixture file",
    )
    
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Use mock LLM instead of real one",
    )
    
    parser.add_argument(
        "--response",
        type=str,
        help="Path to JSON response fixture for mock LLM",
    )
    
    parser.add_argument(
        "--response-data",
        type=str,
        help="Inline JSON response data for mock LLM",
    )
    
    parser.add_argument(
        "--save-output",
        type=str,
        help="Save output state to JSON file",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Load state
    if args.state:
        if not Path(args.state).exists():
            print(f"Error: State file not found: {args.state}", file=sys.stderr)
            sys.exit(1)
        state = create_test_state_from_fixture(args.state)
    else:
        # Create default test state
        state = create_test_state(verbose=args.verbose)
        print("Warning: No state file provided, using default test state", file=sys.stderr)
    
    # Setup mock LLM if requested
    mock_llm = None
    if args.mock_llm:
        if args.response:
            if not Path(args.response).exists():
                print(f"Error: Response file not found: {args.response}", file=sys.stderr)
                sys.exit(1)
            mock_llm = create_mock_llm_from_fixture(args.response)
        elif args.response_data:
            try:
                response_data = json.loads(args.response_data)
                mock_llm = create_mock_llm([response_data])
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in --response-data: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Warning: --mock-llm specified but no response provided, using empty mock", file=sys.stderr)
            mock_llm = create_mock_llm([{}])
    
    # Run agent
    try:
        print(f"Running {args.agent} agent...")
        if args.verbose:
            print(f"State keys: {list(state.keys())}")
            if state.get("current_task_id"):
                print(f"Current task: {state['current_task_id']}")
        
        result = run_agent(
            agent_name=args.agent,
            state=state,
            mock_llm=mock_llm,
            verbose=args.verbose,
        )
        
        # Print result summary
        print("\n" + "="*70)
        print("AGENT RESULT")
        print("="*70)
        print(f"Result keys: {list(result.keys())}")
        
        # Print key fields
        for key in ["current_gap_analysis", "current_implementation_plan", 
                   "current_implementation_result", "current_validation_result",
                   "current_qa_result", "error", "messages"]:
            if key in result:
                value = result[key]
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        if isinstance(v, str) and len(v) > 100:
                            print(f"  {k}: {v[:100]}...")
                        else:
                            print(f"  {k}: {v}")
                elif isinstance(value, list):
                    print(f"\n{key}: {len(value)} items")
                    for item in value[:5]:
                        print(f"  - {item}")
                else:
                    print(f"\n{key}: {value}")
        
        print("="*70)
        
        # Save output if requested
        if args.save_output:
            # Merge result into state for saving
            output_state = dict(state)
            output_state.update(result)
            save_state_fixture(output_state, args.save_output)
            print(f"\nSaved output state to: {args.save_output}")
        
    except Exception as e:
        print(f"\nError running agent: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
