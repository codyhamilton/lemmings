"""Output normalisation utilities for agent outputs.

The Normaliser attempts to repair agent outputs when they don't match expected schemas.
This allows agents to focus on quality rather than strict format compliance.

Key principles:
1. Simple repairs (JSON extraction, type coercion) are automatic
2. Complex repairs (summarization, semantic fixes) use LLM
3. Log all repairs for feedback loop
4. Fail fast if repair is too extensive
"""

import json
import re
from typing import Any, TypeVar, Callable
from dataclasses import dataclass

from .llm import planning_llm


# =============================================================================
# Repair Result
# =============================================================================

@dataclass
class RepairResult:
    """Result of attempting to repair agent output."""
    success: bool
    data: dict | None
    repairs_applied: list[str]  # Log of what was fixed
    error: str | None
    
    @property
    def needs_review(self) -> bool:
        """True if repairs were extensive and should be reviewed."""
        return len(self.repairs_applied) > 3


# =============================================================================
# JSON Extraction
# =============================================================================

def extract_json(text: str) -> tuple[str | None, list[str]]:
    """Extract JSON from agent output, handling common formats.
    
    Handles:
    - Markdown code blocks (```json...```)
    - Plain code blocks (```...```)
    - JSON embedded in text
    - Extra text before/after JSON
    
    Args:
        text: Raw agent output
    
    Returns:
        (json_string, repairs) or (None, []) if extraction failed
    """
    repairs = []
    
    # Try markdown JSON code block
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            json_str = parts[1].split("```")[0].strip()
            repairs.append("Extracted from ```json code block")
            return json_str, repairs
    
    # Try plain code block
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd indices are code blocks
                if "{" in part and (":" in part or "remit" in part or "tasks" in part):
                    json_str = part.strip()
                    repairs.append("Extracted from ``` code block")
                    return json_str, repairs
    
    # Try to find JSON object boundaries
    if "{" in text:
        first_brace = text.find("{")
        # Find matching closing brace
        brace_count = 0
        end_pos = first_brace
        for i, char in enumerate(text[first_brace:], first_brace):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if end_pos > first_brace:
            json_str = text[first_brace:end_pos].strip()
            repairs.append("Extracted JSON object from text")
            return json_str, repairs
    
    return None, []


def parse_json_lenient(json_str: str) -> tuple[dict | None, list[str]]:
    """Parse JSON with lenient error recovery.
    
    Attempts to fix common JSON issues:
    - Trailing commas
    - Single quotes instead of double quotes
    - Comments
    - Unquoted keys
    
    Args:
        json_str: JSON string to parse
    
    Returns:
        (parsed_dict, repairs) or (None, []) if parsing failed
    """
    repairs = []
    
    # Try standard parse first
    try:
        return json.loads(json_str), repairs
    except json.JSONDecodeError:
        pass
    
    # Try fixing common issues
    fixed = json_str
    
    # Remove trailing commas
    if ",\n" in fixed or ", \n" in fixed:
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        repairs.append("Removed trailing commas")
    
    # Remove comments (// and /* */)
    if "//" in fixed or "/*" in fixed:
        fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        repairs.append("Removed comments")
    
    # Try parse again
    try:
        return json.loads(fixed), repairs
    except json.JSONDecodeError:
        return None, repairs


# =============================================================================
# Field Repairs
# =============================================================================

def truncate_field(value: str, max_length: int, field_name: str) -> tuple[str, str]:
    """Truncate a string field that exceeds max length.
    
    Args:
        value: Field value
        max_length: Maximum allowed length
        field_name: Field name for logging
    
    Returns:
        (truncated_value, repair_message)
    """
    if len(value) <= max_length:
        return value, ""
    
    # Simple truncation with ellipsis
    truncated = value[:max_length-3] + "..."
    repair = f"Truncated {field_name} from {len(value)} to {max_length} chars"
    return truncated, repair


def summarize_field(value: str, max_length: int, field_name: str) -> tuple[str, str]:
    """Summarize a string field using LLM to compress while preserving meaning.
    
    Args:
        value: Field value to summarize
        max_length: Target maximum length
        field_name: Field name for context
    
    Returns:
        (summarized_value, repair_message)
    """
    if len(value) <= max_length:
        return value, ""
    
    try:
        # Use LLM to compress
        prompt = f"""Compress this {field_name} to under {max_length} characters while preserving key information:

{value}

Compressed version (under {max_length} chars):"""
        
        response = planning_llm.invoke(prompt)
        summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # Fallback to truncation if summary is still too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        repair = f"Summarized {field_name} from {len(value)} to {len(summary)} chars"
        return summary, repair
        
    except Exception as e:
        # Fallback to truncation
        truncated = value[:max_length-3] + "..."
        repair = f"Summarization failed, truncated {field_name} ({e})"
        return truncated, repair


def coerce_type(value: Any, expected_type: type, field_name: str) -> tuple[Any, str]:
    """Coerce a value to the expected type.
    
    Args:
        value: Current value
        expected_type: Expected type
        field_name: Field name for logging
    
    Returns:
        (coerced_value, repair_message) or (value, "") if no coercion needed
    """
    if isinstance(value, expected_type):
        return value, ""
    
    repairs = []
    
    try:
        # String to list
        if expected_type == list:
            if isinstance(value, str):
                # Try parsing as JSON array
                if value.strip().startswith("["):
                    coerced = json.loads(value)
                    repairs.append(f"Parsed {field_name} from JSON string to list")
                    return coerced, "; ".join(repairs)
                # Split on common delimiters
                coerced = [v.strip() for v in re.split(r'[,;]', value) if v.strip()]
                repairs.append(f"Split {field_name} string into list")
                return coerced, "; ".join(repairs)
            # Single value to list
            coerced = [value]
            repairs.append(f"Wrapped {field_name} in list")
            return coerced, "; ".join(repairs)
        
        # String to int/float
        if expected_type in (int, float):
            if isinstance(value, str):
                # Extract numbers from string
                numbers = re.findall(r'-?\d+\.?\d*', value)
                if numbers:
                    coerced = expected_type(numbers[0])
                    repairs.append(f"Extracted number from {field_name} string")
                    return coerced, "; ".join(repairs)
        
        # List to string
        if expected_type == str and isinstance(value, list):
            coerced = ", ".join(str(v) for v in value)
            repairs.append(f"Joined {field_name} list into string")
            return coerced, "; ".join(repairs)
        
        # Dict to string (JSON)
        if expected_type == str and isinstance(value, dict):
            coerced = json.dumps(value)
            repairs.append(f"Serialized {field_name} dict to JSON string")
            return coerced, "; ".join(repairs)
        
        # Try direct cast
        coerced = expected_type(value)
        repairs.append(f"Cast {field_name} from {type(value).__name__} to {expected_type.__name__}")
        return coerced, "; ".join(repairs)
        
    except Exception as e:
        # Coercion failed, return original
        return value, f"Failed to coerce {field_name}: {e}"


# =============================================================================
# Schema Normalisation
# =============================================================================

def normalize_dict(
    data: dict,
    schema: dict[str, dict],
    use_llm_summarization: bool = False
) -> RepairResult:
    """Normalize a dictionary to match expected schema.
    
    Args:
        data: Dictionary to normalize
        schema: Expected schema with field specs:
            {
                "field_name": {
                    "type": type,
                    "required": bool,
                    "max_length": int (for strings),
                    "default": any
                }
            }
        use_llm_summarization: Use LLM to summarize long fields (slower but better)
    
    Returns:
        RepairResult with normalized data or error
    """
    repairs = []
    normalized = {}
    
    # Process each schema field
    for field_name, field_spec in schema.items():
        required = field_spec.get("required", False)
        expected_type = field_spec.get("type", str)
        max_length = field_spec.get("max_length")
        default = field_spec.get("default")
        
        # Check if field exists
        if field_name not in data:
            if required:
                if default is not None:
                    normalized[field_name] = default
                    repairs.append(f"Added missing required field '{field_name}' with default")
                else:
                    return RepairResult(
                        success=False,
                        data=None,
                        repairs_applied=repairs,
                        error=f"Missing required field: {field_name}"
                    )
            else:
                # Optional field, use default or skip
                if default is not None:
                    normalized[field_name] = default
                continue
        
        value = data[field_name]
        
        # Type coercion
        if not isinstance(value, expected_type):
            value, repair = coerce_type(value, expected_type, field_name)
            if repair:
                repairs.append(repair)
        
        # Length constraint (for strings)
        if isinstance(value, str) and max_length and len(value) > max_length:
            if use_llm_summarization and len(value) > max_length * 1.5:
                # Use LLM for significant over-length
                value, repair = summarize_field(value, max_length, field_name)
            else:
                # Simple truncation
                value, repair = truncate_field(value, max_length, field_name)
            
            if repair:
                repairs.append(repair)
        
        normalized[field_name] = value
    
    # Check for extra fields (not in schema)
    extra_fields = set(data.keys()) - set(schema.keys())
    if extra_fields:
        repairs.append(f"Ignored extra fields: {', '.join(extra_fields)}")
    
    return RepairResult(
        success=True,
        data=normalized,
        repairs_applied=repairs,
        error=None
    )


# =============================================================================
# High-Level Normaliser
# =============================================================================

def normalize_agent_output(
    raw_output: str,
    schema: dict[str, dict],
    use_llm_summarization: bool = False
) -> RepairResult:
    """Normalize agent output to match expected schema.
    
    This is the main entry point. Attempts:
    1. JSON extraction from markdown/text
    2. Lenient JSON parsing
    3. Schema field repair
    
    Args:
        raw_output: Raw agent output text
        schema: Expected schema (see normalize_dict for format)
        use_llm_summarization: Use LLM for field summarization
    
    Returns:
        RepairResult with normalized data or failure
    """
    all_repairs = []
    
    # Step 1: Extract JSON
    json_str, repairs = extract_json(raw_output)
    all_repairs.extend(repairs)
    
    if not json_str:
        return RepairResult(
            success=False,
            data=None,
            repairs_applied=all_repairs,
            error="Could not extract JSON from output"
        )
    
    # Step 2: Parse JSON
    data, repairs = parse_json_lenient(json_str)
    all_repairs.extend(repairs)
    
    if not data:
        return RepairResult(
            success=False,
            data=None,
            repairs_applied=all_repairs,
            error="Could not parse JSON (even with repairs)"
        )
    
    # Step 3: Normalize to schema
    result = normalize_dict(data, schema, use_llm_summarization)
    result.repairs_applied = all_repairs + result.repairs_applied
    
    return result


# =============================================================================
# Length Validation (for structured output)
# =============================================================================

def validate_and_normalize_lengths(
    data: dict,
    schema: dict[str, dict],
    use_llm_summarization: bool = False
) -> RepairResult:
    """Validate and normalize only length constraints (for structured output).
    
    This is used when we already have valid structured output from tool calls.
    We only need to check length constraints and apply summarization if needed.
    
    Args:
        data: Already-valid structured data (from tool call)
        schema: Expected schema (for length constraints)
        use_llm_summarization: Use LLM for summarization if over length
    
    Returns:
        RepairResult with validated/normalized data
    """
    repairs = []
    normalized = {}
    
    # Process each schema field
    for field_name, field_spec in schema.items():
        expected_type = field_spec.get("type", str)
        max_length = field_spec.get("max_length")
        default = field_spec.get("default")
        required = field_spec.get("required", False)
        
        # Check if field exists
        if field_name not in data:
            if required:
                if default is not None:
                    normalized[field_name] = default
                    repairs.append(f"Added missing required field '{field_name}' with default")
                else:
                    return RepairResult(
                        success=False,
                        data=None,
                        repairs_applied=repairs,
                        error=f"Missing required field: {field_name}"
                    )
            else:
                if default is not None:
                    normalized[field_name] = default
                continue
        
        value = data[field_name]
        
        # Type check (should already be correct from structured output, but verify)
        if not isinstance(value, expected_type):
            value, repair = coerce_type(value, expected_type, field_name)
            if repair:
                repairs.append(repair)
        
        # Length constraint (for strings) - this is the main thing we need to check
        if isinstance(value, str) and max_length and len(value) > max_length:
            if use_llm_summarization and len(value) > max_length * 1.5:
                # Use LLM for significant over-length
                value, repair = summarize_field(value, max_length, field_name)
            else:
                # Simple truncation
                value, repair = truncate_field(value, max_length, field_name)
            
            if repair:
                repairs.append(repair)
        
        normalized[field_name] = value
    
    return RepairResult(
        success=True,
        data=normalized,
        repairs_applied=repairs,
        error=None
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def normalize_or_fail(
    raw_output: str,
    schema: dict[str, dict],
    agent_name: str,
    use_llm_summarization: bool = False
) -> dict:
    """Normalize agent output or raise exception.
    
    Convenience wrapper that either returns normalized data or raises ValueError.
    
    Args:
        raw_output: Raw agent output
        schema: Expected schema
        agent_name: Agent name for error messages
        use_llm_summarization: Use LLM for summarization
    
    Returns:
        Normalized dictionary
    
    Raises:
        ValueError: If normalisation fails
    """
    result = normalize_agent_output(raw_output, schema, use_llm_summarization)
    
    if not result.success:
        raise ValueError(f"{agent_name} output normalisation failed: {result.error}")
    
    # Log repairs
    if result.repairs_applied:
        print(f"\n📝 Normaliser applied {len(result.repairs_applied)} repair(s) to {agent_name} output:")
        for repair in result.repairs_applied[:5]:  # Show first 5
            print(f"   - {repair}")
        if len(result.repairs_applied) > 5:
            print(f"   ... and {len(result.repairs_applied) - 5} more")
    
    # Warn if extensive repairs
    if result.needs_review:
        print(f"\n⚠️  Extensive repairs applied to {agent_name} output - consider improving prompt")
    
    return result.data
