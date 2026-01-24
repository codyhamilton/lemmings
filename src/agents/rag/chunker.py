"""Smart code chunking that preserves semantic units."""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class CodeChunk:
    """A semantic chunk of code with metadata."""
    
    path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str  # "class", "function", "config", "scene", "doc", "module"
    symbols: list[str]  # Class/function names for filtering
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    def get_id(self) -> str:
        """Generate unique ID for this chunk."""
        # For chunks with identical line numbers (e.g., JSON keys), include symbol for uniqueness
        if self.start_line == self.end_line == 1 and self.symbols:
            # Use first symbol to make ID unique
            symbol_part = self.symbols[0].replace('/', '_').replace(':', '_')
            return f"{self.path}:{symbol_part}:1-1"
        return f"{self.path}:{self.start_line}-{self.end_line}"


def chunk_gdscript(path: str, content: str) -> list[CodeChunk]:
    """Chunk GDScript files by class and function definitions.
    
    Preserves:
    - Class definitions with their docstrings
    - Function definitions with their docstrings
    - Top-level module code
    
    Args:
        path: File path
        content: File content
    
    Returns:
        List of CodeChunk objects
    """
    chunks = []
    lines = content.split('\n')
    
    # Pattern for class definitions
    class_pattern = re.compile(r'^class\s+(\w+)')
    # Pattern for function definitions
    func_pattern = re.compile(r'^(?:static\s+)?func\s+(\w+)')
    
    current_chunk_start = 0
    current_indent = 0
    current_symbol = None
    current_type = "module"
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        
        # Check for class definition
        class_match = class_pattern.match(stripped)
        if class_match and indent == 0:
            # Save previous chunk if exists
            if current_symbol or i > current_chunk_start:
                chunk_content = '\n'.join(lines[current_chunk_start:i])
                if chunk_content.strip():
                    chunks.append(CodeChunk(
                        path=path,
                        start_line=current_chunk_start + 1,
                        end_line=i,
                        content=chunk_content,
                        chunk_type=current_type,
                        symbols=[current_symbol] if current_symbol else [],
                    ))
            
            current_chunk_start = i
            current_symbol = class_match.group(1)
            current_type = "class"
            current_indent = indent
            i += 1
            continue
        
        # Check for function definition
        func_match = func_pattern.match(stripped)
        if func_match:
            # Save previous chunk if we're at top level or new function at same level
            if current_symbol and indent <= current_indent:
                chunk_content = '\n'.join(lines[current_chunk_start:i])
                if chunk_content.strip():
                    chunks.append(CodeChunk(
                        path=path,
                        start_line=current_chunk_start + 1,
                        end_line=i,
                        content=chunk_content,
                        chunk_type=current_type,
                        symbols=[current_symbol] if current_symbol else [],
                    ))
                
                current_chunk_start = i
            
            current_symbol = func_match.group(1)
            current_type = "function"
            current_indent = indent
        
        i += 1
    
    # Add final chunk
    if current_chunk_start < len(lines):
        chunk_content = '\n'.join(lines[current_chunk_start:])
        if chunk_content.strip():
            chunks.append(CodeChunk(
                path=path,
                start_line=current_chunk_start + 1,
                end_line=len(lines),
                content=chunk_content,
                chunk_type=current_type,
                symbols=[current_symbol] if current_symbol else [],
            ))
    
    return chunks


def chunk_json(path: str, content: str) -> list[CodeChunk]:
    """Chunk JSON config files by top-level keys.
    
    Args:
        path: File path
        content: File content
    
    Returns:
        List of CodeChunk objects
    """
    chunks = []
    
    try:
        data = json.loads(content)
        
        if isinstance(data, dict):
            # For object-based configs (like ships.json with ship definitions)
            for key, value in data.items():
                chunk_content = json.dumps({key: value}, indent=2)
                chunks.append(CodeChunk(
                    path=path,
                    start_line=1,  # JSON line numbers are not reliable
                    end_line=1,
                    content=chunk_content,
                    chunk_type="config",
                    symbols=[key],
                ))
        elif isinstance(data, list):
            # For array-based configs, chunk by item
            for i, item in enumerate(data):
                chunk_content = json.dumps(item, indent=2)
                symbol = item.get("id") or item.get("name") or f"item_{i}"
                chunks.append(CodeChunk(
                    path=path,
                    start_line=1,
                    end_line=1,
                    content=chunk_content,
                    chunk_type="config",
                    symbols=[str(symbol)],
                ))
    except json.JSONDecodeError:
        # If parsing fails, treat as single chunk
        chunks.append(CodeChunk(
            path=path,
            start_line=1,
            end_line=content.count('\n') + 1,
            content=content,
            chunk_type="config",
            symbols=[],
        ))
    
    return chunks


def chunk_tscn(path: str, content: str) -> list[CodeChunk]:
    """Extract key information from TSCN scene files.
    
    TSCN files are Godot's scene format. We extract:
    - Node hierarchy
    - Script references
    - Resource paths
    
    Args:
        path: File path
        content: File content
    
    Returns:
        List of CodeChunk objects
    """
    chunks = []
    lines = content.split('\n')
    
    # Extract node names and types
    nodes = []
    scripts = []
    
    for line in lines:
        # Node definitions: [node name="NodeName" type="NodeType"]
        node_match = re.search(r'\[node name="([^"]+)".*?type="([^"]+)"', line)
        if node_match:
            nodes.append(f"{node_match.group(1)} ({node_match.group(2)})")
        
        # Script references
        script_match = re.search(r'script\s*=.*?"([^"]+)"', line)
        if script_match:
            scripts.append(script_match.group(1))
    
    # Create a summary chunk
    summary_parts = []
    if nodes:
        summary_parts.append("Nodes:\n" + "\n".join(f"- {n}" for n in nodes[:10]))
    if scripts:
        summary_parts.append("Scripts:\n" + "\n".join(f"- {s}" for s in scripts))
    
    if summary_parts:
        chunks.append(CodeChunk(
            path=path,
            start_line=1,
            end_line=len(lines),
            content="\n\n".join(summary_parts),
            chunk_type="scene",
            symbols=nodes[:5],  # Use first few node names as symbols
        ))
    
    return chunks


def chunk_markdown(path: str, content: str) -> list[CodeChunk]:
    """Chunk Markdown documentation by headers.
    
    Args:
        path: File path
        content: File content
    
    Returns:
        List of CodeChunk objects
    """
    chunks = []
    lines = content.split('\n')
    
    current_header = None
    current_start = 0
    
    for i, line in enumerate(lines):
        # Check for headers (# or ##)
        if line.startswith('#'):
            # Save previous section
            if current_start < i:
                chunk_content = '\n'.join(lines[current_start:i]).strip()
                if chunk_content:
                    chunks.append(CodeChunk(
                        path=path,
                        start_line=current_start + 1,
                        end_line=i,
                        content=chunk_content,
                        chunk_type="doc",
                        symbols=[current_header] if current_header else [],
                    ))
            
            current_header = line.lstrip('#').strip()
            current_start = i
    
    # Add final section
    if current_start < len(lines):
        chunk_content = '\n'.join(lines[current_start:]).strip()
        if chunk_content:
            chunks.append(CodeChunk(
                path=path,
                start_line=current_start + 1,
                end_line=len(lines),
                content=chunk_content,
                chunk_type="doc",
                symbols=[current_header] if current_header else [],
            ))
    
    return chunks


def chunk_asset(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    """Chunk asset files (images, etc.) - only store the path/name, not content.
    
    Args:
        file_path: Absolute path to the file
        repo_root: Repository root path
    
    Returns:
        List of CodeChunk objects (single chunk with just the path)
    """
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        # File is outside repo
        return []
    
    # Get file name and directory for context
    file_name = file_path.name
    parent_dir = file_path.parent.name if file_path.parent != file_path.parents[0] else ""
    
    # Create a minimal chunk that only contains the path information
    # This allows searching for asset names without storing binary content
    content = f"Asset file: {rel_path}\nFile name: {file_name}"
    if parent_dir:
        content += f"\nDirectory: {parent_dir}"
    
    return [CodeChunk(
        path=rel_path,
        start_line=1,
        end_line=1,
        content=content,
        chunk_type="asset",
        symbols=[file_name],
    )]


def chunk_file(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    """Chunk a file based on its type.
    
    Args:
        file_path: Absolute path to the file
        repo_root: Repository root path
    
    Returns:
        List of CodeChunk objects
    """
    # Get relative path for storage
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        # File is outside repo
        return []
    
    suffix = file_path.suffix.lower()
    
    # Handle asset files (name only, not content)
    if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.import'}:
        return chunk_asset(file_path, repo_root)
    
    # For text-based files, read content
    try:
        content = file_path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, IOError):
        # Skip binary or unreadable files
        return []
    
    if suffix == '.gd':
        return chunk_gdscript(rel_path, content)
    elif suffix == '.json':
        return chunk_json(rel_path, content)
    elif suffix == '.tscn':
        return chunk_tscn(rel_path, content)
    elif suffix in ('.md', '.markdown'):
        return chunk_markdown(rel_path, content)
    else:
        # For other files, treat as single chunk if small enough
        if len(content) < 5000:  # ~1000 lines
            return [CodeChunk(
                path=rel_path,
                start_line=1,
                end_line=content.count('\n') + 1,
                content=content,
                chunk_type="module",
                symbols=[],
            )]
    
    return []
