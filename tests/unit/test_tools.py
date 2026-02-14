"""Unit tests for agent tools - read and edit tool contracts.

Tests run in a temporary directory (cwd) so the real workspace is not modified.
"""

import os
import pytest
from pathlib import Path

from agents.tools.read import read_file, read_file_lines, get_file_info
from agents.tools.edit import write_file, create_file, apply_edit


@pytest.fixture(autouse=True)
def tools_cwd(tmp_path, monkeypatch):
    """Run tool tests in a temp dir; edit tools need no .git above so cwd is repo root."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestReadFile:
    """Test read_file contract."""

    def test_missing_file_returns_error_message(self, tools_cwd):
        out = read_file.invoke({"path": "nonexistent.txt"})
        assert "File does not exist" in out

    def test_directory_returns_error_message(self, tools_cwd):
        (tools_cwd / "adir").mkdir()
        out = read_file.invoke({"path": "adir"})
        assert "Not a file" in out

    def test_reads_existing_file(self, tools_cwd):
        (tools_cwd / "hello.py").write_text("print('hi')\n", encoding="utf-8")
        out = read_file.invoke({"path": "hello.py"})
        assert out == "print('hi')\n"

    def test_cleans_res_prefix(self, tools_cwd):
        (tools_cwd / "x.gd").write_text("extends Node\n", encoding="utf-8")
        out = read_file.invoke({"path": "res://x.gd"})
        assert "extends Node" in out


class TestReadFileLines:
    """Test read_file_lines contract."""

    def test_missing_file_returns_error(self, tools_cwd):
        out = read_file_lines.invoke({"path": "nope.py"})
        assert "File does not exist" in out

    def test_returns_line_range_with_numbers(self, tools_cwd):
        (tools_cwd / "f.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
        out = read_file_lines.invoke({"path": "f.py", "start_line": 1, "end_line": 2})
        assert "File: f.py" in out
        assert "1|" in out and "2|" in out
        assert "line1" in out and "line2" in out

    def test_respects_max_lines(self, tools_cwd):
        content = "\n".join([f"line{i}" for i in range(1, 20)])
        (tools_cwd / "big.py").write_text(content, encoding="utf-8")
        out = read_file_lines.invoke({"path": "big.py", "start_line": 1, "max_lines": 5})
        assert out.count("|") == 5


class TestGetFileInfo:
    """Test get_file_info contract."""

    def test_missing_returns_error(self, tools_cwd):
        out = get_file_info.invoke({"path": "missing"})
        assert "File does not exist" in out

    def test_directory_returns_message(self, tools_cwd):
        (tools_cwd / "d").mkdir()
        out = get_file_info.invoke({"path": "d"})
        assert "directory" in out

    def test_file_returns_size_and_lines(self, tools_cwd):
        (tools_cwd / "info.py").write_text("a\nb\nc\n", encoding="utf-8")
        out = get_file_info.invoke({"path": "info.py"})
        assert "info.py" in out
        assert "Lines: 3" in out


class TestWriteFile:
    """Test write_file contract - creates or overwrites."""

    def test_creates_new_file(self, tools_cwd):
        out = write_file.invoke({"path": "new.py", "content": "x = 1"})
        assert "Successfully" in out
        assert (tools_cwd / "new.py").read_text() == "x = 1"

    def test_overwrites_existing_file(self, tools_cwd):
        (tools_cwd / "existing.py").write_text("old", encoding="utf-8")
        out = write_file.invoke({"path": "existing.py", "content": "new"})
        assert "Successfully" in out
        assert (tools_cwd / "existing.py").read_text() == "new"

    def test_creates_parent_dirs(self, tools_cwd):
        out = write_file.invoke({"path": "sub/dir/file.py", "content": "ok"})
        assert "Successfully" in out
        assert (tools_cwd / "sub/dir/file.py").read_text() == "ok"

    def test_rejects_path_traversal(self, tools_cwd):
        out = write_file.invoke({"path": "../outside.py", "content": "x"})
        assert "Security error" in out


class TestCreateFile:
    """Test create_file contract - fails if file exists."""

    def test_creates_new_file(self, tools_cwd):
        out = create_file.invoke({"path": "brand_new.py", "content": "created"})
        assert "Successfully created" in out
        assert (tools_cwd / "brand_new.py").read_text() == "created"

    def test_fails_when_file_exists(self, tools_cwd):
        (tools_cwd / "already.py").write_text("existing", encoding="utf-8")
        out = create_file.invoke({"path": "already.py", "content": "new content"})
        assert "File already exists" in out
        assert (tools_cwd / "already.py").read_text() == "existing"


class TestApplyEdit:
    """Test apply_edit contract."""

    def test_file_not_found_returns_error(self, tools_cwd):
        out = apply_edit.invoke({"path": "nope.py", "old_text": "a", "new_text": "b"})
        assert "File does not exist" in out

    def test_replaces_old_with_new(self, tools_cwd):
        (tools_cwd / "edit.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
        out = apply_edit.invoke({
            "path": "edit.py",
            "old_text": "line2",
            "new_text": "replaced",
        })
        assert "Successfully applied" in out
        assert (tools_cwd / "edit.py").read_text() == "line1\nreplaced\nline3\n"

    def test_old_text_not_found_returns_error(self, tools_cwd):
        (tools_cwd / "edit2.py").write_text("only this", encoding="utf-8")
        out = apply_edit.invoke({
            "path": "edit2.py",
            "old_text": "missing",
            "new_text": "x",
        })
        assert "not found" in out
