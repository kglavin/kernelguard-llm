import json

from kernelguard.data.import_sessions import build, tag


def _call(desc, cmd):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"description": desc, "command": cmd}}]}}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")


def test_build(tmp_path):
    _write(tmp_path / "-home-u-proj" / "s1.jsonl", [
        _call("List files", "ls -la"),
        _call("List files", "ls -la"),  # duplicate
        _call("", "ls"),  # no description
        _call("Long one", "x" * 50),
        {"type": "user", "message": {"content": "hi"}},
    ])
    _write(tmp_path / "-home-u-proj" / "s1" / "subagents" / "a.jsonl", [_call("Drop temp table", "psql -c 'DROP TABLE t'")])
    _write(tmp_path / "-home-u-kernelguard" / "s2.jsonl", [_call("Skip me", "echo")])

    items, stats = build(tmp_path, ["kernelguard"], max_chars=40)
    assert stats["files_excluded"] == 1 and stats["duplicates"] == 1 and stats["too_long"] == 1
    assert stats["pairs"] == 2 and stats["sessions"] == 1  # the subagent file belongs to session s1
    assert len(items) == 4 and all(i.condition == "exec_ben" and i.label == 0 for i in items)
    assert {i.meta["form"] for i in items} == {"desc", "desc_cmd"}
    for i in items:
        i.validate()
    assert "List files:\nls -la" in {i.user for i in items if i.meta["form"] == "desc_cmd"}


def test_tag():
    assert tag("sqlite3 db 'DELETE FROM t'") == ("sql", "SQL_MUTATION")
    assert tag("cat .env") == ("cred", "CRED_EXFIL")
    assert tag("rm -rf build") == ("destructive", "SHELL_EXEC")
    assert tag("git status") == ("other", "SHELL_EXEC")
