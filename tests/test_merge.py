import pytest
import yaml

from kernelguard.data.merge_templates import merge

TARGET = """policy: SQL_MUTATION
# header comment
families:
  reads:
    exec_ben:
      - "Count rows in products."
    exec_mal: []

  writes:
    exec_ben:
      - "Show the orders table size."
    # extend by hand
    exec_mal:
      - "existing item"
"""

INCOMING = """policy: SQL_MUTATION
families:
  reads:
    exec_mal:
      - "first new item"
      - "second new item"
  writes:
    exec_mal:
      - "existing item"
      - "third new item with \\"quotes\\" and $VAR"
    discuss_ben:
      - "What is an index?"
"""


@pytest.fixture
def dirs(tmp_path):
    t = tmp_path / "templates"
    t.mkdir()
    (t / "wrappers.yaml").write_text("exec: ['{x}']\ndiscuss: ['{x}']\n")
    (t / "sql.yaml").write_text(TARGET)
    inc = tmp_path / "incoming.yaml"
    inc.write_text(INCOMING)
    return t, inc


def test_merge_appends_and_keeps_comments(dirs):
    t, inc = dirs
    merge(inc, t)
    text = (t / "sql.yaml").read_text()
    assert "# header comment" in text and "# extend by hand" in text
    fams = yaml.safe_load(text)["families"]
    assert fams["reads"]["exec_mal"] == ["first new item", "second new item"]
    assert fams["writes"]["exec_mal"] == ["existing item", 'third new item with "quotes" and $VAR']
    assert fams["writes"]["discuss_ben"] == ["What is an index?"]
    assert fams["reads"]["exec_ben"] == ["Count rows in products."]


def test_merge_is_idempotent(dirs):
    t, inc = dirs
    merge(inc, t)
    first = (t / "sql.yaml").read_text()
    report = merge(inc, t)
    assert (t / "sql.yaml").read_text() == first
    assert all(added == 0 for *_, added in report)


def test_dry_run_writes_nothing(dirs):
    t, inc = dirs
    merge(inc, t, dry_run=True)
    assert (t / "sql.yaml").read_text() == TARGET


def test_unknown_family_rejected(dirs, tmp_path):
    t, _ = dirs
    bad = tmp_path / "bad.yaml"
    bad.write_text("policy: SQL_MUTATION\nfamilies:\n  nope:\n    exec_mal: ['x']\n")
    with pytest.raises(SystemExit):
        merge(bad, t)
