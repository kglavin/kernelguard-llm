import pytest

from kernelguard.data.schema import Item, split_by_family, load_jsonl, save_jsonl
from kernelguard.data.make_seeds import build


def test_label():
    assert Item("a", "SHELL_EXEC", "exec_mal", "f", "x").label == 1
    for c in ("exec_ben", "discuss_mal", "discuss_ben"):
        assert Item("a", "SHELL_EXEC", c, "f", "x").label == 0


def test_validate_rejects_bad_policy():
    with pytest.raises(ValueError):
        Item("a", "NOPE", "exec_mal", "f", "x").validate()


def test_seeds_roundtrip(tmp_path):
    items = build()
    for it in items:
        it.validate()
    p = tmp_path / "s.jsonl"
    save_jsonl(items, p)
    back = load_jsonl(p)
    assert [i.id for i in back] == [i.id for i in items]


def test_split_by_family_is_disjoint():
    items = build()
    train, test = split_by_family(items, 0.25)
    assert {i.family for i in train}.isdisjoint({i.family for i in test})
    assert len(train) + len(test) == len(items)


def test_cache_key_ignores_id_but_not_text():
    from kernelguard.extract.activations import _cache_key
    a = Item("id-1", "SHELL_EXEC", "exec_ben", "f", "list files")
    b = Item("id-2", "SHELL_EXEC", "exec_ben", "f", "list files")
    c = Item("id-1", "SHELL_EXEC", "exec_ben", "f", "list dirs")
    assert _cache_key("m", a) == _cache_key("m", b)
    assert _cache_key("m", a) != _cache_key("m", c)
    assert _cache_key("m", a) != _cache_key("other-model", a)
