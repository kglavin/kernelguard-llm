from pathlib import Path

from kernelguard.data.make_dataset import SEEDS, build, load_templates
from kernelguard.data.schema import load_jsonl, split_by_family, by_policy

TEMPLATES = Path(__file__).resolve().parents[1] / "data" / "templates"


def _items():
    wrappers, rows = load_templates(TEMPLATES)
    return build(load_jsonl(SEEDS), wrappers, rows), wrappers


def test_every_seed_is_included_unwrapped():
    items, _ = _items()
    users = {(i.policy, i.user) for i in items}
    for s in load_jsonl(SEEDS):
        assert (s.policy, s.user) in users


def test_ids_unique_and_no_duplicate_text():
    items, _ = _items()
    assert len({i.id for i in items}) == len(items)
    assert len({(i.policy, i.user) for i in items}) == len(items)


def test_wrapper_variants_share_family():
    items, wrappers = _items()
    by_base = {}
    for i in items:
        by_base.setdefault(i.meta["base"], set()).add(i.family)
    assert all(len(f) == 1 for f in by_base.values())


def test_every_policy_has_both_labels_on_both_sides_of_split():
    items, _ = _items()
    for policy, its in by_policy(items).items():
        train, test = split_by_family(its, 0.25)
        assert {i.label for i in train} == {0, 1}, policy
        assert {i.label for i in test} == {0, 1}, policy
