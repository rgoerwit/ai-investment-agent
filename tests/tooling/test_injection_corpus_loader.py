from __future__ import annotations

import pytest

from tests.helpers.injection_corpus import load_corpus, load_detection_floors

pytestmark = pytest.mark.security


def test_corpus_loads_and_filters_by_source_category_and_expectation():
    all_cases = load_corpus()
    memory_cases = load_corpus(source_kind="memory_write")
    override_cases = load_corpus(category="override")
    must_block_cases = load_corpus(expectation="must_block")

    assert len(all_cases) >= 100
    assert memory_cases
    assert all(case["source_kind"] == "memory_write" for case in memory_cases)
    assert override_cases
    assert all(case["category"] == "override" for case in override_cases)
    assert must_block_cases
    assert all(case["expectation"] == "must_block" for case in must_block_cases)


def test_corpus_ids_are_unique_and_floors_reference_existing_buckets():
    cases = load_corpus()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))

    buckets = {(case["source_kind"], case["category"]) for case in cases}
    for floor in load_detection_floors():
        assert (floor["source_kind"], floor["category"]) in buckets
