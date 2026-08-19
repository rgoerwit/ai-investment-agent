"""The test session must never touch the operator's persistent ChromaDB.

This is a data-loss guard, not hygiene. ``cleanup_all_memories(days=0)`` called
with no ticker skips its prefix filter and calls ``delete_collection`` on *every*
collection it finds — and ``test_contamination_vectors.py`` calls it exactly that
way, twice, to get a "clean slate". Pointed at the default ``./chroma_db``, a
single ``pytest tests/memory/`` destroyed the entire ``lessons_learned`` corpus
along with ``macro_events`` and every legacy collection.

That is what made the retrospective's lesson store keep reappearing empty and
same-day, and it was originally misdiagnosed as an external process. Reproduced
2026-08-16 by seeding a sentinel record, running ``pytest tests/memory/``, and
finding the collection gone.

``tests/conftest.py::setup_test_env`` redirects ``CHROMA_PERSIST_DIR`` to a
session temp directory. These tests fail if that redirect is ever removed or
weakened — before a suite run can eat the corpus again.
"""

from __future__ import annotations

from pathlib import Path

from src.config import config

# Values that would put the test session on top of real operator data.
PRODUCTION_CHROMA_PATHS = {"chroma_db", "./chroma_db"}


def _resolved() -> Path:
    return Path(str(config.chroma_persist_directory)).resolve()


class TestChromaIsolationIsInEffect:
    def test_config_points_to_the_exact_session_temp_directory(
        self, test_chroma_persist_dir: Path
    ):
        """Reject *any* operator path, not only the repository default.

        A weaker guard that merely rejects ``./chroma_db`` still passes when an
        operator configures a different external CHROMA_PERSIST_DIR. That path is
        real data too, and an unscoped cleanup would destroy it just as surely.
        """
        assert _resolved() == test_chroma_persist_dir.resolve()

    def test_the_session_is_not_pointed_at_the_production_directory(self):
        raw = str(config.chroma_persist_directory)
        assert raw not in PRODUCTION_CHROMA_PATHS, (
            "The test session is pointed at the operator's real ChromaDB. "
            "cleanup_all_memories(days=0) will delete every collection in it, "
            "including lessons_learned. Restore the CHROMA_PERSIST_DIR redirect "
            "in tests/conftest.py::setup_test_env."
        )

    def test_the_session_directory_is_outside_the_repository(self):
        """A path inside the repo would also dirty the working tree."""
        repo_root = Path(__file__).resolve().parents[2]
        assert repo_root not in _resolved().parents, (
            f"Test ChromaDB resolves to {_resolved()}, inside the repository at "
            f"{repo_root}. It must live in a temp directory."
        )

    def test_the_real_directory_is_not_the_one_in_use(self):
        real = (Path(__file__).resolve().parents[2] / "chroma_db").resolve()
        assert _resolved() != real

    def test_a_full_cleanup_would_only_reach_the_temp_directory(self):
        """The predicate that matters: what would an unscoped wipe destroy?

        ``cleanup_all_memories`` builds its own PersistentClient from
        ``config.chroma_persist_directory`` — it takes no injected client — so
        that setting is the single chokepoint between a test and real data.
        """
        import inspect

        from src import memory

        source = inspect.getsource(memory.cleanup_all_memories)
        assert "config.chroma_persist_directory" in source, (
            "cleanup_all_memories no longer reads config.chroma_persist_directory; "
            "the test-isolation redirect may no longer protect the operator's data"
        )
        assert (
            _resolved() != (Path(__file__).resolve().parents[2] / "chroma_db").resolve()
        )


class TestTheUnscopedWipeStillExists:
    """Document the sharp edge rather than pretend it was removed.

    ``cleanup_all_memories(days=0)`` with no ticker deleting everything is the
    documented contract ("If None, clean ALL collections"), and the graph's own
    call site passes a ticker so it stays prefix-scoped. The danger is entirely
    in *where it is pointed*, which is what the redirect fixes.
    """

    def test_a_ticker_scoped_cleanup_filters_by_prefix(self):
        import inspect

        from src import memory

        source = inspect.getsource(memory.cleanup_all_memories)
        assert "if target_prefix and not collection_name.startswith(target_prefix)" in (
            source
        ), "the ticker prefix filter is what keeps the graph's cleanup scoped"

    def test_the_graph_call_site_always_passes_a_ticker(self):
        import inspect

        from src.graph import components

        source = inspect.getsource(components)
        assert "cleanup_all_memories(days=0, ticker=ticker)" in source, (
            "graph component cleanup must stay ticker-scoped; an unscoped call "
            "there would delete lessons_learned on every analysis run"
        )
