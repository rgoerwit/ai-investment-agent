import stat
from pathlib import Path

import pytest

from scripts import align_env_file
from scripts.align_env_file import align, effective_values, main, mismatch_count

REAL_TEMPLATE = Path(__file__).resolve().parents[2] / ".env.example"

TEMPLATE = """\
# Header comment
# Section one

# Google key
GOOGLE_API_KEY=your_google_api_key_here
# Optional knob
# TIMEOUT_SECONDS=45
ALPHA=default_alpha
# prose mentioning TIMEOUT_SECONDS would stay prose
# BETA=beta_default
BETA_ACTIVE=1
# BETA_ACTIVE=2
"""


def _run(tmp_path, source_text, template_text=TEMPLATE, *extra):
    source = tmp_path / "src.env"
    template = tmp_path / "template.env"
    source.write_text(source_text, encoding="utf-8")
    template.write_text(template_text, encoding="utf-8")
    out = tmp_path / "out.env"
    code = main(
        [str(source), "--template", str(template), "--output", str(out), *extra]
    )
    return code, out


def test_user_value_replaces_active_default_in_place() -> None:
    text = align("GOOGLE_API_KEY=real\n", TEMPLATE).text
    lines = text.splitlines()
    assert lines[lines.index("# Google key") + 1] == "GOOGLE_API_KEY=real"


def test_user_value_replaces_commented_placeholder() -> None:
    text = align("TIMEOUT_SECONDS=90\n", TEMPLATE).text
    lines = text.splitlines()
    assert lines[lines.index("# Optional knob") + 1] == "TIMEOUT_SECONDS=90"
    assert "# TIMEOUT_SECONDS=45" not in lines


def test_prose_naming_a_key_is_not_a_slot() -> None:
    text = align("TIMEOUT_SECONDS=90\n", TEMPLATE).text
    assert "# prose mentioning TIMEOUT_SECONDS would stay prose" in text.splitlines()
    assert text.count("TIMEOUT_SECONDS=90") == 1


def test_missing_active_default_is_commented_out_not_live() -> None:
    result = align("", TEMPLATE)
    assert "# GOOGLE_API_KEY=your_google_api_key_here" in result.text.splitlines()
    assert effective_values(result.text) == {}
    assert result.commented_defaults == 3


def test_key_active_and_commented_in_template_goes_to_active_line() -> None:
    lines = align("BETA_ACTIVE=7\n", TEMPLATE).text.splitlines()
    assert lines.count("BETA_ACTIVE=7") == 1
    assert "# BETA_ACTIVE=2" in lines


def test_second_commented_slot_for_same_key_is_not_filled() -> None:
    template = "# K=1\n# K=2\n"
    lines = align("K=9\n", template).text.splitlines()
    assert lines == ["K=9", "# K=2"]


@pytest.mark.parametrize(
    "line",
    [
        'QUOTED="has # inside"',
        'MULTI="first\nsecond"',
        "export EXPORTED=yes",
        "INLINE=value  # trailing note",
        "BARE",
        "SINGLE='a b'",
    ],
)
def test_user_text_is_copied_verbatim(line: str) -> None:
    result = align(line + "\n", "# T=1\n")
    assert line + "\n" in result.text
    assert effective_values(result.text) == effective_values(line + "\n")


def test_extra_keys_and_comments_are_carried_over() -> None:
    source = "# my note\nRETIRED=1\nGOOGLE_API_KEY=g\n# Header comment\n"
    result = align(source, TEMPLATE)
    tail = result.text.split("# --- Settings not in .env.example ---\n")[1]
    assert tail.startswith("RETIRED=1\n")
    assert "# my note\n" in tail.split("# --- Comments carried over")[1]
    assert result.carried == 1
    assert result.extra_keys == 1
    assert result.text.count("# Header comment") == 1


def test_duplicate_key_keeps_last_and_comments_out_earlier() -> None:
    result = align("ALPHA=old\nALPHA=new\n", TEMPLATE)
    lines = result.text.splitlines()
    assert "ALPHA=new" in lines
    assert "# ALPHA=old" in lines
    assert effective_values(result.text)["ALPHA"] == "new"


def test_malformed_lines_are_kept_in_carried_section() -> None:
    result = align("not a valid line\nALPHA=1\n", TEMPLATE)
    assert "not a valid line\n" in result.text
    assert effective_values(result.text) == effective_values(
        "not a valid line\nALPHA=1\n"
    )


def test_blank_lines_around_user_settings_do_not_leak_into_output() -> None:
    template = "# a\n\nK=1\n\n# b\n"
    assert align("\n\nK=2\n\n\n", template).text == "# a\n\nK=2\n\n# b\n"
    assert align("", template).text == "# a\n\n# K=1\n\n# b\n"


def test_last_source_line_without_newline() -> None:
    result = align("ALPHA=1", TEMPLATE)
    assert "ALPHA=1\n" in result.text
    assert effective_values(result.text)["ALPHA"] == "1"


def test_empty_source_and_empty_template() -> None:
    assert align("", "").text == ""
    result = align("A=1\n", "")
    assert effective_values(result.text) == {"A": "1"}


def test_mismatch_count_distinguishes_missing_none_and_empty() -> None:
    assert mismatch_count({"A": None}, {"A": ""}) == 1
    assert mismatch_count({"A": None}, {}) == 1
    assert mismatch_count({"A": "1"}, {"A": "1"}) == 0
    assert mismatch_count({"A": "1"}, {"B": "1"}) == 2


def test_cli_writes_output_with_private_mode(tmp_path, capsys) -> None:
    code, out = _run(tmp_path, "GOOGLE_API_KEY=real\n")
    assert code == 0
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert "GOOGLE_API_KEY=real" in out.read_text(encoding="utf-8")
    assert "effective values identical" in capsys.readouterr().out


def test_cli_tightens_mode_of_existing_output(tmp_path) -> None:
    out = tmp_path / "out.env"
    out.write_text("old", encoding="utf-8")
    out.chmod(0o644)
    _run(tmp_path, "ALPHA=1\n")
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert out.read_text(encoding="utf-8") != "old"


def test_cli_check_writes_nothing(tmp_path) -> None:
    code, out = _run(tmp_path, "ALPHA=1\n", TEMPLATE, "--check")
    assert code == 0
    assert not out.exists()


def test_cli_refuses_to_overwrite_source_or_template(tmp_path) -> None:
    source = tmp_path / "src.env"
    template = tmp_path / "template.env"
    source.write_text("ALPHA=1\n", encoding="utf-8")
    template.write_text(TEMPLATE, encoding="utf-8")
    for target in (source, template):
        with pytest.raises(SystemExit) as exc:
            main([str(source), "--template", str(template), "--output", str(target)])
        assert exc.value.code == 2
    assert source.read_text(encoding="utf-8") == "ALPHA=1\n"
    assert template.read_text(encoding="utf-8") == TEMPLATE


def test_report_contains_no_key_or_value(tmp_path, capsys) -> None:
    source = (
        "SECRETKEY_ONE=hunter2value\nGOOGLE_API_KEY=topsecretvalue\n# private note\n"
    )
    code, _ = _run(tmp_path, source)
    captured = capsys.readouterr()
    assert code == 0
    shown = captured.out + captured.err
    for leaked in ("SECRETKEY_ONE", "hunter2value", "GOOGLE_API_KEY", "topsecretvalue"):
        assert leaked not in shown


def test_dropped_key_fails_invariant_writes_nothing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(align_env_file, "_user_text", lambda binding: "")
    code, out = _run(tmp_path, "SECRETKEY_ONE=hunter2value\nALPHA=1\n")
    captured = capsys.readouterr()
    assert code == 2
    assert not out.exists()
    assert "2 keys" in captured.err
    assert "SECRETKEY_ONE" not in captured.err + captured.out
    assert "hunter2value" not in captured.err + captured.out


def test_reordering_that_breaks_interpolation_writes_nothing(
    tmp_path, capsys, monkeypatch
) -> None:
    # The template names LATE before EARLY, so ${EARLY} would no longer resolve.
    monkeypatch.delenv("EARLY", raising=False)
    template = "# LATE=\n# EARLY=\n"
    code, out = _run(tmp_path, "EARLY=base\nLATE=${EARLY}/x\n", template)
    assert code == 2
    assert not out.exists()
    assert "base" not in capsys.readouterr().err


def test_real_template_keeps_every_comment_line_in_order() -> None:
    template_text = REAL_TEMPLATE.read_text(encoding="utf-8")
    source = "GOOGLE_API_KEY=g\nFINNHUB_API_KEY=f\nONLY_IN_SOURCE=1\n"
    result = align(source, template_text)
    template_comments = [
        line
        for line in template_text.splitlines()
        if line.lstrip().startswith("#") or not line.strip()
    ]
    out_lines = iter(result.text.splitlines())
    for expected in template_comments:
        assert any(line == expected for line in out_lines), expected
    assert effective_values(result.text) == effective_values(source)


def test_real_template_aligned_to_itself_is_identical_in_value() -> None:
    template_text = REAL_TEMPLATE.read_text(encoding="utf-8")
    result = align(template_text, template_text)
    assert effective_values(result.text) == effective_values(template_text)
    assert result.extra_keys == 0
