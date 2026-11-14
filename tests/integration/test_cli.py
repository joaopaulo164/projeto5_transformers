"""Integration tests for CLI entrypoint."""

from __future__ import annotations

import pytest

from sentiment_transformer.cli import main


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_no_args() -> None:
    code = main([])
    assert code == 0


def test_cli_invalid_config() -> None:
    code = main(["run", "--config", "non_existent_config.yaml"])
    assert code == 1
