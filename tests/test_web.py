"""Tests for the ``ppa-web`` command line interface."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from proper_pixel_art import web


def test_web_main_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that web.main() launches with default server_name and server_port as None."""
    monkeypatch.setattr(sys, "argv", ["ppa-web"])

    mock_demo = MagicMock()
    with patch("proper_pixel_art.web.create_demo", return_value=mock_demo):
        web.main()

    mock_demo.launch.assert_called_once_with(server_name=None, server_port=None)


def test_web_main_custom_host_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that web.main() parses --host and --port and passes them to demo.launch."""
    monkeypatch.setattr(sys, "argv", ["ppa-web", "--host", "0.0.0.0", "--port", "8080"])

    mock_demo = MagicMock()
    with patch("proper_pixel_art.web.create_demo", return_value=mock_demo):
        web.main()

    mock_demo.launch.assert_called_once_with(server_name="0.0.0.0", server_port=8080)
