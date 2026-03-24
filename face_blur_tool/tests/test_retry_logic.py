"""Tests for retry-once behavior."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call, patch
from src.pipeline.retry import retry_once


class TestRetryOnce:
    def test_success_on_first_attempt(self):
        fn = MagicMock(return_value="ok")
        result, count = retry_once(fn, label="test")
        assert result == "ok"
        assert count == 0
        fn.assert_called_once()

    def test_success_on_second_attempt(self):
        fn = MagicMock(side_effect=[RuntimeError("first"), "ok"])
        with patch("src.pipeline.retry.time.sleep"):
            result, count = retry_once(fn, label="test")
        assert result == "ok"
        assert count == 1
        assert fn.call_count == 2

    def test_fails_both_attempts_raises(self):
        fn = MagicMock(side_effect=[RuntimeError("first"), RuntimeError("second")])
        with patch("src.pipeline.retry.time.sleep"):
            with pytest.raises(RuntimeError, match="second"):
                retry_once(fn, label="test")
        assert fn.call_count == 2

    def test_sleep_between_attempts(self):
        fn = MagicMock(side_effect=[ValueError("oops"), "ok"])
        with patch("src.pipeline.retry.time.sleep") as mock_sleep:
            retry_once(fn, label="test")
        mock_sleep.assert_called_once_with(2)

    def test_args_forwarded(self):
        fn = MagicMock(return_value=42)
        result, _ = retry_once(fn, "a", "b", key="val", label="lbl")
        fn.assert_called_once_with("a", "b", key="val")
        assert result == 42
