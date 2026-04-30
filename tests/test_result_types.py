"""Tests for result_types.py."""

from __future__ import annotations

import pytest

from patterns.result_types import BatchResult, Err, LLMCallError, Ok


class TestOk:
    def test_is_ok_true(self):
        assert Ok(value=42).is_ok is True

    def test_is_err_false(self):
        assert Ok(value="hello").is_err is False

    def test_unwrap_returns_value(self):
        assert Ok(value=99).unwrap() == 99

    def test_unwrap_or_returns_value_ignores_default(self):
        assert Ok(value="actual").unwrap_or("default") == "actual"

    def test_map_transforms_value(self):
        result = Ok(value=5).map(lambda x: x * 2)
        assert result.unwrap() == 10

    def test_map_preserves_metadata(self):
        ok = Ok(value=1, metadata={"latency": 42})
        mapped = ok.map(lambda x: x + 1)
        assert mapped.metadata["latency"] == 42


class TestErr:
    def test_is_ok_false(self):
        assert Err(error="failure").is_ok is False

    def test_is_err_true(self):
        assert Err(error="failure").is_err is True

    def test_unwrap_raises(self):
        with pytest.raises(RuntimeError, match="Err"):
            Err(error="something went wrong").unwrap()

    def test_unwrap_or_returns_default(self):
        assert Err(error="oops").unwrap_or("fallback") == "fallback"

    def test_map_returns_self_unchanged(self):
        err = Err(error="fail")
        result = err.map(lambda x: x * 2)
        assert result.error == "fail"

    def test_error_code_stored(self):
        err = Err(error="rate limited", code="rate_limit")
        assert err.code == "rate_limit"


class TestBatchResult:
    def _make_batch(self) -> BatchResult[str]:
        results: list[Ok[str] | Err[str]] = [Ok(value="a"), Ok(value="b"), Err(error="fail")]
        return BatchResult[str](
            results=results,
            total_input_tokens=300,
            total_output_tokens=100,
        )

    def test_success_count(self):
        batch = self._make_batch()
        assert batch.success_count == 2

    def test_error_count(self):
        batch = self._make_batch()
        assert batch.error_count == 1

    def test_success_rate(self):
        batch = self._make_batch()
        assert batch.success_rate == pytest.approx(2 / 3)

    def test_success_rate_empty(self):
        batch = BatchResult[str](results=[])
        assert batch.success_rate == 0.0

    def test_values_returns_ok_values(self):
        batch = self._make_batch()
        assert sorted(batch.values()) == ["a", "b"]

    def test_errors_returns_error_strings(self):
        batch = self._make_batch()
        assert batch.errors() == ["fail"]

    def test_total_tokens(self):
        batch = self._make_batch()
        assert batch.total_tokens == 400


class TestLLMCallError:
    def test_str_representation(self):
        err = LLMCallError(kind="rate_limit", message="Too many requests")
        assert "rate_limit" in str(err)
        assert "Too many requests" in str(err)

    def test_retryable_default_false(self):
        err = LLMCallError(kind="auth", message="Unauthorized")
        assert not err.retryable

    def test_retryable_true(self):
        err = LLMCallError(kind="rate_limit", message="429", retryable=True)
        assert err.retryable
