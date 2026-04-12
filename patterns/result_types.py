"""Result type patterns for LLM pipeline outputs.

Typed result containers that distinguish success from failure without
exceptions. Inspired by Rust's Result<T, E> — all paths are explicit
in the type signature.

Pattern:
    LLM call returns Ok(value) | Err(reason)
    Caller pattern-matches or uses .unwrap() with a default
    No silent failures, no bare None returns

Usage::

    result = call_llm(prompt)
    match result:
        case Ok(value=resp):
            process(resp)
        case Err(error=err):
            log_and_fallback(err)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass
class Ok(Generic[T]):
    """Successful result wrapping a value.

    Attributes:
        value: The success value.
        metadata: Optional metadata (latency, tokens, etc.)
    """

    value: T
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        """Always True for Ok."""
        return True

    @property
    def is_err(self) -> bool:
        """Always False for Ok."""
        return False

    def unwrap(self) -> T:
        """Return the value (safe — always succeeds for Ok).

        Returns:
            The wrapped value.
        """
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Return value (ignores default for Ok).

        Args:
            default: Unused default value.

        Returns:
            The wrapped value.
        """
        return self.value

    def map(self, fn: Any) -> Ok[Any]:
        """Apply fn to the value, returning a new Ok.

        Args:
            fn: Function to apply.

        Returns:
            Ok wrapping fn(value).
        """
        return Ok(value=fn(self.value), metadata=self.metadata)


@dataclass
class Err(Generic[E]):
    """Failed result wrapping an error.

    Attributes:
        error: The error description or exception.
        code: Optional error code for programmatic handling.
        metadata: Optional metadata for debugging.
    """

    error: E
    code: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        """Always False for Err."""
        return False

    @property
    def is_err(self) -> bool:
        """Always True for Err."""
        return True

    def unwrap(self) -> Any:
        """Raise RuntimeError (Err has no value to unwrap).

        Raises:
            RuntimeError: Always, with the error description.
        """
        msg = f"Called unwrap() on Err: {self.error}"
        raise RuntimeError(msg)

    def unwrap_or(self, default: Any) -> Any:
        """Return the default value (ignores error).

        Args:
            default: Value to return instead of the error.

        Returns:
            The default.
        """
        return default

    def map(self, fn: Any) -> Err[E]:
        """Return self unchanged (can't map over an error).

        Args:
            fn: Unused function.

        Returns:
            Self.
        """
        return self


Result = Ok[T] | Err[str]


@dataclass
class LLMCallError:
    """Structured error from an LLM API call.

    Attributes:
        kind: Error category (``rate_limit``, ``context_length``, ``auth``, etc.)
        message: Human-readable error description.
        retryable: True if the caller should retry.
        http_status: HTTP status code if available.
        request_id: The request ID that failed.
    """

    kind: str
    message: str
    retryable: bool = False
    http_status: int | None = None
    request_id: str = ""

    def __str__(self) -> str:
        return f"LLMCallError({self.kind}): {self.message}"


@dataclass
class BatchResult(Generic[T]):
    """Results from a batch of LLM calls.

    Attributes:
        results: List of individual results (Ok or Err).
        total_input_tokens: Sum of input tokens across successful calls.
        total_output_tokens: Sum of output tokens across successful calls.
        latency_ms: Wall-clock time for the entire batch.
    """

    results: list[Ok[T] | Err[str]]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def success_count(self) -> int:
        """Number of successful results."""
        return sum(1 for r in self.results if r.is_ok)

    @property
    def error_count(self) -> int:
        """Number of failed results."""
        return sum(1 for r in self.results if r.is_err)

    @property
    def success_rate(self) -> float:
        """Fraction of calls that succeeded (0.0-1.0)."""
        if not self.results:
            return 0.0
        return self.success_count / len(self.results)

    def values(self) -> list[T]:
        """Return only the successful values.

        Returns:
            List of unwrapped values from Ok results.
        """
        return [r.value for r in self.results if r.is_ok]  # type: ignore[union-attr]

    def errors(self) -> list[str]:
        """Return only the error descriptions.

        Returns:
            List of error strings from Err results.
        """
        return [r.error for r in self.results if r.is_err]  # type: ignore[union-attr]

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all successful calls."""
        return self.total_input_tokens + self.total_output_tokens
