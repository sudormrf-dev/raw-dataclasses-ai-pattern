"""Dataclass-driven pipeline stages for LLM processing.

Each stage in an LLM pipeline is a typed dataclass transformation:
input → processed input → LLM call → processed output → final result.
No framework, no decorators — plain functions operating on plain dataclasses.

Pattern:
    PipelineInput → PreprocessStage → LLMCallStage → PostprocessStage → PipelineOutput

Usage::

    pipeline = Pipeline(stages=[
        PreprocessStage(system="You are a helpful assistant."),
        LLMCallStage(model="claude-opus-4-6"),
        PostprocessStage(extract_json=True),
    ])
    result = await pipeline.run(PipelineInput(user_text="Hello"))
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineInput:
    """Raw input to an LLM pipeline.

    Attributes:
        user_text: The user's message.
        context: Additional context to inject.
        metadata: Tracing/logging metadata.
        request_id: Unique identifier for this pipeline run.
    """

    user_text: str
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def with_context(self, **kwargs: Any) -> PipelineInput:
        """Return a copy with additional context merged in.

        Args:
            **kwargs: Key-value pairs to add to context.

        Returns:
            New :class:`PipelineInput` with updated context.
        """
        merged = {**self.context, **kwargs}
        return PipelineInput(
            user_text=self.user_text,
            context=merged,
            metadata=self.metadata,
            request_id=self.request_id,
        )


@dataclass
class ProcessedInput:
    """Input after preprocessing: ready for LLM consumption.

    Attributes:
        system_prompt: Final system prompt.
        messages: List of message dicts in API format.
        model: Target model identifier.
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature.
        request_id: Carried through from :class:`PipelineInput`.
    """

    system_prompt: str
    messages: list[dict[str, Any]]
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    request_id: str = ""
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawLLMResponse:
    """Raw response from an LLM API call (before post-processing).

    Attributes:
        content: Raw response text.
        stop_reason: Why the model stopped (``end_turn``, ``max_tokens``, etc.)
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens in the completion.
        model: Model that generated this response.
        latency_ms: Time from request to first token (or full response).
        request_id: Carried through from :class:`ProcessedInput`.
    """

    content: str
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0
    request_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Input + output tokens."""
        return self.input_tokens + self.output_tokens


@dataclass
class PipelineOutput:
    """Final output from a pipeline run.

    Attributes:
        text: Plain text response (always populated).
        parsed: Parsed structured output (e.g., JSON dict), if requested.
        success: Whether the pipeline completed without errors.
        error: Error description if success=False.
        latency_ms: End-to-end pipeline latency.
        usage: Token usage summary.
        request_id: Propagated from the input.
    """

    text: str
    parsed: dict[str, Any] | list[Any] | None = None
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str = ""

    @property
    def has_parsed_output(self) -> bool:
        """True if a structured parse of the output is available."""
        return self.parsed is not None


def _extract_json_from_text(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract a JSON object or array from LLM output text.

    Handles common formats:
    - Raw JSON
    - JSON wrapped in ```json ... ``` fences
    - JSON preceded by explanatory text

    Args:
        text: LLM response text potentially containing JSON.

    Returns:
        Parsed JSON value, or None if no valid JSON found.
    """
    def _cast(v: Any) -> dict[str, Any] | list[Any] | None:
        if isinstance(v, (dict, list)):
            return v
        return None

    # Try raw parse first
    try:
        return _cast(json.loads(text.strip()))
    except (json.JSONDecodeError, ValueError):
        pass

    # Try markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        try:
            return _cast(json.loads(fence_match.group(1).strip()))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding outermost { } or [ ] block
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        match = re.search(pattern, text)
        if match:
            try:
                return _cast(json.loads(match.group(0)))
            except (json.JSONDecodeError, ValueError):
                continue

    return None


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing stage.

    Attributes:
        system: System prompt template (use {key} for context interpolation).
        max_context_chars: Truncate context values to this length.
        prepend_context: If True, inject context as prefix in user message.
    """

    system: str = "You are a helpful assistant."
    model: str = "claude-opus-4-6"
    max_tokens: int = 1024
    temperature: float = 0.0
    max_context_chars: int = 2000
    prepend_context: bool = False


def preprocess(inp: PipelineInput, config: PreprocessConfig) -> ProcessedInput:
    """Transform a raw pipeline input into a processed input.

    Interpolates context into the system prompt, optionally prepends
    context to the user message, and assembles API-format messages.

    Args:
        inp: Raw pipeline input.
        config: Preprocessing configuration.

    Returns:
        :class:`ProcessedInput` ready for an LLM call.
    """
    # Interpolate context into system prompt
    try:
        system = config.system.format(**inp.context)
    except KeyError:
        system = config.system

    # Build user message
    user_text = inp.user_text
    if config.prepend_context and inp.context:
        context_lines = "\n".join(
            f"{k}: {str(v)[:config.max_context_chars]}" for k, v in inp.context.items()
        )
        user_text = f"Context:\n{context_lines}\n\n{inp.user_text}"

    return ProcessedInput(
        system_prompt=system,
        messages=[{"role": "user", "content": user_text}],
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        request_id=inp.request_id,
    )


@dataclass
class PostprocessConfig:
    """Configuration for the postprocessing stage.

    Attributes:
        extract_json: If True, attempt to parse JSON from the response.
        strip_thinking: If True, remove <thinking>...</thinking> blocks.
        max_output_chars: Truncate output to this length (0 = no limit).
    """

    extract_json: bool = False
    strip_thinking: bool = True
    max_output_chars: int = 0


def postprocess(
    response: RawLLMResponse,
    config: PostprocessConfig,
    pipeline_start: float,
) -> PipelineOutput:
    """Transform a raw LLM response into a pipeline output.

    Applies JSON extraction, thinking-block stripping, and length limits.

    Args:
        response: Raw LLM API response.
        config: Postprocessing configuration.
        pipeline_start: ``time.monotonic()`` when the pipeline started.

    Returns:
        :class:`PipelineOutput` with text and optional parsed output.
    """
    text = response.content

    # Strip thinking blocks
    if config.strip_thinking:
        text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE).strip()

    # Extract JSON
    parsed = None
    if config.extract_json:
        parsed = _extract_json_from_text(text)

    # Length limit
    if config.max_output_chars and len(text) > config.max_output_chars:
        text = text[: config.max_output_chars]

    return PipelineOutput(
        text=text,
        parsed=parsed,
        success=True,
        latency_ms=(time.monotonic() - pipeline_start) * 1000,
        usage={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        },
        request_id=response.request_id,
    )
