"""Tests for pipeline_stages.py."""

from __future__ import annotations

import time

from patterns.pipeline_stages import (
    PipelineInput,
    PostprocessConfig,
    PreprocessConfig,
    RawLLMResponse,
    _extract_json_from_text,
    postprocess,
    preprocess,
)


class TestPipelineInput:
    def test_with_context_merges(self):
        inp = PipelineInput("hello", context={"lang": "fr"})
        updated = inp.with_context(user="alice")
        assert updated.context["lang"] == "fr"
        assert updated.context["user"] == "alice"

    def test_with_context_does_not_mutate_original(self):
        inp = PipelineInput("hello", context={"a": 1})
        inp.with_context(b=2)
        assert "b" not in inp.context


class TestPreprocess:
    def test_basic_preprocessing(self):
        config = PreprocessConfig(system="Be helpful.", model="claude-opus-4-6")
        inp = PipelineInput("Hello!")
        result = preprocess(inp, config)
        assert result.system_prompt == "Be helpful."
        assert result.messages[0]["content"] == "Hello!"
        assert result.model == "claude-opus-4-6"

    def test_context_interpolation(self):
        config = PreprocessConfig(system="You are {name}.")
        inp = PipelineInput("Hi", context={"name": "Alice"})
        result = preprocess(inp, config)
        assert result.system_prompt == "You are Alice."

    def test_missing_context_key_graceful(self):
        config = PreprocessConfig(system="Hello {missing_key}.")
        inp = PipelineInput("Hi", context={})
        result = preprocess(inp, config)
        # Falls back to original system prompt
        assert "{missing_key}" in result.system_prompt

    def test_prepend_context_adds_to_message(self):
        config = PreprocessConfig(prepend_context=True)
        inp = PipelineInput("What should I do?", context={"task": "write tests"})
        result = preprocess(inp, config)
        assert "task" in result.messages[0]["content"]

    def test_request_id_propagated(self):
        config = PreprocessConfig()
        inp = PipelineInput("Hi", request_id="req-123")
        result = preprocess(inp, config)
        assert result.request_id == "req-123"


class TestPostprocess:
    def _make_response(self, content: str, **kwargs: object) -> RawLLMResponse:
        return RawLLMResponse(
            content=content,
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
            **kwargs,
        )

    def test_plain_text_passthrough(self):
        config = PostprocessConfig()
        resp = self._make_response("Hello there!")
        output = postprocess(resp, config, time.monotonic())
        assert output.text == "Hello there!"
        assert output.success is True

    def test_thinking_block_stripped(self):
        config = PostprocessConfig(strip_thinking=True)
        resp = self._make_response("<thinking>internal reasoning</thinking>\nFinal answer.")
        output = postprocess(resp, config, time.monotonic())
        assert "<thinking>" not in output.text
        assert "Final answer." in output.text

    def test_thinking_not_stripped_when_disabled(self):
        config = PostprocessConfig(strip_thinking=False)
        resp = self._make_response("<thinking>keep this</thinking>")
        output = postprocess(resp, config, time.monotonic())
        assert "<thinking>" in output.text

    def test_json_extracted(self):
        config = PostprocessConfig(extract_json=True)
        resp = self._make_response('{"name": "Alice", "age": 30}')
        output = postprocess(resp, config, time.monotonic())
        assert output.parsed is not None
        assert output.parsed["name"] == "Alice"

    def test_json_extraction_fails_gracefully(self):
        config = PostprocessConfig(extract_json=True)
        resp = self._make_response("Not JSON at all.")
        output = postprocess(resp, config, time.monotonic())
        assert output.parsed is None
        assert output.success is True

    def test_max_length_truncates(self):
        config = PostprocessConfig(max_output_chars=10)
        resp = self._make_response("a" * 100)
        output = postprocess(resp, config, time.monotonic())
        assert len(output.text) == 10

    def test_usage_populated(self):
        config = PostprocessConfig()
        resp = RawLLMResponse(content="OK", stop_reason="end_turn", input_tokens=200, output_tokens=30)
        output = postprocess(resp, config, time.monotonic())
        assert output.usage["input_tokens"] == 200
        assert output.usage["output_tokens"] == 30
        assert output.usage["total_tokens"] == 230

    def test_latency_ms_set(self):
        config = PostprocessConfig()
        start = time.monotonic()
        resp = self._make_response("OK")
        output = postprocess(resp, config, start)
        assert output.latency_ms >= 0


class TestExtractJsonFromText:
    def test_raw_json_object(self):
        result = _extract_json_from_text('{"key": "value"}')
        assert result == {"key": "value"}

    def test_raw_json_array(self):
        result = _extract_json_from_text("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_json_in_code_fence(self):
        text = '```json\n{"name": "Alice"}\n```'
        result = _extract_json_from_text(text)
        assert result == {"name": "Alice"}

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"score": 42} as requested.'
        result = _extract_json_from_text(text)
        assert result == {"score": 42}

    def test_no_json_returns_none(self):
        result = _extract_json_from_text("no JSON here at all")
        assert result is None
