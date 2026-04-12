# raw-dataclasses-ai-pattern

Pure Python dataclass patterns for LLM pipelines. Zero dependencies, full type safety, copy-paste ready.

## Patterns

### `message_types.py` — Conversation object model
Typed containers for the full Anthropic Messages API: `Message`, `ConversationTurn`, `TextContent`, `ToolUseContent`, `UsageMetadata`, `StreamDelta`.

### `pipeline_stages.py` — Typed pipeline transformations
`PipelineInput → ProcessedInput → RawLLMResponse → PipelineOutput` with preprocessing (context interpolation, structured prompt building) and postprocessing (JSON extraction, thinking-block stripping).

### `result_types.py` — Explicit success/failure
`Ok[T]` / `Err[str]` result types. No silent `None` returns. `BatchResult` for parallel call batches with success rates and token aggregation.

## Usage

```python
from patterns.message_types import ConversationTurn, UserMessage
from patterns.pipeline_stages import PipelineInput, PreprocessConfig, preprocess
from patterns.result_types import Ok, Err

turn = ConversationTurn(model="claude-opus-4-6")
turn.add_user("What is 2+2?")

config = PreprocessConfig(system="You are a math tutor.")
processed = preprocess(PipelineInput("What is 2+2?"), config)
```

## License
MIT
