"""Dataclass message types for LLM conversation pipelines.

Pure Python dataclasses covering the full conversation object model:
messages, tool calls, tool results, usage metadata, and streaming deltas.
No Pydantic, no validation overhead — just typed containers with helpers.

Usage::

    msg = UserMessage(content="What is 2+2?")
    turn = ConversationTurn(messages=[msg])
    print(turn.last_user_message)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Conversation participant roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, Enum):
    """Content block types for multimodal messages."""

    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"


@dataclass
class TextContent:
    """A plain text content block."""

    text: str
    type: str = ContentType.TEXT

    def __str__(self) -> str:
        return self.text


@dataclass
class ImageContent:
    """An image content block (base64 or URL)."""

    source: dict[str, Any]
    type: str = ContentType.IMAGE


@dataclass
class ToolUseContent:
    """A tool use request from the assistant."""

    id: str
    name: str
    input: dict[str, Any]
    type: str = ContentType.TOOL_USE


@dataclass
class ToolResultContent:
    """Result returned from a tool execution."""

    tool_use_id: str
    content: str | list[dict[str, Any]]
    type: str = ContentType.TOOL_RESULT
    is_error: bool = False


ContentBlock = TextContent | ImageContent | ToolUseContent | ToolResultContent


@dataclass
class Message:
    """A single message in a conversation.

    Args:
        role: Who sent this message.
        content: Single string (auto-wrapped) or list of content blocks.
        timestamp: Unix timestamp (auto-set on creation).
        metadata: Arbitrary metadata dict for tracing/logging.
    """

    role: Role
    content: str | list[ContentBlock]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return the plain text of this message."""
        if isinstance(self.content, str):
            return self.content
        parts = [block.text for block in self.content if isinstance(block, TextContent)]
        return " ".join(parts)

    @property
    def tool_uses(self) -> list[ToolUseContent]:
        """Return all tool use blocks in this message."""
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if isinstance(b, ToolUseContent)]

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to Anthropic Messages API format.

        Returns:
            Dict with ``role`` and ``content`` in API format.
        """
        role_str = self.role.value
        if isinstance(self.content, str):
            return {"role": role_str, "content": self.content}
        content_list = [asdict(block) for block in self.content]
        return {"role": role_str, "content": content_list}


def UserMessage(content: str | list[ContentBlock], **kwargs: Any) -> Message:
    """Convenience constructor for user messages.

    Args:
        content: Message content.
        **kwargs: Additional :class:`Message` fields.

    Returns:
        :class:`Message` with ``role=Role.USER``.
    """
    return Message(role=Role.USER, content=content, **kwargs)


def AssistantMessage(content: str | list[ContentBlock], **kwargs: Any) -> Message:
    """Convenience constructor for assistant messages.

    Args:
        content: Message content.
        **kwargs: Additional :class:`Message` fields.

    Returns:
        :class:`Message` with ``role=Role.ASSISTANT``.
    """
    return Message(role=Role.ASSISTANT, content=content, **kwargs)


def SystemMessage(content: str, **kwargs: Any) -> Message:
    """Convenience constructor for system messages.

    Args:
        content: System prompt text.
        **kwargs: Additional :class:`Message` fields.

    Returns:
        :class:`Message` with ``role=Role.SYSTEM``.
    """
    return Message(role=Role.SYSTEM, content=content, **kwargs)


@dataclass
class UsageMetadata:
    """Token usage from an LLM API response.

    Attributes:
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens in the completion.
        cache_creation_input_tokens: Tokens written to prompt cache.
        cache_read_input_tokens: Tokens read from prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def effective_input_tokens(self) -> int:
        """Input tokens minus cache hits (tokens we actually paid to process)."""
        return self.input_tokens - self.cache_read_input_tokens


@dataclass
class ConversationTurn:
    """A complete conversation with messages and metadata.

    Args:
        messages: Ordered list of messages.
        system: Optional system prompt (not included in messages list).
        model: Model identifier used for this conversation.
        usage: Cumulative token usage.
    """

    messages: list[Message] = field(default_factory=list)
    system: str = ""
    model: str = ""
    usage: UsageMetadata = field(default_factory=UsageMetadata)

    def add(self, message: Message) -> None:
        """Append a message to this conversation.

        Args:
            message: Message to add.
        """
        self.messages.append(message)

    def add_user(self, content: str) -> None:
        """Append a user message with text content.

        Args:
            content: Text to send as user.
        """
        self.messages.append(UserMessage(content))

    def add_assistant(self, content: str) -> None:
        """Append an assistant message with text content.

        Args:
            content: Text from assistant.
        """
        self.messages.append(AssistantMessage(content))

    @property
    def last_message(self) -> Message | None:
        """Return the most recent message, or None if empty."""
        return self.messages[-1] if self.messages else None

    @property
    def last_user_message(self) -> Message | None:
        """Return the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == Role.USER:
                return msg
        return None

    @property
    def last_assistant_message(self) -> Message | None:
        """Return the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.role == Role.ASSISTANT:
                return msg
        return None

    def to_api_messages(self) -> list[dict[str, Any]]:
        """Serialize all messages to Anthropic API format.

        Returns:
            List of message dicts ready for the API ``messages`` parameter.
        """
        return [m.to_api_dict() for m in self.messages if m.role != Role.SYSTEM]

    @property
    def message_count(self) -> int:
        """Number of messages in this conversation."""
        return len(self.messages)


@dataclass
class StreamDelta:
    """A single chunk from a streaming LLM response.

    Attributes:
        text: Incremental text content (may be empty for non-text events).
        stop_reason: Set on the final chunk when streaming ends.
        usage: Usage metadata (only present on the final chunk).
        tool_use_id: Set when this delta is part of a tool use block.
        tool_name: Tool name (first delta of a tool use block).
        tool_input_json: Partial JSON input for a tool call.
    """

    text: str = ""
    stop_reason: str | None = None
    usage: UsageMetadata | None = None
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input_json: str = ""

    @property
    def is_final(self) -> bool:
        """True if this is the last chunk in the stream."""
        return self.stop_reason is not None

    @property
    def is_tool_use(self) -> bool:
        """True if this delta contains tool use content."""
        return bool(self.tool_use_id)
