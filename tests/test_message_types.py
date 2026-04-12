"""Tests for message_types.py."""

from __future__ import annotations

from patterns.message_types import (
    AssistantMessage,
    ConversationTurn,
    Role,
    StreamDelta,
    SystemMessage,
    TextContent,
    ToolUseContent,
    UsageMetadata,
    UserMessage,
)


class TestMessage:
    def test_user_message_role(self):
        msg = UserMessage("hello")
        assert msg.role == Role.USER

    def test_assistant_message_role(self):
        msg = AssistantMessage("hi there")
        assert msg.role == Role.ASSISTANT

    def test_system_message_role(self):
        msg = SystemMessage("You are helpful.")
        assert msg.role == Role.SYSTEM

    def test_text_property_from_string(self):
        msg = UserMessage("hello world")
        assert msg.text == "hello world"

    def test_text_property_from_blocks(self):
        blocks = [TextContent("part one"), TextContent("part two")]
        msg = UserMessage(blocks)
        assert "part one" in msg.text
        assert "part two" in msg.text

    def test_tool_uses_empty_for_string_content(self):
        msg = UserMessage("no tools here")
        assert msg.tool_uses == []

    def test_tool_uses_returns_tool_blocks(self):
        blocks = [
            TextContent("let me search"),
            ToolUseContent(id="tu_1", name="search", input={"query": "foo"}),
        ]
        msg = AssistantMessage(blocks)
        assert len(msg.tool_uses) == 1
        assert msg.tool_uses[0].name == "search"

    def test_to_api_dict_string_content(self):
        msg = UserMessage("hello")
        d = msg.to_api_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_to_api_dict_block_content(self):
        blocks = [TextContent("hi")]
        msg = AssistantMessage(blocks)
        d = msg.to_api_dict()
        assert d["role"] == "assistant"
        assert isinstance(d["content"], list)

    def test_timestamp_set_automatically(self):
        msg = UserMessage("test")
        assert msg.timestamp > 0


class TestUsageMetadata:
    def test_total_tokens(self):
        usage = UsageMetadata(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_effective_input_tokens(self):
        usage = UsageMetadata(input_tokens=100, cache_read_input_tokens=30)
        assert usage.effective_input_tokens == 70

    def test_defaults_zero(self):
        usage = UsageMetadata()
        assert usage.total_tokens == 0


class TestConversationTurn:
    def test_add_user_message(self):
        turn = ConversationTurn()
        turn.add_user("hello")
        assert turn.message_count == 1
        assert turn.messages[0].role == Role.USER

    def test_last_message(self):
        turn = ConversationTurn()
        turn.add_user("q1")
        turn.add_assistant("a1")
        assert turn.last_message.role == Role.ASSISTANT

    def test_last_user_message(self):
        turn = ConversationTurn()
        turn.add_user("q1")
        turn.add_assistant("a1")
        turn.add_user("q2")
        assert turn.last_user_message.text == "q2"

    def test_last_assistant_message(self):
        turn = ConversationTurn()
        turn.add_user("q1")
        turn.add_assistant("a1")
        turn.add_user("q2")
        assert turn.last_assistant_message.text == "a1"

    def test_last_message_empty(self):
        turn = ConversationTurn()
        assert turn.last_message is None

    def test_last_user_message_none_when_absent(self):
        turn = ConversationTurn()
        turn.add_assistant("hi")
        assert turn.last_user_message is None

    def test_to_api_messages_excludes_system(self):
        turn = ConversationTurn()
        turn.add(SystemMessage("system"))
        turn.add_user("user")
        api_msgs = turn.to_api_messages()
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"

    def test_message_count(self):
        turn = ConversationTurn()
        for _ in range(5):
            turn.add_user("x")
        assert turn.message_count == 5


class TestStreamDelta:
    def test_is_final_false(self):
        delta = StreamDelta(text="hello")
        assert not delta.is_final

    def test_is_final_true(self):
        delta = StreamDelta(text="", stop_reason="end_turn")
        assert delta.is_final

    def test_is_tool_use_false(self):
        delta = StreamDelta(text="hello")
        assert not delta.is_tool_use

    def test_is_tool_use_true(self):
        delta = StreamDelta(tool_use_id="tu_1", tool_name="search")
        assert delta.is_tool_use
