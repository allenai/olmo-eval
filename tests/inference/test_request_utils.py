"""Chat construction shared by the multimodal providers: image placement and system turns."""

from __future__ import annotations

import pytest
from jinja2 import Environment
from jinja2.exceptions import TemplateError

from olmo_eval.inference.request_utils import chat_with_images, render_chat

IMAGE = {"type": "image", "image": "img"}


def _raise(message: str):
    raise TemplateError(message)


def _apply(template: str):
    env = Environment()
    env.globals["raise_exception"] = _raise

    def apply(chat):
        return env.from_string(template).render(messages=chat)

    return apply


def _text(message) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(part["text"] for part in content if part["type"] == "text")


#: A generic template that never names the system role but renders every turn.
GENERIC = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}\n{% endfor %}"
#: Like Molmo2's: roles must alternate starting from a user turn.
ALTERNATING = (
    "{% for m in messages %}"
    "{% if (loop.index % 2 == 1 and m['role'] != 'user') %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/...') }}"
    "{% endif %}[{{ m['role'] }}]{% endfor %}"
)

CHAT = [
    {"role": "system", "content": [{"type": "text", "text": "Answer concisely."}]},
    {"role": "user", "content": [{"type": "text", "text": "Who won?"}]},
]


class TestRenderChat:
    def test_a_template_that_accepts_the_system_turn_renders_it(self):
        rendered = render_chat(_apply(GENERIC), CHAT)
        assert "[system]" in rendered
        assert "Answer concisely." in rendered

    def test_a_template_that_rejects_the_system_turn_gets_the_chat_without_it(self):
        assert render_chat(_apply(ALTERNATING), CHAT) == "[user]"

    def test_other_template_errors_still_raise(self):
        assistant_first = [{"role": "assistant", "content": "hi"}, CHAT[1]]
        with pytest.raises(TemplateError):
            render_chat(_apply(ALTERNATING), assistant_first)
        with pytest.raises(TemplateError):
            render_chat(_apply("{{ raise_exception('broken') }}"), CHAT[1:])


class TestChatWithImages:
    def test_images_go_where_the_message_places_them(self):
        messages = (
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "q"},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "instr"},
                ],
            },
        )
        chat = chat_with_images(messages, [dict(IMAGE, image="a"), dict(IMAGE, image="b")])
        assert chat[0] == {"role": "system", "content": [{"type": "text", "text": "sys"}]}
        assert [part.get("image") or part.get("text") for part in chat[1]["content"]] == [
            "q",
            "a",
            "b",
            "instr",
        ]

    def test_without_placeholders_images_lead_the_first_user_turn(self):
        chat = chat_with_images(({"role": "user", "content": "q"},), [IMAGE])
        assert chat == [{"role": "user", "content": [IMAGE, {"type": "text", "text": "q"}]}]

    def test_a_placeholder_count_that_does_not_match_raises(self):
        messages = ({"role": "user", "content": [{"type": "image"}, {"type": "image"}]},)
        with pytest.raises(ValueError):
            chat_with_images(messages, [IMAGE])
        with pytest.raises(ValueError):
            chat_with_images(({"role": "user", "content": [{"type": "image"}]},), [IMAGE, IMAGE])
