"""Lost in Conversation suites.

The paper reports each model's performance averaged over its task types, in the sharded
setting and with the fully specified instruction in one turn; the drop from
``lost_in_conversation:full`` to ``lost_in_conversation`` is its multi-turn degradation.
The database task is not implemented, so both averages are over the other five tasks.
"""

from olmo_eval.evals.suites.registry import make_suite

TASKS: tuple[str, ...] = (
    "lost_in_conversation_math",
    "lost_in_conversation_code",
    "lost_in_conversation_actions",
    "lost_in_conversation_data2text",
    "lost_in_conversation_summary",
)

make_suite(
    "lost_in_conversation",
    TASKS,
    description="Lost in Conversation, instructions revealed a shard per turn",
)
make_suite(
    "lost_in_conversation:full",
    tuple(f"{task}:full" for task in TASKS),
    description="Lost in Conversation, fully specified instructions in one turn",
)
