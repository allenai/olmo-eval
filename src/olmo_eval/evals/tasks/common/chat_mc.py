"""Shared configuration for deterministic chat-native multiple-choice tasks."""

from olmo_eval.common.formatters import MultipleChoiceChatFormatter
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.types import SamplingParams
from olmo_eval.evals.extract import extract_mcq_answer

CHAT_MC_SYSTEM_PROMPT = (
    "Answer the multiple-choice question. Reason carefully, then end your response with "
    '"ANSWER: X", where X is the letter of the best answer.'
)
CHAT_MC_FORMATTER = MultipleChoiceChatFormatter(system_prompt=CHAT_MC_SYSTEM_PROMPT)
CHAT_MC_ACCURACY = AccuracyMetric(scorer=MultipleChoiceScorer)
CHAT_MC_SAMPLING = SamplingParams(max_tokens=4096, temperature=0.0)
CHAT_MC_ANSWER_EXTRACTOR = extract_mcq_answer
