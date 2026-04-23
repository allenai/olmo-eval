"""Extraction functions for handling thinking/reasoning traces"""

import re


# Based on oe-eval implementation; returns thinking trace if no answer is found
def extract_think_answer(text: str) -> str:
    # Deepseek-R1 style reasoning/answer extraction, assuming pattern is
    # <think>REASONING</think><answer>ANSWER</answer> with some flexibility
    # (mostly split on </think>, then remove other tags)
    answer = re.sub("(?ms).*</think>", "", text)
    answer = re.sub("(?ms)^\\s*<answer>\\s*", "", answer)
    answer = re.sub("(?ms)</answer>\\s*$", "", answer)
    return answer


# Based on the same extraction logic; returns None if no </think> tags are found
def extract_think_answer_only(text: str) -> str | None:
    # Deepseek-R1 style reasoning/answer extraction, assuming pattern is
    # <think>REASONING</think><answer>ANSWER</answer> with some flexibility
    # (mostly split on </think>, then remove other tags)
    reasoning = re.findall("(?ms)^(?:\\s*<think>\\s*)?(.*)\\s*</think>", text)
    if reasoning != []:
        answer = re.sub("(?ms).*</think>", "", text)
        answer = re.sub("(?ms)^\\s*<answer>\\s*", "", answer)
        answer = re.sub("(?ms)</answer>\\s*$", "", answer)
    else:
        return None
    return answer
