"""Prompts and answer descriptions from the Lost in Conversation reference implementation.

Copied verbatim from microsoft/lost_in_conversation at
c865793fe34a929d316119b0451d01bd9183bcfd (MIT License, copyright Microsoft Corporation): the
prompt files named above each constant, and the answer descriptions and follow-up message
written in ``tasks/<task>/task_<task>.py``. The prompts keep the files' typos ("absoltutely",
"substringz") and their ``[[PLACEHOLDER]]`` markers, which the reference fills with
``str.replace``.
"""

# prompts/system_turn_categorization.txt
TURN_CATEGORIZATION_PROMPT = """\
You are reviewing a multi-turn conversation between a user and an assistant, and are given the \
last turn of the conversation.

Here is the full specification of the problem the system is attempting to solve:
[[INITIAL_SHARD]]

Specification:
[[SHARDS]]

You must classify the response of the assistant according to the response type:
- `answer_attempt`: The response contains a complete answer attempt to the user's question (not \
templated or hypothetical), that can be extracted verbatim. See the task-specific answer \
description for more details.
- `clarification`: The response  is short (less than 100 words) and contains a single question \
addressed to the user that directly inquires about an aspect of the user's query. A \
clarification turn cannot be long (see `discussion`), cannot contain a vague question (see \
`discussion`) and cannot contain multiple questions (see `interrogation`).
- `interrogation`: The response contains multiple questions addressed to the user, sometimes \
organized in a list or bullet-points.
- `discussion`: The response discusses the question in detail, without providing a final \
answer, asking a specific clarification question, or a refusal to answer. The response may or \
may not contain a vague question (e.g., “What else can I help you with?”).
- `hedge`: The response contains multiple answer candidates based on hypotheticals (ifs) or \
branching (case 1, case 2) with corresponding descriptions.
- `refuse`: The response contains an explicit or implicit refusal to answer the user's question \
without a follow-up question or a request.
- `missing`: The response is empty/blank.

You must output your answer in the following JSON format:
{"response_type": "refuse|missing|answer_attempt|hedge|clarification|interrogation|discussion"}

Rules:
- The assistant giving a hint at how an answer could look like is not a final answer. You \
should only select `answer_attempt` if the conversation could end at this stage with the user \
having an entirely final answer to the problem they've formulated.
- [Task Specific Answer] [[ANSWER_DESCRIPTION]]

Conversation's last turn:
[[CONVERSATION_SO_FAR]]"""

# prompts/system_answer_extraction_gen.txt
ANSWER_EXTRACTION_PROMPT = """\
You are reviewing a response from an AI assistant to a user query.
Your goal is to extract verbatim what the answer is. The answer to the user query is expected \
to be: [[ANSWER_DESCRIPTION]]

Rules:
- [Exact Answer Only] only extract the exact answer, and nothing else (including ``` for code \
blocks, or intro/outro text).
- [Verbatim Only] Only extract verbatim text, do not modify the text in any way. If there's a \
typo, an error, you must absoltutely include it, and not correct it in any way. The program \
will check that the string you extract is exactly present as a substringz in the passage \
provided.
- [String output] the <answer_str> must be a string, not a number and not a dictionary.

You must output your answer in the following JSON format:
{"answer": "<answer_str>"}

Response:
[[ASSISTANT_RESPONSE]]"""

# prompts/summary/summhay_evaluation.txt
SUMMARY_COVERAGE_PROMPT = """\
You are given a list of bullet points (each with a unique number), and a specific reference \
insight. Your objective is to determine whether the reference insight is covered in any of the \
bullet points. You must further determine if the insight is partially covered \
("PARTIAL_COVERAGE") or fully covered ("FULL_COVERAGE") by the bullet points. If the insight is \
not covered at all, you must return "NO_COVERAGE". See examples below:

Example Reference Insight 1: "The doctor asks the patient about their medical history".

Example Bullet Points 1:
{"bullets": [
    {"bullet_id": 1, "text": "The patient often mention that they are worried about medication \
side-effect[12][14]."},
    {"bullet_id": 2, "text": "The doctor and patient spend time going over symptoms [3][8], \
particularly the initial symptoms and the progression in the last few months[5]."},
    {"bullet_id": 3, "text": "The doctor and patient discuss medical history within the \
patient's family[1][5], with the patient often unaware that some of the conditions are \
hereditary[2]."}
]
}

Example Output 1:
{"coverage": "FULL_COVERAGE", "bullet_id": 3}

Example Reference Insight 2: "The doctor asks the patient about their medical history".

Example Bullet Points 2:
{"bullets": [
    {"bullet_id": 1, "text": "The patient often mention that they are worried about medication \
side-effect[12][14]."},
    {"bullet_id": 2, "text": "The doctor and patient spend time going over symptoms [3][8], \
particularly the initial symptoms and the progression in the last few months[5]."}
]
}

Example Output 2:
{"coverage": "NO_COVERAGE", "bullet_id": "NA"}

Example Reference Insight 3: "The doctor asks the patient about their medical history".

Example Bullet Points 3:
{"bullets": [
    {"bullet_id": 1, "text": "The patient often mention that they are worried about medication \
side-effect[12][14]."},
    {"bullet_id": 2, "text": "The doctor and patient catch up after a long time, with the \
patient mentioning feeling unwell for a while, and knowing of other family member's similar \
experiences[20]."},
    {"bullet_id": 3, "text": "The doctor and patient spend time going over symptoms [3][8], \
particularly the initial symptoms and the progression in the last few months[5]."}
]
}

Example Output 3:
{"coverage": "PARTIAL_COVERAGE", "bullet_id": 2}

Now complete the task for the following insight and bullet points:

Reference Insight:
[[INSIGHT]]

Bullet Points:
[[BULLETS]]


Requirements:
- Do not hallucinate that the insight is covered by the bullet points if it is not.
- Your response should only be the JSON output in the format above, such that it can directly \
parsed by Python's json module. DO NOT OUTPUT ANY EXPLANATION OR ANYTHING THAT IS NOT THE JSON \
RESPONSE."""

# prompts/math/math_system_prompt.txt
MATH_SYSTEM_PROMPT = """\
As an expert problem solver solve step by step the following mathematical questions."""

# prompts/math/math_full_prompt.txt
MATH_FULL_PROMPT = """\
Q: [[QUESTION]]
A:"""

# prompts/lcb/lcb_system_prompt.txt
CODE_SYSTEM_PROMPT = """\
You are an expert Python programmer. You will be given a question (problem specification) and \
will generate a correct Python program that matches the specification and passes all tests.

Format:
- [Standalone] Make sure that your answer consists of only one Python function at the top \
level. Do not wrap with a class or split into multiple functions."""

# prompts/lcb/lcb_full_prompt.txt
LCB_FULL_PROMPT = """\
### Question:
[[QUESTION]]

### Format: [[FORMATTING_PREAMBLE]]
[[FORMATTING]]

### Answer: (use the provided format with backticks)
"""

# prompts/humaneval/humaneval_full_prompt.txt
HUMANEVAL_FULL_PROMPT = """\
You must generate a valid Python function that satisfies the following instruction:

Instruction:

[[INSTRUCTION]]

Rules:
- [Format] You must only return the Python function, matching the signature exactly.
- [Validity] The Python function must be valid and executable. Do not include any other text or \
comments. (```python```)"""

# prompts/actions/actions_system_prompt.txt
ACTIONS_SYSTEM_PROMPT = """\
You are an expert in composing functions. You are given a question and a set of possible \
functions. Based on the question, you will need to make one or more function/tool calls to \
achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters \
required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of \
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.

[[FUNCTIONS]]"""

# prompts/actions/actions_full_prompt.txt
ACTIONS_FULL_PROMPT = """\
[[QUESTION]]"""

# prompts/data2text/data2text_system_prompt.txt
DATA2TEXT_SYSTEM_PROMPT = """\
You are an analyst with an eye for detail that accomplishes tasks carefully and thoroughly."""

# prompts/data2text/data2text_full_prompt.txt
DATA2TEXT_FULL_PROMPT = """\
Please generate a one-sentence description of the following table. Focus on the highlighted \
cells if present.

Table:
[[TABLE_HTML]]

Consider the following context:
[[CONTEXT]]

Here are some examples of descriptions of other tables. You can use it as references to \
understand how to describe the table:
[[FEWSHOT_DESCRIPTIONS]]

Generate a clear and natural one-sentence description."""

# prompts/summary/summary_system_prompt.txt
SUMMARY_SYSTEM_PROMPT = """\
You are a professional document analyst. You receive documents and are tasked with writing \
specific summaries of the document, which you do carefully."""

# prompts/summary/summary_full_prompt_conv.txt
SUMMARY_FULL_PROMPT_CONV = """\
You are given [[N_DOCS]] conversations within the following scenario: "[[TOPIC]]".
In each conversation, the participants might be different, with different names, but all the \
conversations fall into the same scenario.

```
[[DOCUMENTS]]
```

Your objective is to produce a summary of all the document with [[N_INSIGHTS]] bullet points \
covering the main insights regarding the following query: [[QUERY]].
Careful:
- [Format] You should format your summary as a bullet point list, where each bullet point is a \
different insight.
- [References] For each insight, you should refer to the relevant conversations by using the \
IDs given in the conversation list. For example: "some of the doctors ask patients for their \
medical history[1][4]" which means that the insight is supported by Document 1 and Document 4. \
No need to say "Conversation 14", you can just use the following structure: "[14]".
- [Length] Your summary should be no longer than 300 words in total."""

# prompts/summary/summary_full_prompt_news.txt
SUMMARY_FULL_PROMPT_NEWS = """\
You are given [[N_DOCS]] news articles about the main subject "[[TOPIC]]."

```
[[DOCUMENTS]]
```

In some of the conversation the following topic is discussed: "[[QUERY]]".
Your objective is to summarize the [[N_INSIGHTS]] main insights from the conversations \
regarding that topic.
Careful:
- [Format] You should format your summary as a bullet point list, where each bullet point is a \
different insight consisting of a single sentence.
- [References] For each insight, you should refer to the relevant articles by using the IDs \
given in the article list. For example: "Increased demand for vaccines may strain already \
fragile supply chains [1][4]" which means that the insight is supported by Document 1 and \
Document 4. No need to say "Article 14", you can just use the following structure: "[14]".
- [Length] Your summary should be no longer than 300 words in total."""

# Each task's get_answer_description(), which fills [[ANSWER_DESCRIPTION]]
MATH_ANSWER_DESCRIPTION = (
    "The answer should be a single number (it could be decimal, or negative, or a fraction, etc.)."
)
CODE_ANSWER_DESCRIPTION = (
    "A final answer must be a valid Python function that is defined (def ...) and returns "
    "something. The function should handle multiple test cases according to the problem "
    "specification. If the answer is just a natural language explanation, it is not a valid "
    "answer. Special symbols like newlines and quotes must be escaped."
)
ACTIONS_ANSWER_DESCRIPTION = (
    "The answer should be a series of valid function calls in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), "
    "func_name2(params)]. An answer may contain multiple function calls."
)
DATA2TEXT_ANSWER_DESCRIPTION = "A one-sentence description of the table."
SUMMARY_ANSWER_DESCRIPTION = (
    "A complete summary potentially containing multiple lines, and citation."
)

# tasks/code/task_code.py's formatting preamble for LiveCodeBench problems
LCB_FORMATTING_PREAMBLE = (
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters."
)

# tasks/summary/task_summary.py's message for every summary turn after the first
SUMMARY_FOLLOW_UP_PROMPT = (
    "I have found a few additional documents, please rewrite the summary considering all "
    "documents so far (from before, and the new ones).\n\n"
)
