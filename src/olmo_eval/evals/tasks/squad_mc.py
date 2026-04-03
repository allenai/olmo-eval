from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import MultipleChoiceFormatter
from olmo_eval.common.metrics import BPBMetric, LogprobMCAccuracyMetric, LogprobPerCharMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType, SamplingParams, Split
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# fmt: off
SQUAD_MC_FIXED_FEWSHOT = [
    {
        "id": "squad_mc_format_fewshot_0",
        "choices": {
            "text": [
                "Saint Thomas Aquinas",
                "Saint Bernadette Soubirous",
                "Saint Francis of Assisi",
                "Saint Joan of Arc",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "title_original": "University_of_Notre_Dame",
        "context_original": 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
        "question_original": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    },
    {
        "id": "squad_mc_format_fewshot_1",
        "choices": {"text": ["2003", "1995", "1999", "2010"], "label": ["A", "B", "C", "D"]},
        "answerKey": "A",
        "title_original": "Beyonc\u00e9",
        "context_original": 'Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".',
        "question_original": "When did Beyonce leave Destiny's Child and become a solo singer?",
    },
    {
        "id": "squad_mc_format_fewshot_2",
        "choices": {"text": ["10th", "25th", "50th", "4th"], "label": ["A", "B", "C", "D"]},
        "answerKey": "D",
        "title_original": "Montana",
        "context_original": 'Montana i/m\u0252n\u02c8t\u00e6n\u0259/ is a state in the Western region of the United States. The state\'s name is derived from the Spanish word monta\u00f1a (mountain). Montana has several nicknames, although none official, including "Big Sky Country" and "The Treasure State", and slogans that include "Land of the Shining Mountains" and more recently "The Last Best Place". Montana is ranked 4th in size, but 44th in population and 48th in population density of the 50 United States. The western third of Montana contains numerous mountain ranges. Smaller island ranges are found throughout the state. In total, 77 named ranges are part of the Rocky Mountains.',
        "question_original": "What is the states rank in size?",
    },
    {
        "id": "squad_mc_format_fewshot_3",
        "choices": {
            "text": [
                "That political destruction was sufficient",
                "That economic destruction was necessary",
                "That biological-physical destruction was necessary",
                "That cultural destruction was sufficient",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "title_original": "Genocide",
        "context_original": 'In the same judgement the ECHR reviewed the judgements of several international and municipal courts judgements. It noted that International Criminal Tribunal for the Former Yugoslavia and the International Court of Justice had agreed with the narrow interpretation, that biological-physical destruction was necessary for an act to qualify as genocide. The ECHR also noted that at the time of its judgement, apart from courts in Germany which had taken a broad view, that there had been few cases of genocide under other Convention States municipal laws and that "There are no reported cases in which the courts of these States have defined the type of group destruction the perpetrator must have intended in order to be found guilty of genocide".',
        "question_original": "Two bodies of the United Nations agreed with what restricted provision in defining genocide?",
    },
    {
        "id": "squad_mc_format_fewshot_4",
        "choices": {
            "text": [
                "Ciprofloxacin and vancomycin",
                "Penicillin and erythromycin",
                "Azithromycin and doxycycline",
                "Amoxicillin and tetracycline",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "title_original": "Antibiotics",
        "context_original": "The emergence of resistance of bacteria to antibiotics is a common phenomenon. Emergence of resistance often reflects evolutionary processes that take place during antibiotic therapy. The antibiotic treatment may select for bacterial strains with physiologically or genetically enhanced capacity to survive high doses of antibiotics. Under certain conditions, it may result in preferential growth of resistant bacteria, while growth of susceptible bacteria is inhibited by the drug. For example, antibacterial selection for strains having previously acquired antibacterial-resistance genes was demonstrated in 1943 by the Luria\u2013Delbr\u00fcck experiment. Antibiotics such as penicillin and erythromycin, which used to have a high efficacy against many bacterial species and strains, have become less effective, due to the increased resistance of many bacterial strains.",
        "question_original": "Which two antibiotics that have high efficacy are much less useful now?",
    },
    {
        "id": "squad_mc_format_fewshot_5",
        "choices": {
            "text": ["Warsaw", "Paris", "Krak\u00f3w", "Vienna"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "title_original": "Fr\u00e9d\u00e9ric_Chopin",
        "context_original": 'Fr\u00e9d\u00e9ric Fran\u00e7ois Chopin (/\u02c8\u0283o\u028ap\u00e6n/; French pronunciation: \u200b[f\u0281e.de.\u0281ik f\u0281\u0251\u0303.swa \u0283\u0254.p\u025b\u0303]; 22 February or 1 March 1810 \u2013 17 October 1849), born Fryderyk Franciszek Chopin,[n 1] was a Polish and French (by citizenship and birth of father) composer and a virtuoso pianist of the Romantic era, who wrote primarily for the solo piano. He gained and has maintained renown worldwide as one of the leading musicians of his era, whose "poetic genius was based on a professional technique that was without equal in his generation." Chopin was born in what was then the Duchy of Warsaw, and grew up in Warsaw, which after 1815 became part of Congress Poland. A child prodigy, he completed his musical education and composed his earlier works in Warsaw before leaving Poland at the age of 20, less than a month before the outbreak of the November 1830 Uprising.',
        "question_original": "Where did Chopin grow up?",
    },
    {
        "id": "squad_mc_format_fewshot_6",
        "choices": {
            "text": [
                "European historians",
                "Ming dynasty emperors",
                "Tibetan monks",
                "Mainland Chinese scholars",
            ],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
        "title_original": "Sino-Tibetan_relations_during_the_Ming_dynasty",
        "context_original": "The exact nature of relations between Tibet and the Ming dynasty of China (1368\u20131644) is unclear. Analysis of the relationship is further complicated by modern political conflicts and the application of Westphalian sovereignty to a time when the concept did not exist. Some Mainland Chinese scholars, such as Wang Jiawei and Nyima Gyaincain, assert that the Ming dynasty had unquestioned sovereignty over Tibet, pointing to the Ming court's issuing of various titles to Tibetan leaders, Tibetans' full acceptance of these titles, and a renewal process for successors of these titles that involved traveling to the Ming capital. Scholars within China also argue that Tibet has been an integral part of China since the 13th century and that it was thus a part of the Ming Empire. But most scholars outside China, such as Turrell V. Wylie, Melvin C. Goldstein, and Helmut Hoffman, say that the relationship was one of suzerainty, that Ming titles were only nominal, that Tibet remained an independent region outside Ming control, and that it simply paid tribute until the Jiajing Emperor (1521\u20131566), who ceased relations with Tibet.",
        "question_original": "Who were Wang Jiawei and Nyima Gyaincain?",
    },
    {
        "id": "squad_mc_format_fewshot_7",
        "choices": {
            "text": ["Sony", "Microsoft", "Apple", "Samsung"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "C",
        "title_original": "IPod",
        "context_original": "The iPod is a line of portable media players and multi-purpose pocket computers designed and marketed by Apple Inc. The first line was released on October 23, 2001, about 8\u00bd months after iTunes (Macintosh version) was released. The most recent iPod redesigns were announced on July 15, 2015. There are three current versions of the iPod: the ultra-compact iPod Shuffle, the compact iPod Nano and the touchscreen iPod Touch.",
        "question_original": "Which company produces the iPod?",
    },
    {
        "id": "squad_mc_format_fewshot_8",
        "choices": {
            "text": ["November 2006", "November 2005", "October 2006", "December 2006"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "A",
        "title_original": "The_Legend_of_Zelda:_Twilight_Princess",
        "context_original": "The Legend of Zelda: Twilight Princess (Japanese: \u30bc\u30eb\u30c0\u306e\u4f1d\u8aac \u30c8\u30ef\u30a4\u30e9\u30a4\u30c8\u30d7\u30ea\u30f3\u30bb\u30b9, Hepburn: Zeruda no Densetsu: Towairaito Purinsesu?) is an action-adventure game developed and published by Nintendo for the GameCube and Wii home video game consoles. It is the thirteenth installment in the The Legend of Zelda series. Originally planned for release on the GameCube in November 2005, Twilight Princess was delayed by Nintendo to allow its developers to refine the game, add more content, and port it to the Wii. The Wii version was released alongside the console in North America in November 2006, and in Japan, Europe, and Australia the following month. The GameCube version was released worldwide in December 2006.[b]",
        "question_original": "When was Twilight Princess launched in North America?",
    },
    {
        "id": "squad_mc_format_fewshot_9",
        "choices": {
            "text": ["Thirty", "Twenty-four", "Fifteen", "Twenty"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "title_original": "Spectre_(2015_film)",
        "context_original": "Spectre (2015) is the twenty-fourth James Bond film produced by Eon Productions. It features Daniel Craig in his fourth performance as James Bond, and Christoph Waltz as Ernst Stavro Blofeld, with the film marking the character's re-introduction into the series. It was directed by Sam Mendes as his second James Bond film following Skyfall, and was written by John Logan, Neal Purvis, Robert Wade and Jez Butterworth. It is distributed by Metro-Goldwyn-Mayer and Columbia Pictures. With a budget around $245 million, it is the most expensive Bond film and one of the most expensive films ever made.",
        "question_original": "How many James Bond films has Eon Productions produced?",
    },
]
# fmt: on


def _process_squad_mc_doc(doc: dict[str, Any], index: int) -> Instance | None:
    title = doc.get("title_original", "")
    passage = doc.get("context_original", "").strip()
    question = doc.get("question_original", "")
    choices_data = doc.get("choices", {})
    choices = choices_data.get("text", [])
    answer_key = doc.get("answerKey", "")

    if not question or not choices:
        return None

    gold_idx = ord(answer_key) - ord("A") if answer_key else 0
    gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

    return Instance(
        question=f"Title: {title}\nPassage: {passage}\nQuestion: {question}",
        choices=tuple(choices),
        gold_answer=answer_key,
        metadata={
            "id": doc.get("id", f"squad_mc_{index}"),
            "index": index,
            "dataset": "squad_mc",
            "gold_idx": gold_idx,
            "gold_text": gold_text,
        },
    )


def _build_squad_mc_fixed_fewshot(
    raw_docs: list[dict[str, Any]], num_fewshot: int, seed: int
) -> list[Instance]:
    instances = []
    for doc in raw_docs:
        title = doc["title_original"]
        passage = doc["context_original"].strip()
        question = doc["question_original"]
        choices = tuple(doc["choices"]["text"])
        answer_key = doc["answerKey"]
        gold_idx = ord(answer_key) - ord("A")
        gold_text = choices[gold_idx] if 0 <= gold_idx < len(choices) else ""

        instances.append(
            Instance(
                question=f"Title: {title}\nPassage: {passage}\nQuestion: {question}",
                choices=choices,
                gold_answer=gold_text,
                metadata={
                    "gold_idx": gold_idx,
                    "gold_text": gold_text,
                    "mc_answer": answer_key,
                },
            )
        )

    if num_fewshot and num_fewshot < len(instances):
        instances = instances[:num_fewshot]
    return instances


def _format_mc(question: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"{question}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def _format_rc(question: str, answer: str | None = None) -> str:
    prompt = f"{question}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


class _SquadMCBase(Task):
    metrics = (LogprobMCAccuracyMetric(),)
    num_fewshot = 5
    fewshot_split = "train"
    sampling_params = SamplingParams(temperature=0.0)
    _fewshot_source_name = "squad_mc_fixed"

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        return _process_squad_mc_doc(doc, index)

    def _build_fewshot(self) -> list[Instance]:
        if getattr(self.config, "fewshot_source", None) == self._fewshot_source_name:
            return _build_squad_mc_fixed_fewshot(
                SQUAD_MC_FIXED_FEWSHOT, self.config.num_fewshot, self.config.fewshot_seed
            )
        return super()._build_fewshot()

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()
        is_mc = self.config.formatter is not None

        parts: list[str] = []
        for ex in fewshot:
            if is_mc:
                answer = ex.metadata.get("mc_answer", "")
                parts.append(_format_mc(ex.question, ex.choices or (), answer))
            else:
                answer = ex.gold_answer or ex.metadata.get("gold_text", "")
                parts.append(_format_rc(ex.question, answer))

        if is_mc:
            parts.append(_format_mc(instance.question, instance.choices or ()))
            continuations = tuple(
                f" {chr(ord('A') + i)}" for i in range(len(instance.choices or ()))
            )
        else:
            parts.append(_format_rc(instance.question))
            continuations = tuple(f" {c}" for c in (instance.choices or ()))

        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=continuations,
        )


@register("squad:mc")
class SquadMC(_SquadMCBase):
    data_source = DataSource(path="allenai/squad_mc", split="validation")
    split = Split.VALIDATION
    formatter = MultipleChoiceFormatter()
    fewshot_source = "squad_mc_fixed"


@register("squad:rc")
class SquadRC(_SquadMCBase):
    data_source = DataSource(path="allenai/squad_mc", split="validation")
    split = Split.VALIDATION
    metrics = (LogprobPerCharMCAccuracyMetric(),)
    fewshot_source = "squad_mc_fixed"


register_variant(
    "squad:mc",
    "olmo3base",
    limit=10_000,
    seed=1234,
    fewshot_source="squad_mc_fixed",
)

register_variant(
    "squad:rc",
    "olmo3base",
    limit=10_000,
    seed=1234,
    fewshot_source="squad_mc_fixed",
)


@register("squad:bpb")
class SquadBPB(_SquadMCBase):
    data_source = DataSource(path="allenai/squad_mc", split="validation")
    split = Split.VALIDATION
    metrics = (BPBMetric(),)
    fewshot_source = "squad_mc_fixed"

    def format_request(self, instance: Instance) -> LMRequest:
        fewshot = self.get_fewshot()

        # Use RC-style formatting to match oe-eval-internal's SquadRC BPB computation.
        # Fewshot examples: "Title: ...\nQuestion: ...\nAnswer: <gold_text>"
        # Test prompt: "Title: ...\nQuestion: ...\nAnswer:"
        # Continuation: only the gold answer text (for BPB).
        parts: list[str] = []
        for ex in fewshot:
            answer = ex.gold_answer or ex.metadata.get("gold_text", "")
            parts.append(_format_rc(ex.question, answer))

        parts.append(_format_rc(instance.question))

        gold_idx = instance.metadata.get("gold_idx", 0)
        if instance.choices and 0 <= gold_idx < len(instance.choices):
            gold_text = instance.choices[gold_idx]
        else:
            gold_text = instance.gold_answer or ""

        prompt = "\n\n".join(parts)
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=prompt,
            continuations=(f" {gold_text}",),
        )


register_variant(
    "squad:bpb",
    "olmo3base",
    limit=10_000,
    seed=1234,
    fewshot_source="squad_mc_fixed",
)
