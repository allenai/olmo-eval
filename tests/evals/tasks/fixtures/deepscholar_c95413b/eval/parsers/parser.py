from abc import ABC, abstractmethod

import pandas as pd

from .parser_type import ParserType


class Parser(ABC):
    parser_type: ParserType

    def __init__(self, folder_path: str, config: dict):
        self.config = config
        self.folder_path = folder_path
        self.file_id = (
            self.config["file_id"]
            if "file_id" in self.config
            else self.folder_path.split("/")[-1]
        )
        self.use_local_reference_map = self.config.get(
            "use_local_reference_map",
            True,
        )
        self.raw_generated_text: str | None = None
        self.clean_text: str | None = None
        self.docs: list[dict[str, str]] | None = None
        self.citations_for_cite_quality: list[tuple[str, str]] | None = None
        self._load_dataset()
        self._load_file()

    def _load_dataset(self):
        if "s_map_groundtruth" in self.config:
            self.s_map_groundtruth = self.config["s_map_groundtruth"]
        else:
            if "dataset" in self.config:
                dataset = self.config["dataset"]
            else:
                dataset = pd.read_csv(self.config["dataset_path"])
            row = dataset.iloc[int(self.file_id)]
            self.s_map_groundtruth = {
                "title": row["title"],
                "abstract": row["abstract"],
                "arxiv_link": row["arxiv_link"],
                "related_works_section": row.get(
                    "clean_latex_related_works",
                    "",
                ),
                "arxiv_id": row["arxiv_id"],
            }

    @abstractmethod
    def _load_file(self):
        pass
