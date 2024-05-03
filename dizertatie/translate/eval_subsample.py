import dataclasses
import itertools
import json
import pathlib
import random
from pprint import pprint

import pandas

from dizertatie.configs.common import PROJECT_SEED, DATA_PATH
from dizertatie.dataset.dataset import DatasetConfig, load, load_translations


def generate_samples():
    data_path = DATA_PATH

    sizes = {
        'RoSent': 4_000,
        'Rupert': 5_000,
        'RoTextSummarization': 8_000
    }
    translations = {
        'RoTextSummarization': DATA_PATH.joinpath("translations/ro_text_summarization_subsample_translations.json"),
        'RoSent': DATA_PATH.joinpath("translations/ro_sent_subsample_translations.json"),
        'Rupert': DATA_PATH.joinpath("translations/rupert_subsample_translations.json"),
    }

    random.seed(PROJECT_SEED)

    M = 10

    translators = ["mistral", "chatgpt3.5", "deepl", "gcloud"]
    matchups = list(itertools.combinations(translators, 2))
    print(matchups)

    @dataclasses.dataclass
    class Match:
        id: int
        dataset: str
        dataset_row_id: int
        original_text_ro: str
        left_translator: str
        right_translator: str
        left_text_en: str
        right_text_en: str
        is_target: bool

        def to_json(self):
            """
                {
      "id": "41_rupert",
      "meta": {
        "left": "text_en_gcloud",
        "right": "text_en_deepl"
      },
      "original": "O, zei!\nE noapte t\u00e2rziu,\nTrozne\u0219te ciudat mobila calului troian\n\u00cen mijlocul cet\u0103\u021bii.\n\u0218i troienii, cuprin\u0219i de spaim\u0103,\nSe \u00eentreab\u0103:\nOare ce nenorocire\nNe mai pa\u0219te?\nC\u00e2nd trozne\u0219te f\u0103r\u0103 rost,\nCa o bomb\u0103 veche, uitat\u0103 din cel\u0103lalt r\u0103zboi,\nGarderoba calului troian\n\u00cen mijlocul cet\u0103\u021bii,\nNu este bine.\nO, zei!",
      "left": "Oh gods! It's late at night, The Trojan horse's furniture rattles strangely In the middle of the fortress. And the Trojans, seized with fear, Ask themselves: What misfortune awaits us? When it thunders pointlessly, Like an old bomb, forgotten from the other war, The wardrobe of the Trojan horse In the middle of the fortress, It is not good. Oh gods!",
      "right": "Oh, gods!\nIt's late at night,\nStrange trots the Trojan horse's furniture\nIn the midst of the city\nAnd the Trojans, in terror,\nThey wonder:\nWhat misfortune\nIs there more to come?\nWhen it thunders uselessly,\nLike an old bomb, forgotten from another war,\nThe Trojan horse's wardrobe\nIn the midst of the city\nIt's no good\nO gods!"
    }
            """
            return {
                "id": self.id,
                "meta": {
                    "left": self.left_translator,
                    "right": self.right_translator,
                    "dataset": self.dataset,
                    "row_id": self.dataset_row_id,
                    "is_target": self.is_target
                },
                "original": self.original_text_ro,
                "left": self.left_text_en,
                "right": self.right_text_en
            }

    matches = []
    match_id = 0


    for dataset_name, dataset_size in sizes.items():
        dataset_translation_config = DatasetConfig(
            subsample_size=dataset_size,
            shuffle_seed=PROJECT_SEED,
            path=data_path,
        )
        dataset = load(dataset_translation_config, dataset_name)
        dataset = load_translations(dataset, translations[dataset_name], translators)
        print(dataset)

        for match in matchups:
            samples = list(range(dataset_size))
            samples = random.choices(samples, k = M)

            for sample in samples:
                if random.random() < 0.5:
                    match = match[::-1]

                matches.append(Match(
                    id=match_id,
                    dataset=dataset_name,
                    dataset_row_id=sample,
                    original_text_ro=dataset[sample]['text_ro'],
                    left_translator=match[0],
                    left_text_en=dataset[sample][f'text_en_{match[0]}'],
                    right_translator=match[1],
                    right_text_en=dataset[sample][f'text_en_{match[1]}'],
                    is_target=False
                ))
                match_id += 1

                if 'target_ro' in dataset.column_names:
                    matches.append(Match(
                        id=match_id,
                        dataset=dataset_name,
                        dataset_row_id=sample,
                        original_text_ro=dataset[sample]['target_ro'],
                        left_translator=match[0],
                        left_text_en=dataset[sample][f'target_en_{match[0]}'],
                        right_translator=match[1],
                        right_text_en=dataset[sample][f'target_en_{match[1]}'],
                        is_target=True
                    ))
                    match_id += 1

    random.shuffle(matches)
    print(len(matches))
    json.dump({
        "version": 3,
        "texts": list(map(lambda x: x.to_json(), matches))
    }, open("evaluation_samples.json", "w"))


if __name__ == "__main__":
    generate_samples()