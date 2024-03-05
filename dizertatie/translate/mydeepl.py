import html
import json
import pathlib
import time
from typing import List

import deepl
from google.cloud import translate_v3
from google.oauth2 import service_account
from openai import OpenAI

from dizertatie.configs.common import PROJECT_SEED
from dizertatie.dataset.dataset import DatasetConfig, load
from dizertatie.translate.utils import prep_text

DATA_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath("data")


def main():
    raise Exception("Be careful, translation incurs costs.")


def __translate_rupert():
    rupert = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=5_000, path=DATA_PATH),
        "Rupert",
    )

    with open(DATA_PATH.joinpath("translations", "rupert_ids.json"), "w") as f:
        json.dump(rupert["id"], f)

    texts = list(map(prep_text, rupert["text_ro"]))[230:]
    deepl_translate(texts, "deepl_rupert_2.json", 10)


def __translate_ro_sent():
    ro_sent = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=4_000, path=DATA_PATH),
        "RoSent",
    )

    with open(DATA_PATH.joinpath("translations", "ro_sent_ids.json"), "w") as f:
        json.dump(ro_sent["id"], f)

    texts = list(map(prep_text, ro_sent["text_ro"]))
    deepl_translate(texts, "deepl_ro_sent.json", 10)


auth_key = ""
translator = deepl.Translator(auth_key)


def deepl_translate(texts: List[str], output_file: str, batch_size: int):
    print(f"Translating for {output_file}")

    for i in range(0, len(texts), batch_size):
        print(f"Processing {i+1}/{len(texts)}")
        batch = texts[i : i + batch_size]
        batch_result = __translate_batch(batch)

        b_id = i // batch_size

        output_path = str(DATA_PATH.joinpath("translations", output_file)).replace(
            ".json", f"_{b_id}.json"
        )
        with open(output_path, "w") as f:
            json.dump(batch_result, f)
            print(f"Saved batch {b_id}")

        time.sleep(0.1)


def __translate_batch(texts: List[str]):
    r = translator.translate_text(texts, source_lang="RO", target_lang="EN-US")
    return list(map(str, r))


if __name__ == "__main__":
    main()
