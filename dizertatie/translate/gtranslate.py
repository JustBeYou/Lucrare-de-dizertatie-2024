import html
import json
import pathlib
import time
from typing import List

from google.cloud import translate_v3
from google.oauth2 import service_account

from dizertatie.configs.common import PROJECT_SEED
from dizertatie.dataset.dataset import DatasetConfig, load
from dizertatie.translate.utils import prep_text

DATA_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath("data")
CREDENTIALS_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath(
    "data", "keys", "gcp-credentials.json"
)
PROJECT = "projects/dizertatie-414817"


def main():
    __translate_ro_text_summarization_long()

def __translate_ro_text_summarization_long():
    ro_text_summarization = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=8_000, path=DATA_PATH),
        "RoTextSummarization",
    )

    with open(
        DATA_PATH.joinpath("translations", "ro_text_summarization_ids.json"), "w"
    ) as f:
        json.dump(ro_text_summarization["id"], f)

    texts = list(map(prep_text, ro_text_summarization["text_ro"]))
    gcloud_translate(texts, "gcloud_ro_text_summarization_long/gcloud_ro_text_summarization_long.json", 1)


def __translate_ro_text_summarization_short():
    ro_text_summarization = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=8_000, path=DATA_PATH),
        "RoTextSummarization",
    )

    texts = list(map(prep_text, ro_text_summarization["target_ro"]))
    gcloud_translate(texts, "gcloud_ro_text_summarization_summary/gcloud_ro_text_summarization_summary.json", 10)


def __translate_rupert():
    rupert = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=5_000, path=DATA_PATH),
        "Rupert",
    )

    with open(DATA_PATH.joinpath("translations", "rupert_ids.json"), "w") as f:
        json.dump(rupert["id"], f)

    texts = list(map(prep_text, rupert["text_ro"]))
    gcloud_translate(texts[940:], "gcloud_rupert_2.json", 1)


def __translate_ro_sent():
    ro_sent = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=4_000, path=DATA_PATH),
        "RoSent",
    )

    with open(DATA_PATH.joinpath("translations", "ro_sent_ids.json"), "w") as f:
        json.dump(ro_sent["id"], f)

    texts = list(map(prep_text, ro_sent["text_ro"]))
    gcloud_translate(texts, "gcloud_ro_sent.json", 10)


def gcloud_translate(texts: List[str], output_file: str, batch_size: int):
    print(f"Translating for {output_file}")
    client = translate_v3.TranslationServiceClient(
        credentials=service_account.Credentials.from_service_account_file(
            str(CREDENTIALS_PATH)
        )
    )

    output_path = DATA_PATH.joinpath("translations", output_file)
    results = []

    try:
        for i in range(0, len(texts), batch_size):
            print(f"Processing {i+1}/{len(texts)}")
            batch = texts[i : i + batch_size]
            batch_result = __translate_batch(client, batch)
            results.extend(batch_result)
            time.sleep(0.1)
    finally:
        with open(output_path, "w") as f:
            json.dump(results, f)


def __translate_batch(
    client: translate_v3.TranslationServiceClient, texts: List[str]
) -> List[str]:
    request = translate_v3.TranslateTextRequest(
        {
            "contents": texts,
            "source_language_code": "ro",
            "target_language_code": "en",
            "parent": PROJECT,
        }
    )

    resp = client.translate_text(request)
    return [html.unescape(t.translated_text) for t in resp.translations]


if __name__ == "__main__":
    main()
