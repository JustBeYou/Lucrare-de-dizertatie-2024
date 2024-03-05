import json
import pathlib
import time
from typing import List

from openai import OpenAI

from dizertatie.configs.common import PROJECT_SEED
from dizertatie.dataset.dataset import DatasetConfig, load
from dizertatie.translate.utils import prep_text

DATA_PATH = pathlib.Path(__file__).parent.parent.parent.joinpath("data")


def main():
    raise Exception("Be careful, translation incurs costs.")


def __translate_ro_text_summarization_short():
    ro_text_summarization = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=8_000, path=DATA_PATH),
        "RoTextSummarization",
    )

    texts = list(map(prep_text, ro_text_summarization["target_ro"]))
    chatgpt_translate(texts, "chatgpt3.5_ro_text_summarization_summary.json")


def __translate_ro_text_summarization_long():
    ro_text_summarization = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=8_000, path=DATA_PATH),
        "RoTextSummarization",
    )

    with open(
        DATA_PATH.joinpath("translations", "ro_text_summarization_ids.json"), "w"
    ) as f:
        json.dump(ro_text_summarization["id"], f)

    texts = list(map(prep_text, ro_text_summarization["text_ro"][6768 + 1000 :]))
    chatgpt_translate(texts, "chatgpt3.5_ro_text_summarization_3.json")


def __translate_ro_sent():
    ro_sent = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=4_000, path=DATA_PATH),
        "RoSent",
    )

    texts = list(map(prep_text, ro_sent["text_ro"]))
    chatgpt_translate(texts, "chatgpt3.5_ro_sent.json")


def __translate_rupert():
    ro_sent = load(
        DatasetConfig(shuffle_seed=PROJECT_SEED, subsample_size=5_000, path=DATA_PATH),
        "Rupert",
    )

    texts = list(map(prep_text, ro_sent["text_ro"]))
    chatgpt_translate(texts, "chatgpt3.5_rupert.json")


def chatgpt_translate(texts: List[str], output_file: str):
    print(f"Translating for {output_file}")
    client = OpenAI()

    batch_idx = 0
    output_path = DATA_PATH.joinpath("translations", output_file)
    results = []

    try:
        for i, text in enumerate(texts):
            print(f"Processing {i+1}/{len(texts)}")
            results.append(__translate(client, text))
            # time.sleep(0.1)

            if len(results) > 500:
                with open(
                    str(output_path).replace(".json", f"_{batch_idx}.json"), "w"
                ) as f:
                    json.dump(results, f)
                    print(f"Saved batch {batch_idx}")

                batch_idx += 1

                results = []
    finally:
        with open(str(output_path).replace(".json", f"_{batch_idx}.json"), "w") as f:
            json.dump(results, f)
            print(f"Saved batch {batch_idx}")


def __translate(client: OpenAI, text: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """\
    You are a professional translator specialized in Romanian to English translations. 
    You will receive messages with the format 'Translate: <Romanian text>' and you will respond
    with Translation: <English text>.""",
            },
            {"role": "user", "content": f"Translate: {text}"},
        ],
    )

    pad = len("Translation: ")
    return completion.choices[0].message.content[pad:]


if __name__ == "__main__":
    main()
