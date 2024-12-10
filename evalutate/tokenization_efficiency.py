import argparse
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

MODELS = [
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-3.1-8B-Instruct", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "hf", "n_bit": 16},
]


data_files = {
    "en": "dev/eng_Latn.parquet",
    "fr": "dev/fra_Latn.parquet",
    "ru": "dev/rus_Cyrl.parquet",
    "es": "dev/spa_Latn.parquet",
    "uk": "dev/ukr_Cyrl.parquet",
    "vi": "dev/vie_Latn.parquet",
    "id": "dev/ind_Latn.parquet",
    "hi": "dev/hin_Deva.parquet",
    "zh": "dev/cmn_Hans.parquet",
}

LANGS = data_files.keys()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--token", type=str, default="hf_" + "0" * 34)
    args = parser.parse_args()

    models = MODELS

    loader = lambda path, code: Dataset.from_parquet(
        f"hf://datasets/openlanguagedata/flores_plus/{path}",
        split=code,
        token=args.token,
    )

    datasets = {code: loader(path, code) for code, path in data_files.items()}
    traindata = DatasetDict(datasets)

    for quant in models:

        tokenizer = AutoTokenizer.from_pretrained(quant["path"], token=args.token)

        l = {}

        for lang in LANGS:

            final = "\n\n".join(
                traindata[lang]["text"][: min(10000, len(traindata[lang]["text"]))]
            )

            encodings = tokenizer(final, return_tensors="pt")

            print(f"[{lang}] token length: {len(encodings[0])}")
            l[lang] = len(encodings[0])

        print(quant)
        print(l)
