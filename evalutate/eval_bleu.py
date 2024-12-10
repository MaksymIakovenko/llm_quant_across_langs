import pandas as pd
from datasets import Dataset, DatasetDict
import math
import os
import argparse
from sacrebleu.metrics import BLEU

srcs = [
    "../eval_results/flores/7B",
    "../eval_results/flores/8B",
    "../eval_results/flores/3B",
]


model_names = [
    "LLaMA 2 7B Chat",
    "LLaMA 3.1 8B Instruct",
    "LLaMA 3.2 3B Instruct",
]


quants = [
    ("hf", "16"),
    ("hf", "4"),
    ("rtn", "4"),
    ("awq", "4"),
    ("gptq", "4"),
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--token", type=str, default="hf_" + "0" * 34
    )  # either BASE or LN
    args = parser.parse_args()

    data_files = {
        "en": "devtest/eng_Latn.parquet",
        "fr": "devtest/fra_Latn.parquet",
        "ru": "devtest/rus_Cyrl.parquet",
        "es": "devtest/spa_Latn.parquet",
        "uk": "devtest/ukr_Cyrl.parquet",
        "vi": "devtest/vie_Latn.parquet",
        "id": "devtest/ind_Latn.parquet",
        "hi": "devtest/hin_Deva.parquet",
        "zh": "devtest/cmn_Hans.parquet",
    }

    loader = lambda path, code: Dataset.from_parquet(
        f"hf://datasets/openlanguagedata/flores_plus/{path}",
        split=code,
        token=args.token,
    )

    datasets = {code: loader(path, code) for code, path in data_files.items()}
    reference_data = DatasetDict(datasets)

    langs = ["fr", "ru", "uk", "es", "vi", "id", "zh", "hi"]

    bleu = BLEU(tokenize="flores101")

    for src in srcs:
        if not os.path.exists(f"{src}/BLEU.csv"):
            bleu_summary_avg = None
            columns = ["quant", "language", "BLEU", "direction"]

            count = 0

            for method, precision in quants:
                ref_reverse = reference_data["en"]["text"]
                for lang in langs:
                    data = pd.read_csv(
                        f"{src}/{lang}/{method}_{precision}.csv", index_col=0
                    )

                    ref = reference_data[lang]["text"]

                    list_to = data["to"].to_list()
                    list_from = data["from"].to_list()

                    for n in range(len(ref_reverse)):
                        # print(n, lang, method, precision)

                        if (type(list_to[n]) == float) and math.isnan(list_to[n]):
                            bleu_to = 0
                        else:
                            bleu_to = bleu.corpus_score([list_to[n]], [[ref[n]]]).score

                        if (type(list_from[n]) == float) and math.isnan(list_from[n]):
                            bleu_from = 0
                        else:
                            bleu_from = bleu.corpus_score(
                                [list_from[n]], [[ref_reverse[n]]]
                            ).score

                        line = pd.DataFrame(
                            [
                                [f"{method} {precision}bit", lang, bleu_to, "to"],
                                [f"{method} {precision}bit", lang, bleu_from, "from"],
                            ],
                            columns=columns,
                        )

                        if bleu_summary_avg is None:
                            bleu_summary_avg = line
                        else:
                            bleu_summary_avg = pd.concat(
                                [bleu_summary_avg, line], axis=0, ignore_index=True
                            )

            bleu_summary_avg.to_csv(f"{src}/BLEU.csv")
