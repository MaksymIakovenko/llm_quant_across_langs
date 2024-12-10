import pandas as pd
from datasets import Dataset, DatasetDict
import math
import os
import argparse

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

    for src in srcs:
        if not os.path.exists(f"{src}/COMET.csv"):

            from comet import download_model, load_from_checkpoint

            model_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(model_path)

            comet_summary_avg = None
            columns = ["quant", "language", "COMET", "direction"]

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

                    list_to = [
                        (
                            " "
                            if ((type(list_to[n]) == float) and math.isnan(list_to[n]))
                            else list_to[n]
                        )
                        for n in range(len(list_to))
                    ]
                    list_from = [
                        (
                            " "
                            if (
                                (type(list_from[n]) == float)
                                and math.isnan(list_from[n])
                            )
                            else list_from[n]
                        )
                        for n in range(len(list_from))
                    ]

                    triplet_to = [
                        {"src": ref_reverse[n], "mt": list_to[n], "ref": ref[n]}
                        for n in range(len(list_to))
                    ]

                    triplet_from = [
                        {"src": ref[n], "mt": list_from[n], "ref": ref_reverse[n]}
                        for n in range(len(list_from))
                    ]

                    print("a")
                    to_scores = comet_model.predict(triplet_to, batch_size=4).scores
                    print("b")
                    from_scores = comet_model.predict(triplet_from, batch_size=4).scores

                    for n in range(len(ref_reverse)):

                        line = pd.DataFrame(
                            [
                                [f"{method} {precision}bit", lang, to_scores[n], "to"],
                                [
                                    f"{method} {precision}bit",
                                    lang,
                                    from_scores[n],
                                    "from",
                                ],
                            ],
                            columns=columns,
                        )

                        if comet_summary_avg is None:
                            comet_summary_avg = line
                        else:
                            comet_summary_avg = pd.concat(
                                [comet_summary_avg, line], axis=0, ignore_index=True
                            )

            comet_summary_avg.to_csv(f"{src}/COMET.csv")
