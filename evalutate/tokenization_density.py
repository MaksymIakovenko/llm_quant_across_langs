from transformers import AutoTokenizer
from datasets import load_dataset

MODELS = [
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-3.1-8B-Instruct", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "hf", "n_bit": 16},
]


data_files = {
    "en": f"../../Eval/floresp-v2.0-rc.3/dev/dev.eng_Latn",
    "fr": f"../../Eval/floresp-v2.0-rc.3/dev/dev.fra_Latn",
    "ru": f"../../Eval/floresp-v2.0-rc.3/dev/dev.rus_Cyrl",
    "uk": f"../../Eval/floresp-v2.0-rc.3/dev/dev.ukr_Cyrl",
    "es": f"../../Eval/floresp-v2.0-rc.3/dev/dev.spa_Latn",
    "vi": f"../../Eval/floresp-v2.0-rc.3/dev/dev.vie_Latn",
    "id": f"../../Eval/floresp-v2.0-rc.3/dev/dev.ind_Latn",
    "hi": f"../../Eval/floresp-v2.0-rc.3/dev/dev.hin_Deva",
    "zh": f"../../Eval/floresp-v2.0-rc.3/dev/dev.cmn_Hans",
}
LANGS = data_files.keys()

# Only used for smaller test runs
limit = 10000

device = "cuda"

RUN_NAME = "base"

resume = False

if __name__ == "__main__":

    models = MODELS

    traindata = load_dataset("text", data_files=data_files)

    for quant in models:

        tokenizer = AutoTokenizer.from_pretrained(quant["path"])

        l = {}

        for lang in LANGS:

            final = "\n\n".join(
                traindata[lang]["text"][: min(limit, len(traindata[lang]["text"]))]
            )

            encodings = tokenizer(final, return_tensors="pt")

            print(f"[{lang}] token length: {len(encodings[0])}")
            l[lang] = len(encodings[0])

        print(quant)
        print(l)

    # with open(f"{res_name}.json", "w+", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)
