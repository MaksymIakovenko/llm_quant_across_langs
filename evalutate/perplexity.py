import json
import os.path
import gc
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

from utils import load_model


MODELS = [
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "hf", "n_bit": 4},
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "rtn", "n_bit": 4},
    {"path": "../quants/BASE/llama_7b_awq_4bit_128g", "type": "awq", "n_bit": 4},
    {"path": "../quants/BASE/llama_7b_gptq_4bit_128g", "type": "gptq", "n_bit": 4},
]

MODELS_8B = [
    {"path": "meta-llama/Llama-3.1-8B-Instruct", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-3.1-8B-Instruct", "type": "hf", "n_bit": 4},
    {"path": "meta-llama/Llama-3.1-8B-Instruct", "type": "rtn", "n_bit": 4},
    {
        "path": "../../quants/BASE/meta-llama/Llama-3.1-8B-Instruct_awq_4bit_128g",
        "type": "awq",
        "n_bit": 4,
    },
    {
        "path": "../../quants/BASE/meta-llama/Llama-3.1-8B-Instruct_gptq_4bit_128g",
        "type": "gptq",
        "n_bit": 4,
    },
]

MODELS_3B = [
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "hf", "n_bit": 4},
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "rtn", "n_bit": 4},
    {
        "path": "../../quants/BASE/meta-llama/Llama-3.2-3B-Instruct_awq_4bit_128g",
        "type": "awq",
        "n_bit": 4,
    },
    {
        "path": "../../quants/BASE/meta-llama/Llama-3.2-3B-Instruct_gptq_4bit_128g",
        "type": "gptq",
        "n_bit": 4,
    },
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

# Only used for smaller test runs
limit = 10

device = "cuda"
res_path = "../eval_results/perplexity/"

RUN_NAME = "base__"

resume = False


# Code based on sample implementation from https://huggingface.co/docs/transformers/perplexity
def eval_ppl(model, encodings):
    max_length = min(
        model.config.max_position_embeddings, 4096
    )  # Limiting the context size because GPU-poor
    stride = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str, default="7B")
    parser.add_argument("-n", "--run_name", type=str, default=RUN_NAME)
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("-t", "--token", type=str, default="hf_" + "0" * 34)

    args = parser.parse_args()

    match args.category:
        case "7B":
            models = MODELS
        case "8B":
            models = MODELS_8B
        case "3B":
            models = MODELS_3B
        case _:
            raise ValueError("Invalid subset of models specified!")
    RUN_NAME = args.run_name

    res_path = res_path + f"/{args.category}/"
    res_name = res_path + RUN_NAME

    resume = args.resume

    tokenizer = AutoTokenizer.from_pretrained(
        models[0]["path"]
    )  # We always use the same tokenizer anyways

    if resume:
        with open(f"{res_name}.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}
        for l in LANGS:
            results[l] = {}

    loader = lambda path, code: Dataset.from_parquet(
        f"hf://datasets/openlanguagedata/flores_plus/{path}",
        split=code,
        token=args.token,
    )

    datasets = {code: loader(path, code) for code, path in data_files.items()}
    traindata = DatasetDict(datasets)

    l = []

    for quant in models:

        llama = load_model(quant)

        for lang in LANGS:
            # if lang in ("fr", "ru"):
            #     final = "\n\n".join(
            #         map(lambda x: eval(x).decode(), traindata[lang]["text"][:LIMIT])
            #     )
            # else:

            final = "\n\n".join(
                traindata[lang]["text"][: min(limit, len(traindata[lang]["text"]))]
            )

            encodings = tokenizer(final, return_tensors="pt")

            print(f"[{lang}] token length: {len(encodings[0])}")
            l.append(len(encodings[0]))

            ppl = eval_ppl(llama, encodings).cpu().item()

            config_name = f"{quant['type']} {quant['n_bit']}"
            if "subtype" in quant:
                config_name += " " + quant["subtype"]
            results[lang][config_name] = {
                "model": os.path.basename(quant["path"]),
                "method": quant["type"],
                "precision": quant["n_bit"],
                "perplexity": ppl,
            }

        os.makedirs(res_path, exist_ok=True)
        with open(f"{res_name}.json", "w+", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        llama.cpu()
        del llama
        gc.collect()
        torch.cuda.empty_cache()
