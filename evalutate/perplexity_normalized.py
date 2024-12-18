import json
import os.path
import gc
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

from utils import load_model


device = "cuda"
# torch.cuda.set_device(device)


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
        "path": "../quants/BASE/meta-llama/Llama-3.1-8B-Instruct_awq_4bit_128g",
        "type": "awq",
        "n_bit": 4,
    },
    {
        "path": "../quants/BASE/meta-llama/Llama-3.1-8B-Instruct_gptq_4bit_128g",
        "type": "gptq",
        "n_bit": 4,
    },
]

MODELS_3B = [
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "hf", "n_bit": 4},
    {"path": "meta-llama/Llama-3.2-3B-Instruct", "type": "rtn", "n_bit": 4},
    {
        "path": "../quants/BASE/meta-llama/Llama-3.2-3B-Instruct_awq_4bit_128g",
        "type": "awq",
        "n_bit": 4,
    },
    {
        "path": "../quants/BASE/meta-llama/Llama-3.2-3B-Instruct_gptq_4bit_128g",
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

res_path = "../eval_results/perplexity/"
RUN_NAME = "norm"

resume = False


# Modified code based on sample implementation from https://huggingface.co/docs/transformers/perplexity
def eval_ppl(model, lang_encodings, base_sizes):

    max_length = 1024  # sliding window
    # max_length = model.config.max_position_embeddings
    normalized_nlls = []

    seq = []

    for n, sentence in tqdm(enumerate(lang_encodings.input_ids)):
        seq += sentence[
            1:
        ]  # first token is always the sequence start token, repeating it may confuse the model
        ids = torch.asarray(seq[-max_length:]).unsqueeze(0)

        begin_loc = -(len(sentence) + 1)

        input_ids = ids.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :begin_loc] = (
            -100
        )  # Ignore elements from outside the current sentence

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            neg_log_likelihood = outputs.loss

        # The idea here is to normalize the overall weight of a given sentence to be
        # the same as the weight of its English counterpart
        normalized_nlls.append(neg_log_likelihood * (len(sentence) / base_sizes[n]))

    ppl = torch.exp(torch.stack(normalized_nlls).mean())

    return ppl


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", type=str, default="7B")  # either BASE or LN
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

    resume = args.resume

    RUN_NAME = args.run_name

    res_path = res_path + f"/{args.category}/"
    res_name = res_path + RUN_NAME

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

    for quant in models:

        en_encodings = tokenizer(traindata["en"]["text"])
        base_lengths = [len(enc) for enc in en_encodings.input_ids]

        llama = load_model(quant, token=args.token)

        for lang in LANGS:
            encodings = tokenizer(traindata[lang]["text"])

            ppl = eval_ppl(llama, encodings, base_sizes=base_lengths).cpu().item()

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
