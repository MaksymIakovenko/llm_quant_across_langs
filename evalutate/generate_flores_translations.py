import os.path
import gc
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
import pandas as pd

from utils import load_model

BATCH_SIZE = 8
RUN_NAME = "base"

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
    "en": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.eng_Latn",
    "fr": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.fra_Latn",
    "ru": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.rus_Cyrl",
    "es": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.spa_Latn",
    "uk": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.ukr_Cyrl",
    "vi": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.vie_Latn",
    "id": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.ind_Latn",
    "hi": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.hin_Deva",
    "zh": f"../../Eval/floresp-v2.0-rc.3/devtest/devtest.cmn_Hans",
}
LANGS = [
    ("fr", "French"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("es", "Spanish"),
    ("vi", "Vietnamese"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("zh", "Chinese"),
]
SRC_LANG = ("en", "English")

sys_prompt = "You are a capable translation system. You only output the translation with no extra commentary."

main_prompt = (
    "Output the exact translation of the following text from {source} to {dest}:"
)


def produce_output(
    pipe,
    data,
    sys_prompt,
    main_prompt,
    src_lang_name,
    dst_lang_name,
):
    def datagen(data, sys_prompt, main_prompt, tokenizer):
        for line in data:
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": main_prompt.format(
                        source=src_lang_name, dest=dst_lang_name
                    )
                    + "\n"
                    + line,
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            prompt += "  Sure, here is the translation:"
            yield prompt

    # outputs = pipe(
    yield from pipe(
        datagen(data, sys_prompt, main_prompt, pipe.tokenizer),
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        # eos_token_id=pipe.model.config.eos_token_id,
        # required to suppress the warning messages
        pad_token_id=pipe.model.config.eos_token_id[0],
        return_full_text=False,
        batch_size=BATCH_SIZE,
    )

    # return outputs[0]["generated_text"]#[len(prompt) :]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--category", type=str, default="BASE"
    )  # either BASE or LN
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-n", "--run_name", type=str, default=RUN_NAME)
    args = parser.parse_args()

    match args.category:
        case "BASE":
            models = MODELS
        case "8B":
            models = MODELS_8B
        case "3B":
            models = MODELS_3B
        case _:
            raise ValueError("Invalid subset of models specified!")

    BATCH_SIZE = args.batch_size
    RUN_NAME = args.run_name

    tokenizer = AutoTokenizer.from_pretrained(
        models[0]["path"], padding_side="left"
    )  # We always use the same tokenizer anyways

    results = {}

    traindata = load_dataset("text", data_files=data_files)

    for quant in models:

        llama = load_model(quant)

        pipe = pipeline(
            "text-generation", model=llama, tokenizer=tokenizer, device_map="auto"
        )

        for lang, dst_name in LANGS:

            source = traindata[SRC_LANG[0]]["text"]
            dest = traindata[lang]["text"]

            to_translations = []
            from_translations = []

            for n, batch in tqdm(
                enumerate(
                    produce_output(
                        pipe, source, sys_prompt, main_prompt, SRC_LANG[1], dst_name
                    )
                ),
                total=len(dest),
            ):

                for response in batch:

                    response = response["generated_text"].replace("\n", " ")
                    to_translations.append(response)

            for n, batch in tqdm(
                enumerate(
                    produce_output(
                        pipe, dest, sys_prompt, main_prompt, dst_name, SRC_LANG[1]
                    )
                ),
                total=len(dest),
            ):

                for response in batch:

                    response = response["generated_text"].replace("\n", " ")
                    from_translations.append(response)

            df = pd.DataFrame(
                zip(to_translations, from_translations), columns=["to", "from"]
            )

            config_name = f"{quant['type']}_{quant['n_bit']}"
            if "subtype" in quant:
                config_name += "_" + quant["subtype"]
            os.makedirs(f"../eval_results/flores/{RUN_NAME}/{lang}/", exist_ok=True)
            df.to_csv(f"../eval_results/flores/{RUN_NAME}/{lang}/{config_name}.csv")

        llama.cpu()
        del llama
        gc.collect()
        torch.cuda.empty_cache()
