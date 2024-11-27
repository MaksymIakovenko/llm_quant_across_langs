import random
import argparse

import numpy as np
import torch
import torch.nn as nn

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from datasets import load_dataset
from transformers import AutoTokenizer

from tqdm import tqdm


# os.makedirs(quantized_model_dir, exist_ok=True)
def prepare_dataset(
    nsamples, seed, seqlen, model, path="wikitext", name="wikitext-2-raw-v1"
):

    traindata = load_dataset(path, name, split="train")

    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in tqdm(range(nsamples)):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset


def prepare_wiki40b(
    nsamples, seed, seqlen, model, path="../../Eval/wiki40b", name="default"
):

    traindata = load_dataset(path, name, split="train")

    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    # except Exception:
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(
        "\n\n".join(list(map(lambda x: eval(x).decode(), traindata["text"][:1000]))),
        return_tensors="pt",
    )

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in tqdm(range(nsamples)):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--n_bit", type=int)
    parser.add_argument("-g", "--group_size", type=int, default=128)
    parser.add_argument("-s", "--src_dir", type=str)
    parser.add_argument("-d", "--dst_dir", type=str)
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-hf", "--hf_api_key", type=str, default=None)
    args = parser.parse_args()

    quant_config = BaseQuantizeConfig(
        bits=args.n_bit,
        group_size=args.group_size,
        desc_act=False,
    )

    traindata = prepare_dataset(128, 0, 2048, args.src_dir)

    model = AutoGPTQForCausalLM.from_pretrained(
        args.src_dir,
        quant_config,
        device_map="auto",
        token=args.hf_api_key,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.src_dir,
        trust_remote_code=True,
        token=args.hf_api_key,
    )

    model.quantize(traindata, use_triton=False)

    model.save_quantized(args.dst_dir)
    tokenizer.save_pretrained(args.dst_dir)
