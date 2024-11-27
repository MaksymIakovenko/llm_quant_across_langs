import random
import argparse

import numpy as np
import torch
import torch.nn as nn

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--n_bit", type=int)
    parser.add_argument("-g", "--group_size", type=int, default=128)
    parser.add_argument("-s", "--src_dir", type=str)
    parser.add_argument("-d", "--dst_dir", type=str)
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-hf", "--hf_api_key", type=str, default=None)
    args = parser.parse_args()

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.n_bit,
        "version": "GEMM",
    }

    traindata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )["text"]

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        args.src_dir,
        device_map="auto",
        download_kwargs={
            "token": args.hf_api_key,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.src_dir,
        trust_remote_code=True,
        token=args.hf_api_key,
    )

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, calib_data=traindata)

    # Save quantized model
    model.save_quantized(args.dst_dir)
    tokenizer.save_pretrained(args.dst_dir)
