#!/bin/bash

models=('meta-llama/Llama-2-7b-chat-hf'
        'meta-llama/Llama-3.1-8B-Instruct'
        'meta-llama/Llama-3.2-3B-Instruct'
)

# Insert your Huggingface API key here if necessary
hf_api_key='hf_000000000000000'

for model in "${models[@]}"
do
    echo "gptq 4-bit base"
    python quantize_gptq.py -s $model -d ../quants/BASE/${model}_gptq_${prec}bit_128g -g 128 -b 4 -l en -hf $hf_api_key

    echo "awq 4-bit base"
    python quantize_awq.py -s $model -d ../quants/BASE/${model}_awq_${prec}bit_128g -g 128 -b 4 -l en -hf $hf_api_key
done