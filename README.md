# Comparing Modern LLM Quantization Methods Across Natural Languages

This repository contains the evaluation code and the relevant results for the paper titled "Comparing Modern LLM Quantization Methods Across Natural Languages" which 
This file will outline the general structure of the project and provide detailed instructions on how to use the code in this project.

### Abstract

Weight quantization has become a key tool for democratizing access to large language models (LLMs). Despite the technique's growing popularity and potential to aid speakers of diverse languages worldwide, new LLM quantization methods are predominantly validated in monolingual English contexts. This study explores ways to consistently evaluate the multilingual performance of a variety of LLaMA-based models under different quantization configurations. We identify links between the multilingual performance of widely adopted LLM quantization methods and multiple factors such as language's prevalence in the training set and similarity to model's dominant language.

## Project Structure

This project is organized as follows:

* The `eval_results` folder aggregates all of the relevant evaluation results from this study
* The `evaluate` folder contains the code relevant to the benchmarking of the quantization methods employed in this work
* The `visualize` folder contains a series of jupyter notebooks containing visualization code for the benchmarks
* The `quantize` folder contains scripts used to quantize the models
* The `quants` folder is meant to contain the quantized models, which are omitted due to their volume

Note that the results of the FLORES+ evaluation have been put into a password-protected zip archive, the password is the name of this repository, ie `llm_quant_across_langs`. This is a preventive measure to protect generated answers from getting scraped as that would constitute an inderect and noisy leak of FLORES+.

## Installation

All the code in this project was run on version 3.11.8 of Python, the main dependencies for this project are specified in the `requirements.txt` file and can accordingly be installed in a dedicated environment using:

```bash
pip install -r requirements.txt
```

## Usage Guide

This section outlines the general usage guide for various parts of this project

### Model Quantization

To quantize the models using the same configuration as in the experiments, simply run the `mass_quantize.sh` script from the `quantize` folder to add quantized models into the `quants` folder. If you do not yet have all the LLaMA models downloaded from Huggingface Hub, consider specifying your API key in the `hf_api_key` field in this script, or by using the `huggingface-cli login` command, as LLaMA models are gated behind a license you need to accept.

### Benchmarking

The FLORES+ BLEU evaluation can be performed by first running the `generate_flores_translations.py` script from within the `evaluate` folder as follows:

```bash
python3 generate_flores_translations.py -c CATEGORY -b BATCH_SIZE -n RUN_NAME -t HUGGINGFACE_TOKEN
```

The category parameter refer to the base model used for evaluation, either `7B` for `LLaMA 2 7B Chat`, `8B` for `LLaMA 3.1 8B Instruct` or `3B` for `LLaMA 3.2 3B Instruct`

Considering the updated mode of distribution of the FLORES+ dataset, you'll need to specify a valid huggingface token after agreeing to the terms on the [FLORES+ Huggingface Hub page](https://huggingface.co/datasets/openlanguagedata/flores_plus)  in order for the dataset to be downloaded.

Once the translations have been generated, they can be evaluated using either BLEU or COMET metric like follows, the results for each model will be stored in an appropriate subfolder of ´./eval_results/flores/´:

```bash
python3 eval_bleu.py -t HUGGINGFACE_TOKEN

python3 eval_comet.py -t HUGGINGFACE_TOKEN
```

The perplexity evaluation is done through the `perplexity.py` and `perplexity_normalized.py` scripts from within the `evaluate` folder, for regular sliding window perplexity and the language-adjusted approach respectively:

```bash
python3 perplexity.py -c CATEGORY -n RUN_NAME -t HUGGINGFACE_TOKEN
```

```bash
python3 perplexity_normalized.py -c CATEGORY -n RUN_NAME -t HUGGINGFACE_TOKEN
```

### Other Relevant Files

Other files of note include:

* `./evaluate/utils.py`: contains pseudo-quantization functions for round-to-nearest and mixed precision approaches