# Minimalist-Llama
This is an implementation of a minimalist version of Llama2. This is offered to the students of the LUH [Advanced NLP course](https://sites.google.com/view/jen-web/sose-2025). A walkthrough of the code offers deeper insights into the programmatic implementation of various components of the transformer architecture based on that implemented for LLaMA 2.

In this assignment, you will implement some important components of the Llama2 model to better understand its architecture. 

The solution will be released on May 12, 2025. It will contain all the code for the unimplemented functions which you can use to compare your solutions with. 
 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with this model.

## Assignment Details

### Your task
The code to implement can be found in `rope.py`, `optimizer.py`, `llama.py`, and `classifier.py`. You are reponsible for writing _core components_ of Llama2 (one of the leading open source language models). In doing so, you will gain a strong understanding of neural language modeling. We will load pretrained weights for your language model from `stories42M.pt`; an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (a dataset of machine-generated children's stories). This model is small enough that it can be trained (slowly) without a GPU. You are encouraged to use Colab or a personal GPU machine (e.g. a Macbook) to be able to iterate more quickly.

Once you have implemented these components, you will test our your model in 3 settings:
1) Generate a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2) Perform zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3) Perform task-specific finetuning of your Llama2 model, after implementing a classification head in `classifier.py`. This will give much stronger classification results.
4) Since you've come this far, try implementing something new on top of your hand-written language modeling system! Here are some suggestions:
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the language modeling objective to do domain adaptation
    * enable zero-shot prompting using a more principled inference algorithm than our current implementation. For example, we did not include an attention mask despite right-padding all inputs (to enable batch prediction); this could be improved.
    * perform [prompt-based finetuning](https://arxiv.org/abs/2109.01247)
    * add [regularization](https://arxiv.org/abs/1909.11299) to our finetuning process
    * try parameter-efficient finetuning (see Section 2.2 [here](https://arxiv.org/abs/2110.04366) for an overview)
    * try alternative fine-tuning algorithms e.g. [SMART](https://www.aclweb.org/anthology/2020.acl-main.197) or [WiSE-FT](https://arxiv.org/abs/2109.01903)

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies. Before running the setup script activate your virtual environment as in the two steps below.
    * conda create -n llama_hw python=3.11
    * conda activate llama_hw
    Note: In the `setup.sh` script, if `curl -O` does not work for you, replace it with `wget`
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
* The `data/cfimdb-test.txt` file provided to you does **not** contain gold-labels, and contains a placeholder negative (-1) label. Evaluating your code against this set will show lower accuracies so do not worry if the numbers don't make sense.

## Reference outputs/accuracies: 

*Text Continuation* (`python run_llama.py --option generate`)
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is likely to be logically inconsistent and may contain some coherence or grammar errors).

*Zero Shot Prompting*
Zero-Shot Prompting for SST:

`python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]`

Prompting for SST:
Dev Accuracy: 0.213 (0.000)
Test Accuracy: 0.224 (0.000)

Zero-Shot Prompting for CFIMDB:

`python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]`

Prompting for CFIMDB:
Dev Accuracy: 0.502
Test Accuracy: 0.213

*Classification Finetuning*

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]`

Finetuning for SST:
Dev Accuracy: 0.414 (0.014)
Test Accuracy: 0.418 (0.017)

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]`

Finetuning for CFIMDB:
Dev Accuracy: 0.800 (0.115)
Test Accuracy: -

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

### Acknowledgement
HW assignment version without solution by Vijay Viswanathan (based on the [minllama-assignment](https://github.com/neubig/minllama-assignment)). This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
