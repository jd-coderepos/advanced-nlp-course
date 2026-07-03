# Programming Assignment: Train a Tiny Retrieval-Oriented Embedding Model

## Overview

In this assignment, you will train a small retrieval-oriented embedding model. The goal is to make a query vector close to a relevant document vector and far from irrelevant document vectors.

This assignment follows the idea from the RAG lecture: for retrieval, we do not only want embeddings that represent meaning in general. We want embeddings where relevant query-document pairs score higher than irrelevant query-document pairs.

The code is intentionally small enough to run on a laptop CPU. It does not download BERT, sentence-transformer models, or large datasets.

## Learning objectives

By completing this assignment, you should be able to:

- explain the difference between general-purpose embeddings and retrieval-oriented embeddings;
- describe positive documents, negative documents, and hard negatives;
- implement and interpret a contrastive hinge objective for retrieval;
- train a small dual-encoder-style model;
- evaluate retrieval using Recall@1, Recall@3, and MRR; and
- diagnose at least one retrieval success or failure.

## Repository structure

```text
retrieval_embedding_assignment/
├── README.md
├── requirements.txt
├── data/
│   ├── documents.tsv
│   ├── train_triplets.tsv
│   ├── dev_queries.tsv
│   └── test_queries.tsv
├── src/
│   ├── data_utils.py
│   ├── metrics.py
│   ├── model.py
│   ├── objective.py
│   ├── retrieve.py
│   └── train.py
└── tests/
    └── test_objective.py
```

## Setup

Create and activate a virtual environment, then install the dependency.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

After activation, your terminal should show `(.venv)` before the prompt.

## The most important function

The main function for this assignment is in `src/objective.py`:

```python
def retrieval_hinge_loss(
    query_embeddings,
    positive_doc_embeddings,
    negative_doc_embeddings,
    margin=0.2,
):
    ...
```

It implements the following objective:

```text
loss = max(0, margin + score(query, negative_doc) - score(query, positive_doc))
```

where `score` is cosine similarity.

The loss is zero if the positive document already scores at least `margin` higher than the negative document. Otherwise, the model is penalized. In other words, the objective pushes positive documents closer to the query and negative documents farther away.

## Step 1: Test the objective

The two small tests are written in this file:

```text
tests/test_objective.py
```

Open `tests/test_objective.py` and read the two test cases. They use very small 2-dimensional vectors so that you can calculate the expected losses by hand.

Then run the tests from the repository root:

```bash
python -m unittest tests/test_objective.py
```

This command does not print the loss values directly. If the tests pass, you should see output similar to this:

```text
..
----------------------------------------------------------------------
Ran 2 tests in ...s

OK
```

The two dots mean that two tests ran successfully. `OK` means that the computed losses matched the expected values inside `tests/test_objective.py`.

The first test checks that the loss is `0.0`, because the positive document is already closer to the query than the negative document by at least the margin.

The second test checks that the loss is `1.2`, because the negative document is closer to the query than the positive document. With margin `0.2`, the loss is:

```text
0.2 + 1.0 - 0.0 = 1.2
```

## Step 2: Train the model

Run:

```bash
python -m src.train --epochs 50 --margin 0.2 --seed 13
```

The script will:

1. load the document collection;
2. load training triplets of the form `(query, positive document, negative document)`;
3. evaluate retrieval before training;
4. train the model with the hinge objective;
5. evaluate retrieval after training; and
6. save the trained model to `outputs/tiny_retriever.pt`.

The default model is a very small mean-pooled word-embedding encoder. It is not meant to be strong; it is meant to make the training objective easy to inspect.

## Step 3: Try retrieval with your trained model

After training, run:

```bash
python -m src.retrieve "Which Luna has rabbit food and a yellow collar?" --top-k 5
```

Try at least two additional queries of your own.

## Step 4: Run one small experiment

Choose one of the following changes and run training again:

- change the margin, for example `--margin 0.1` or `--margin 0.5`;
- change the number of epochs;
- change the random seed;
- compare the default model with the weighted bag-of-words model:

```bash
python -m src.train --model bow --epochs 20 --seed 13
```

## What to report

In your README report, include:

1. the command you used for your main run;
2. the retrieval metrics before and after training;
3. a short explanation of the objective function in `src/objective.py`;
4. one query where retrieval worked well;
5. one query where retrieval failed or was uncertain; and
6. one observation from your small experiment.

Use this table format for the metrics:

| Run | Split | Recall@1 | Recall@3 | MRR |
|---|---|---:|---:|---:|
| Before training | test | ... | ... | ... |
| After training | test | ... | ... | ... |

For the failure or uncertain case, explain whether the problem appears to come from the training data, the model size, the negative examples, or the objective settings.

## Submission

Write your response in your main homework solution document.

Your response should include:

1. the command you used for your main run;
2. the retrieval metrics before and after training;
3. a short explanation of the objective function in `src/objective.py`;
4. one query where retrieval worked well;
5. one query where retrieval failed or was uncertain; and
6. one observation from your small experiment.

If you changed any code, also include a link to your modified repository or clearly describe the changes in your homework document.

## Sources consulted

- Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering*. https://arxiv.org/abs/2004.04906
- Reimers and Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. https://aclanthology.org/D19-1410/
- Izacard et al. (2022), *Unsupervised Dense Information Retrieval with Contrastive Learning*. https://arxiv.org/abs/2112.09118
- PyTorch documentation, `TripletMarginLoss`. https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
