# Programming Assignment: Build a Simple RAG System

## Overview

In this assignment, you will build a small Retrieval-Augmented Generation (RAG) system in Python. The system will retrieve relevant facts from `cat-facts.txt` and use a local language model to answer a user's question from those facts.

Your final pipeline should:

1. load and chunk a text file;
2. create an embedding for every chunk;
3. retrieve the chunks most similar to a query;
4. place the retrieved chunks in a grounded prompt; and
5. generate and stream an answer with Ollama.

## Learning objectives

By completing the assignment, you should be able to:

- explain the roles of chunking, embeddings, similarity search, and generation in a RAG system;
- implement cosine-similarity retrieval without a vector-database library;
- inspect retrieval results before generation;
- reduce hallucinations by grounding a model in retrieved context; and
- identify common RAG failure modes.

## Prerequisites

- Basic Python programming
- Familiarity with embeddings and cosine similarity
- A local installation of [Ollama](https://ollama.com)

Install the Python package and pull the two models used in this assignment:

```bash
pip install ollama
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

You may use a virtual environment and a dependency file if you prefer.

## Files

* `cat-facts.txt`: the main knowledge base; each non-empty line is one chunk
* `cat-db-fiction-confusing.txt`: the second evaluation knowledge base; each non-empty line is one chunk
* `demo.py`: the RAG program you will create

The file `cat-db-fiction-confusing.txt` contains fictional challenge records with similar entity names, updates, exceptions, and facts that may need to be combined. Treat these records as true only within this assignment’s knowledge base.

## Assignment steps

### 1. Load and chunk the knowledge base

Read `cat-facts.txt` into a list named `dataset`. Remove surrounding whitespace and skip empty lines. For this assignment, one line is one chunk.

```python
with open("cat-facts.txt", "r", encoding="utf-8") as file:
    dataset = [line.strip() for line in file if line.strip()]
```

Checkpoint: print the number of chunks and the first two chunks. Confirm that blank lines were not included.

### 2. Build the in-memory vector store

Use `ollama.embed()` to embed every chunk. Store each chunk together with its embedding in `VECTOR_DB`.

```python
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

VECTOR_DB = []


def add_chunk_to_database(chunk):
    # Request an embedding and append (chunk, embedding) to VECTOR_DB.
    pass


for chunk in dataset:
    add_chunk_to_database(chunk)
```

Checkpoint: `len(VECTOR_DB)` should equal `len(dataset)`. Each stored embedding should be a list of numbers.

### 3. Implement cosine similarity

Implement the function below without using a vector-database library:

```python
def cosine_similarity(a, b):
    """Return the cosine similarity between vectors a and b."""
    pass
```

Your implementation should compute:

```text
cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Decide how your function should handle zero-length vectors. Add at least two small tests whose expected results you can calculate by hand.

### 4. Retrieve relevant chunks

Implement `retrieve(query, top_n=3)`. It should:

1. embed the query with the same embedding model;
2. compare the query embedding with every chunk embedding;
3. sort the results by similarity in descending order; and
4. return the top `top_n` `(chunk, similarity)` pairs.

```python
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query,
    )["embeddings"][0]

    similarities = []
    for chunk, embedding in VECTOR_DB:
        score = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, score))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]
```

Checkpoint: try several queries and print both the retrieved text and score. Inspect whether the evidence needed for each answer appears in the top results.

### 5. Construct a grounded prompt

Build a prompt that clearly separates instructions from retrieved context. It must tell the model to:

- answer only from the supplied context;
- say that the answer is not in the knowledge base when the context is insufficient; and
- avoid treating an older record as current when a dated update is available.

For example:

```python
input_query = input("Ask a question: ")
retrieved_knowledge = retrieve(input_query, top_n=3)

print("\nRetrieved knowledge:")
for chunk, similarity in retrieved_knowledge:
    print(f"- ({similarity:.3f}) {chunk}")

context = "\n".join(
    f"- {chunk}" for chunk, _similarity in retrieved_knowledge
)

instruction_prompt = f"""You are a grounded question-answering assistant.
Use only the context below to answer the user's question.
If the context does not contain enough evidence, say that the answer is not in the knowledge base.
When records conflict, prefer a clearly dated newer record and explain the update briefly.

Context:
{context}
"""
```

### 6. Generate and stream the answer

Pass the grounded prompt as the system message and the original query as the user message.

```python
stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": input_query},
    ],
    stream=True,
)

print("\nAnswer:")
for response_chunk in stream:
    print(response_chunk["message"]["content"], end="", flush=True)
print()
```

Checkpoint: the program must show the retrieved chunks before the generated answer. This separation is important when diagnosing whether an error came from retrieval or generation.

### 7. Integrate the full pipeline

Put all parts into `demo.py`. Running the script should:

1. load the knowledge base;
2. build the in-memory vector store;
3. ask the user for a query;
4. retrieve and display relevant chunks; and
5. generate and stream a grounded answer.

Keep the functions small enough to test independently. A `main()` function is recommended.

### 8. Test and evaluate the system

After completing `demo.py`, evaluate your RAG system with both knowledge-base files.

Use `top_n=3` for all four questions. Run the questions exactly as written so that results can be compared across submissions.

#### Part A: Evaluation with `cat-facts.txt`

First, build the vector database using **only** `cat-facts.txt`.

Run these two questions:

1. **Medium:** What additional organ allows a cat to smell besides its nose, and where is that organ located?
2. **Harder:** A cat is nine years old. Based on the knowledge base, approximately how many years of its life has it been awake?

#### Part B: Evaluation with `cat-db-fiction-confusing.txt`

Now clear or rebuild your vector database. Then load **only** `cat-db-fiction-confusing.txt`.

Do not keep any chunks from `cat-facts.txt` in the vector database.

Run these two questions:

3. **Medium:** Which Luna needs rabbit-based food, and what color collar does she wear?
4. **Harder:** Can Juniper attend the adoption event scheduled for 2026-05-02? Explain the evidence for your answer.

#### What to report

In your README, include a short evaluation table with one row per question.

For each question, report:

* the knowledge-base file used;
* the value of `top_n`;
* the generated answer;
* whether the answer was correct; and
* if the answer was incorrect, one possible fix using the current assignment setup.

For the fix, suggest a concrete change such as increasing `top_n`, improving the prompt, checking whether the right knowledge-base file was loaded, clearing the old vector database before switching files, or inspecting whether the necessary chunks were retrieved.


Use this format:

| File used                      | Question no. | `top_n` | Generated answer | Correct? | If incorrect, how could it be fixed? |
| ------------------------------ | -----------: | ------: | ---------------- | -------- | ------------------------------------ |
| `cat-facts.txt`                |            1 |       3 | ...              | Yes / No | ...                                  |
| `cat-facts.txt`                |            2 |       3 | ...              | Yes / No | ...                                  |
| `cat-db-fiction-confusing.txt` |            3 |       5 | ...              | Yes / No | ...                                  |
| `cat-db-fiction-confusing.txt` |            4 |       5 | ...              | Yes / No | ...                                  |


When judging correctness, check whether the generated answer is supported by the retrieved chunks. If the retrieved chunks do not contain enough evidence, the correct behavior is for the model to say that the answer is not in the knowledge base.


## 🧠 Bonus Tasks (Optional)

- Add support for reranking the retrieved chunks using another model.
- Support chunking by paragraph instead of line.
- Add a web or CLI interface.
- Explore Hybrid or Graph RAG extensions (e.g., using a small KG or external API).

## Submission

Submit a GitHub repository containing your completed assignment.

Your repository must contain:

* `demo.py` with the complete RAG pipeline;
* `cat-facts.txt`;
* `cat-db-fiction-confusing.txt`;
* `README.md`;
* a dependency file, such as `requirements.txt`, if you use additional Python packages.

Your `README.md` must include:

* setup instructions;
* instructions for running `demo.py`;
* the evaluation table from Section 8.

## Completion checklist

Before submitting, check that:

* [ ] The repository contains `demo.py`, `cat-facts.txt`, `cat-db-fiction-confusing.txt`, and `README.md`.
* [ ] The program runs from start to finish.
* [ ] Empty lines are excluded from the dataset.
* [ ] Every chunk has an embedding.
* [ ] Cosine similarity is implemented and tested.
* [ ] Retrieval returns chunks in descending similarity-score order.
* [ ] Retrieved chunks and similarity scores are printed before generation.
* [ ] The prompt instructs the model to answer only from the retrieved context.
* [ ] The model gives an explicit fallback answer when the retrieved context is insufficient.
* [ ] The README includes the required evaluation table from Section 8.

