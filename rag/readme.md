
# 🧪 Programming Assignment: Implement a Simple RAG System in Python

## 🎯 Objective
In this exercise, you will build a simple **Retrieval-Augmented Generation (RAG)** system using Python and the [Ollama](https://ollama.com) interface for local LLM inference. You’ll implement the full RAG pipeline including document chunking, embedding, retrieval, and response generation.

---

## 📦 Prerequisites

- Basic Python programming skills
- Familiarity with vector embeddings and cosine similarity
- Installed `ollama` and required models (see Task 1)

---

## 🗂️ Tasks

### ✅ Task 1: Setup Environment and Download Models

1. Install the `ollama` command-line tool from [ollama.com](https://ollama.com).
2. Pull the required models:
   ```bash
   ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
   ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
   ```
3. Install the Python interface:
   ```bash
   pip install ollama
   ```

---

### ✅ Task 2: Load and Chunk the Dataset

1. Use the provided `cat-facts.txt` file (containing one cat fact per line), or feel free to substitute it with your own text file of short factual sentences.
2. Write Python code to:
   - Load the dataset
   - Treat each line as a "chunk" of knowledge

---

### ✅ Task 3: Embed the Chunks

1. Use `ollama.embed()` to compute embeddings for each chunk.
2. Store the results in a `VECTOR_DB` list as tuples of `(chunk, embedding)`.

```python
VECTOR_DB = []
def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))
```

---

### ✅ Task 4: Implement Cosine Similarity

1. Write a function `cosine_similarity(a, b)` that computes the similarity between two vectors.

---

### ✅ Task 5: Implement Retrieval

1. Implement a function `retrieve(query, top_n=3)` that:
   - Embeds the query
   - Calculates cosine similarity between query and all chunk embeddings
   - Returns top-k most relevant chunks

---

### ✅ Task 6: Construct a Prompt and Generate a Response

1. Implement prompt construction using the retrieved context.
2. Call `ollama.chat()` using:
   - A system message that includes the prompt
   - A user message with the original query
3. Stream and print the chatbot's response in real time.

---

### ✅ Task 7: Integrate the Full Pipeline

1. Combine retrieval and generation into a single script `demo.py`.
2. On running, it should:
   - Load the dataset
   - Build the vector store
   - Take a user query
   - Retrieve relevant chunks
   - Generate and print an answer

---

### ✅ Task 8: Experimentation & Reflection

- Try different values for `top_n` in retrieval.
- Modify the prompt template.
- Ask questions with varying specificity.

---

## 🧠 Bonus Tasks (Optional)

- Add support for reranking the retrieved chunks using another model.
- Support chunking by paragraph instead of line.
- Add a web or CLI interface.
- Explore Hybrid or Graph RAG extensions (e.g., using a small KG or external API).

---

## 📤 Submission

Submit your completed `demo.py` file and a short reflection answering:
- What limitations did you observe?
- What could be improved with a larger dataset or more advanced models?
