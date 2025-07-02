
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
   
```python
# Load the dataset

dataset = []
with open('cat-facts.txt', 'r') as file:
  # complete code line
```

---

### ✅ Task 3: Embed the Chunks

1. Use `ollama.embed()` to compute embeddings for each chunk.
2. Store the results in a `VECTOR_DB` list as tuples of `(chunk, embedding)`.

```python
# Implement the retrieval system

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

# In this example, we will consider each line in the dataset as a chunk for simplicity.
for i, chunk in enumerate(dataset):
  #complete code line
```

---

### ✅ Task 4: Implement Cosine Similarity

1. Write a function `cosine_similarity(a, b)` that computes the similarity between two vectors.

```
def cosine_similarity(a, b):
  #complete the function code lines
```

---

### ✅ Task 5: Retrieval

1. You are given a function `retrieve(query, top_n=3)` that:
   - Embeds the query
   - Calculates cosine similarity between query and all chunk embeddings
   - Returns top-k most relevant chunks

```
def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]
```

---

### ✅ Task 6: Construct a Prompt and Generate a Response

1. Implement prompt construction using the retrieved context.
2. Call `ollama.chat()` using:
   - A system message that includes the prompt
   - A user message with the original query
3. Stream and print the chatbot's response in real time.

```
# For example, a prompt can be constructed as follows:
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}

```

Then use `ollama` to generate the response. In this example, we will use `instruction_prompt` as system message:

```
stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

---

### ✅ Task 7: Now put everything together -- Integrate the Full Pipeline

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

Please submit a code repository containing the following:
- A completed `demo.py` file
- A short README.md that includes:
    - A brief introduction to your code (what it does and how to run it)
    - A short reflection answering the following questions:
      - What limitations did you observe?
      - What could be improved with a larger dataset or more advanced models?

Make sure your repository is clearly structured and includes any necessary dependencies or instructions for reproducibility.