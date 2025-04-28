# 🦙 Llama2 Implementation Guide

This guide outlines the components of the Llama2 model that require implementation and provides a recommended order to approach them. It also includes testing procedures to validate each component.

## 📁 Project Structure Overview

```
minimalist-llm/
   ├── llama.py - This file contains the Llama2 model whose backbone is the [transformer](https://arxiv.org/pdf/1706.03762.pdf). We recommend walking through Section 3 of the paper to understand each component of the transformer.
   ├── rope.py - Rotary Positional Embedding (RoPE) functions.
   ├── rope_test.py - Unit test for the RoPE implementation.
   ├── sanity_check.py - Integration test for the Llama2 model.   
   ├── optimizer.py - Custom implementation of the AdamW optimizer.
   ├── optimizer_test.py - Unit test for the AdamW optimizer.   
   ├── classifier.py - Sentence classification pipeline using Llama2.
   ├── base_llama.py - This is the base class for the Llama model. You won't need to modify this file in this assignment.
   ├── tokenizer.py - This is the tokenizer we will use. You won't need to modify this file in this assignment.
   ├── config.py - This is where the configuration class is defined. You won't need to modify this file in this assignment.
   ├── utils.py - This file contains utility functions for various purpose. You won't need to modify this file in this assignment.
   ├── README.md
```

## 🛠️ Implementation Steps
### ✅ Phase 1: Core Model Implementation & Verification

#### Attention
Refer to ANLP Lecture 2, Slide 43
The multi-head attention layer of the transformer. This layer maps a query and a set of key-value pairs to an output. The output is calculated as the weighted sum of the values, where the weight of each value is computed by a function that takes the query and the corresponding key. To implement this layer, you can:
1. linearly project the queries, keys, and values with their corresponding linear layers
2. split the vectors for multi-head attention
3. follow the equation to compute the attended output of each head
4. concatenate multi-head attention outputs to recover the original shape

$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Llama2 uses a modified version of this procedure called [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) where, instead of each attention head having its own "query", "key", and "vector" head, some groups of "query" heads share the same "key" and "vector" heads. To simplify your implementation, we've taken care of steps #1, 2, and 4 here; you only need to follow the equation to compute the attended output of each head. Refer to ANLP Lecture 3, Slide 17.

#### ROPE
Here, you will implement rotary positional embeddings. This may be tricky; Refer to ANLP Lecture 2 Slide 49,  and Section 3 in https://arxiv.org/abs/2104.09864 for reference. 

#### LlamaLayer
Refer to ANLP Lecture 3, Slide 16.
This corresponds to one transformer layer which has 
1. layer normalization of the input (via Root Mean Square layer normalization)
2. self-attention on the layer-normalized input
3. a residual connection (i.e., add the input to the output of the self-attention)
4. layer normalization on the output of the self-attention
5. a feed-forward network on the layer-normalized output of the self-attention
6. a residual connection from the unnormalized self-attention output added to the output of the feed-forward network

#### todos
Focus on the essential parts of the Llama model that are critical for the forward pass:

* `llama.Attention.forward`: Compute the attention outputs using the provided queries, keys, and values.​
* `llama.RMSNorm._norm`: Implement the Root Mean Square Layer Normalization.​
* `rope.apply_rotary_emb`: Apply rotary positional embeddings to the query and key tensors.​ (this one may be tricky! After implementing this, use `rope_test.py` to test your implementation. If everything is in order, you should get the response "Rotary embedding test passed!")
* `llama.LlamaLayer.forward`: Define the forward pass for a single transformer layer, with RMSNorm, Attention, Residuals, Feedforward.​

### Sanity check (Llama forward pass integration test)
Once these components are implemented, you can run the `sanity_check.py` script to verify that the forward pass produces the expected outputs. The function tests the Llama implementation. It will reload two embeddings that were computed with a reference implementation and check whether the new implementation outputs match the original. ​If everything is in order you will get the message "Your Llama implementation is correct!"

### 🧪 Phase 2: Optimizer Implementation & Testing

After ensuring that the model's forward pass is correct, proceed to implement the optimizer:​

This is where `AdamW` is defined.
You will need to update the `step()` function based on [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) and [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
There are a few slight variations on AdamW, pleae note the following:
- The reference uses the "efficient" method of computing the bias correction mentioned at the end of section 2 "Algorithm" in Kigma & Ba (2014) in place of the intermediate m hat and v hat method.
- The learning rate is incorporated into the weight decay update (unlike Loshchiloc & Hutter (2017)).
- There is no learning rate schedule.

#### todos

   * `optimizer.AdamW.step`: Define the parameter update rules based on the AdamW optimization algorithm.​

You can test your optimizer implementation using the `optimizer_test.py` script, which will validate the correctness of your optimizer against expected behaviors. If everything is in order you will get the message: "Optimizer test passed!"

### classifier.py
This file contains the pipeline to 
* load a pretrained model
* generate an example sentence (to verify that your implemention works)
* call the Llama2 model to encode the sentences for their contextualized representations
* feed in the encoded representations for the sentence classification task
* fine-tune the Llama2 model on the downstream tasks (e.g. sentence classification)
The complete functionality is completed in Phase 3 and 4 below.

### 🧠 Phase 3: Model Generation Functionality

Objective: Enable the model to generate sequences.​

#### Llama
This is the Llama model that takes in input ids and returns next-token predictions and contextualized representation for each word. The structure of ```Llama``` is:
1. an embedding layer that consists of token embeddings ```tok_embeddings```.
2. A stack of ```config.num_hidden_layers``` ```LlamaLayer``` modules, each implementing a decoder block consisting of:
   - Grouped Query Attention (GQA): Multiple query heads share a smaller number of key and value projections, reducing memory and compute overhead while maintaining model expressiveness.
   - Feedforward Network (MLP)
   - Residual Connections and Layer Normalization
3. a projection layer for each hidden state which predicts token IDs (for next-word prediction)
4. a "generate" function which uses temperature sampling to generate long continuation strings. Note that, unlike most practical implementations of temperature sampling, you should not perform nucleus/top-k sampling in your sampling procedure.

The desired outputs are
1. ```logits```: logits (output scores) over the vocabulary, predicting the next possible token at each point
2. ```hidden_state```: the final hidden state at each token in the given document

#### todos

   * `llama.Llama.generate`: Implement Generation Method​
    
Use `classifier.py` to generate example sentences and verify the model's generation capability as follows.

`python run_llama.py --option generate`

### ✅ Phase 4: Implement the Sentence Classifier

Once the model and optimizer are verified, implement the sentence classification pipeline:

#### LlamaSentClassifier (to be implemented)
This class is used to
* encode the sentences using Llama2 to obtain the hidden representation from the final word of the sentence.
* classify the sentence by applying dropout to the pooled-output and project it using a linear layer.

#### todos

- `classifier.LlamaEmbeddingClassifier.forward`: This class is responsible for:
  - Encoding the sentences using the Llama model to obtain the hidden representation of the final word.
  - Applying dropout to the pooled output.
  - Projecting the output through a linear layer to classify the sentence into the desired label categories.

It handles fine-tuning the model on downstream tasks and evaluate its performance.​


*ATTENTION:* The variable names that correspond to Llama2 parameters should not be changed. Any changes to the variable names will fail to load the pre-trained weights.



## Summary
### To be implemented
Components that require your implementations are comment with ```#todo```. The detailed instructions can be found in their corresponding code blocks
* ```llama.Attention.forward```
* ```llama.RMSNorm.norm```
* ```rope.apply_rotary_emb``` 
* ```llama.LlamaLayer.forward```
* ```optimizer.AdamW.step```
* ```llama.Llama.generate```
* ```classifier.LlamaEmbeddingClassifier.forward```

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- [Grouped-Query Attention in LLaMA2](https://arxiv.org/abs/2305.13245)
- [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)

## 🚀 Final Notes

- Do not rename parameters that correspond to Llama2 pre-trained weights (e.g., `tok_embeddings`, `compute_query`, `compute_key`), or loading the reference weights may fail.
- You are free to reorganize internal functions within each file for clarity, but maintain the external API and variable naming.
- Temperature sampling during generation should not include nucleus/top-k sampling — keep it simple!

Good luck, and enjoy building Llama! 🦙