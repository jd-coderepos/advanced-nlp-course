# Relatable NLP Prompt Templating Demo

A tiny project showing how Jinja helps manage conditional prompt variants for sentiment-analysis tasks.

## Dataset Summary

Total reviews: 8

Reviews with possible sarcasm: 1

Gold-label distribution:

- mixed: 3
- positive: 2
- neutral: 1
- negative: 2

## Prompt Variants

### 1. document_three_class

- Mode: `document`
- Labels: positive, neutral, negative
- Few-shot examples: yes, up to 3- JSON-only response: yes
This variant forces a single three-class document-level decision.

### 2. document_with_mixed

- Mode: `document`
- Labels: positive, neutral, negative, mixed
- Few-shot examples: yes, up to 5- JSON-only response: yes
This variant allows a `mixed` sentiment class, which is useful for reviews with both praise and criticism.

### 3. aspect_based

- Mode: `aspect`
- Labels: positive, neutral, negative, mixed, not_mentioned
- Few-shot examples: no- JSON-only response: yes
This variant changes the output schema from one document-level label to a list of aspect-level labels.


## Why Jinja Is Useful Here

The Python code does not need separate long strings for every NLP prompt variant.  
The templates handle:

- conditional task modes,
- optional few-shot examples,
- different label spaces,
- sarcasm warnings,
- document-level versus aspect-level JSON schemas,
- batch versus single-review prompts.