# NLP Jinja Sentiment Demo

A small Python project showing how Jinja templates help manage prompt variants for a classical NLP task: sentiment analysis.

The demo renders prompts for:

- document-level three-class sentiment,
- document-level sentiment with a `mixed` class,
- aspect-based sentiment analysis,
- batch annotation prompts.

The value of Jinja appears when the prompt needs conditional logic:

- include few-shot examples only for some tasks,
- switch between document-level and aspect-level output schemas,
- change allowed label sets,
- include a sarcasm warning only when relevant,
- keep long prompt text outside Python code.

## Run

```bash
pip install -r requirements.txt
python src/nlp_jinja_sentiment_demo/render_prompts.py
```

Generated files appear in `outputs/`.
