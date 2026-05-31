# Jinja Templating Demo

A small Python project demonstrating why a templating library such as [Jinja](https://github.com/pallets/jinja/) is useful when prompts and output documents contain conditional logic.

The project keeps reusable templates outside the Python code and fills them with values from a small JSON data file. This is useful when prompts need to change depending on task type, output format, priority, citation requirements, or available context files.

## What this demonstrates

- External templates stored in `templates/`
- A small data file stored in `data/research_tasks.json`
- Conditional logic in templates using `{% if ... %}`
- Loops using `{% for ... %}`
- Filters such as `join`, `upper`, and a custom `slugify` filter
- Rendering both a Markdown report and task-specific prompt files

## Project structure

```text
jinja-templating-demo/
├── data/
│   └── research_tasks.json
├── outputs/
│   └── generated files appear here
├── src/
│   └── jinja_templating_demo/
│       ├── __init__.py
│       └── render.py
├── templates/
│   ├── report.md.j2
│   └── prompts/
│       └── task_prompt.txt.j2
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/jinja_templating_demo/render.py
```

This creates:

```text
outputs/task_report.md
outputs/prompt_extract-ald-process-parameters.txt
outputs/prompt_summarize-survey-table-findings.txt
outputs/prompt_check-ontology-alignment-candidates.txt
```

You can also choose custom paths:

```bash
python src/jinja_templating_demo/render.py \
  --data data/research_tasks.json \
  --templates templates \
  --out outputs
```

## Why Jinja is justified here

Without a template engine, Python code often becomes cluttered with string concatenation and many small `if` statements. In this project, the template itself decides what to render:

- If the output format is `json`, the prompt asks for valid JSON and lists JSON fields.
- If the output format is `markdown`, the prompt asks for Markdown bullets.
- If citations are required, the prompt includes a citation rule.
- If no context files are available, the prompt explicitly says so.
- If a task is high priority, the prompt adds a stricter quality check.

This keeps application logic in Python and presentation/prompt wording in external files.
