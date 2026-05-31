from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "research_tasks.json"
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "templates"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def slugify(value: str) -> str:
    """Convert a task title into a safe filename fragment."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def load_data(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_environment(template_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["slugify"] = slugify
    return env


def render_project(data_path: Path, template_dir: Path, output_dir: Path) -> list[Path]:
    data = load_data(data_path)
    env = make_environment(template_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_files: list[Path] = []

    report_template = env.get_template("report.md.j2")
    report = report_template.render(**data)
    report_path = output_dir / "task_report.md"
    report_path.write_text(report, encoding="utf-8")
    rendered_files.append(report_path)

    prompt_template = env.get_template("prompts/task_prompt.txt.j2")
    for task in data["tasks"]:
        prompt = prompt_template.render(project=data["project"], task=task)
        prompt_path = output_dir / f"prompt_{slugify(task['title'])}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        rendered_files.append(prompt_path)

    return rendered_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Jinja templates using demo research-task data.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the JSON data file.")
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_DIR, help="Path to the templates directory.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for rendered files.")
    args = parser.parse_args()

    rendered_files = render_project(args.data, args.templates, args.out)
    print("Rendered files:")
    for path in rendered_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
