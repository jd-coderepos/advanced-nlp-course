from pathlib import Path
from collections import Counter
import json

from jinja2 import Environment, FileSystemLoader, StrictUndefined


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
TEMPLATES_DIR = PROJECT_DIR / "templates"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    reviews = load_json(DATA_DIR / "reviews.json")
    examples = load_json(DATA_DIR / "few_shot_examples.json")
    config = load_json(DATA_DIR / "task_configs.json")

    project = config["project"]
    tasks = config["tasks"]

    env = build_environment()
    single_prompt_template = env.get_template("sentiment_prompt.txt.j2")
    batch_prompt_template = env.get_template("batch_prompt.txt.j2")
    report_template = env.get_template("report.md.j2")

    # One single-review prompt for every task variant and every review.
    for task in tasks:
        task_output_dir = OUTPUTS_DIR / task["name"]
        task_output_dir.mkdir(parents=True, exist_ok=True)

        for review in reviews:
            rendered = single_prompt_template.render(
                project=project,
                task=task,
                review=review,
                examples=examples,
            )
            (task_output_dir / f"{review['id']}.txt").write_text(rendered, encoding="utf-8")

        batch_rendered = batch_prompt_template.render(task=task, reviews=reviews)
        (task_output_dir / "batch_prompt.txt").write_text(batch_rendered, encoding="utf-8")

    gold_distribution = dict(Counter(review["gold_sentiment"] for review in reviews))
    report = report_template.render(
        project=project,
        reviews=reviews,
        tasks=tasks,
        gold_distribution=gold_distribution,
    )
    (OUTPUTS_DIR / "project_report.md").write_text(report, encoding="utf-8")

    print(f"Generated prompts and report under: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
