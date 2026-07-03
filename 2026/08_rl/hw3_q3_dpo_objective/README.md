# Programming Assignment: Direct Preference Optimization Objective

This assignment focuses on the objective function behind Direct Preference Optimization (DPO). It is designed to run on a normal laptop: no GPU, no large language model download, and no external dataset are required.

Students will implement only the DPO loss function. The provided toy dataset contains prompts, preferred and rejected responses, and precomputed log probabilities from a current policy model and a reference model.

## What is the point of this homework?

The point of this homework is not to train a real chatbot. It is a small implementation exercise so that students understand what the DPO objective is doing before seeing it inside a larger RLHF or alignment pipeline.

In this assignment, students should learn:

1. What kind of data DPO needs: a prompt, a chosen response, and a rejected response.
2. How DPO compares the current policy against a fixed reference policy.
3. What the DPO logit means: a positive logit means the current policy prefers the chosen response more than the rejected response after accounting for the reference policy; a negative logit means the opposite.
4. What the DPO loss does: it becomes smaller when the chosen response is preferred more strongly than the rejected response.
5. Why the exercise can run on a laptop: the assignment uses precomputed toy log probabilities instead of loading a large language model.

A concise summary of the homework aim is:

> In this assignment, you will implement and test the Direct Preference Optimization objective on a small preference dataset, and interpret how the DPO loss encourages a policy to assign higher relative probability to chosen responses than to rejected responses.

## Learning goals

By the end of the assignment, one should be able to:

1. Explain how DPO learns directly from preference pairs.
2. Implement the DPO objective in a numerically stable way.
3. Interpret how the current policy is rewarded or penalized relative to a reference policy.
4. Test the objective on a small preference dataset.

## Background

Lecture 8, slide 44 introduces DPO as a way to train directly from pairwise preference examples instead of first training a scalar reward model and then doing unstable RL updates. The original DPO paper frames this as a simple classification-style objective over chosen and rejected responses.

For a prompt `x`, a preferred response `y_w`, and a rejected response `y_l`, the DPO loss is:

```text
L_DPO = - log sigmoid(
    beta * [
        (log pi_theta(y_w | x) - log pi_theta(y_l | x))
        -
        (log pi_ref(y_w | x) - log pi_ref(y_l | x))
    ]
)
```

Here:

- `pi_theta` is the current policy being optimized.
- `pi_ref` is the fixed reference policy.
- `beta` controls how strongly the objective penalizes deviation from the reference policy.
- `y_w` is the preferred or winning response.
- `y_l` is the rejected or losing response.

## Files

```text
dpo_programming_assignment/
├── assignment.md
├── data/
│   └── toy_preferences.jsonl
├── references.md
├── requirements.txt
├── run_experiment.py
├── src/
│   ├── __init__.py
│   ├── dpo_objective.py
│   └── utils.py
└── tests/
    └── test_dpo_objective.py
```

## Setup

From inside this folder:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the requirements:

```bash
pip install -r requirements.txt
```

## Student task

Open:

```text
src/dpo_objective.py
```

Implement the function:

```python
dpo_loss(...)
```

Do not change the tests. You may add helper functions if you want.

## Run the tests

```bash
pytest -q
```

Before implementation, several tests should fail. After implementing the loss correctly, all tests should pass.

## Run the toy experiment

```bash
python run_experiment.py --data data/toy_preferences.jsonl --beta 0.5
```

Expected behavior after implementation:

- The script prints the mean DPO loss.
- It prints the fraction of examples where the policy already prefers the chosen response more than the rejected response, after accounting for the reference policy.
- It prints per-example diagnostics so students can inspect which preference pairs already agree with the current policy and which do not.

## How to interpret the toy experiment

The experiment is a diagnostic, not a full training run.

- A **positive DPO logit** means the current policy favors the chosen response over the rejected response more than the reference policy does.
- A **negative DPO logit** means the current policy still favors the rejected response too much, relative to the chosen response.
- A **lower loss** means the objective is more satisfied for that example.
- A **higher loss** means the model would receive a stronger update if this were part of a real training loop.
- The preference accuracy is the fraction of examples where the DPO logit is positive.

Because the dataset uses fixed toy log probabilities, the script does not update model parameters. It only shows whether the implemented objective behaves as expected on concrete preference examples.

## Notes

This assignment does not train a language model. The dataset already contains the log probabilities that would normally be produced by a policy model and a reference model. This keeps the assignment focused on the DPO objective itself.

## Submission instructions

Submit the following:

1. The completed file `src/dpo_objective.py`.
2. The output of `pytest -q`, either as a screenshot or copied text.
3. The output of the toy experiment command:

   ```bash
   python run_experiment.py --data data/toy_preferences.jsonl --beta 0.5
   ```

4. Brief answers to these questions:
   - What does a positive DPO logit mean?
   - Why does DPO compare the current policy against a reference policy?
   - What does the `beta` parameter control?
   - Pick one example with a low loss and one example with a high loss. Explain the difference.


# References

- Lecture 8, slide 44: "Contrasting Pairwise Examples", Advanced Natural Language Processing, Lecture 8, Reinforcement Learning from Human Feedback. The slide introduces direct preference optimization as a way to learn directly from preference pairs instead of scalar rewards and unstable RL updates.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290. https://arxiv.org/abs/2305.18290