"""Compare the rule-based fallback against the official LLM baseline.

This script is for benchmark analysis, not submission-time inference. Its job is
to answer a specific quality question:

    "Does the environment reward only simple threshold rules, or does the LLM
    baseline gain measurable headroom on the harder tasks?"

If the LLM score is only equal to the heuristic score, the environment may be
too shallow. If the LLM consistently improves on the heuristic baseline,
especially on the hard task, that is evidence the benchmark can differentiate
between simple rules and stronger policy behavior.
"""

from __future__ import annotations

from typing import Dict, List

from meverse import SurveillanceAction, build_llm_client, choose_surveillance_action, list_task_names, load_policy_config, select_action
from meverse.server.meverse_environment import MarketSurveillanceEnvironment


def run_policy(task_name: str, policy_name: str) -> Dict[str, float]:
    """Run one policy on one task and return the grader output."""

    env = MarketSurveillanceEnvironment(task=task_name, eval_mode=True, demo_mode=False)
    observation = env.reset(task=task_name)

    policy_config = load_policy_config()
    llm_client = build_llm_client(policy_config) if policy_name == "llm" else None

    while not observation.done:
        if policy_name == "llm":
            action = select_action(
                observation,
                client=llm_client,
                config=policy_config,
                allow_fallback=False,
            )
        elif policy_name == "heuristic":
            action = choose_surveillance_action(observation)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        observation = env.step(SurveillanceAction(action_type=action))

    return env.grade()


def summarize(task_names: List[str]) -> None:
    """Print a compact comparison table for all tasks and both policies."""

    llm_total = 0.0
    heuristic_total = 0.0

    print("task\t\t\theuristic\tllm\tdelta")
    print("-" * 64)
    for task_name in task_names:
        heuristic_grade = run_policy(task_name, "heuristic")
        llm_grade = run_policy(task_name, "llm")
        heuristic_score = heuristic_grade["score"]
        llm_score = llm_grade["score"]
        delta = llm_score - heuristic_score
        heuristic_total += heuristic_score
        llm_total += llm_score
        print(f"{task_name:24}\t{heuristic_score:.4f}\t\t{llm_score:.4f}\t{delta:+.4f}")

    avg_heuristic = heuristic_total / max(1, len(task_names))
    avg_llm = llm_total / max(1, len(task_names))
    print("-" * 64)
    print(f"{'average':24}\t{avg_heuristic:.4f}\t\t{avg_llm:.4f}\t{(avg_llm - avg_heuristic):+.4f}")


def main() -> None:
    """Entry point for environment-depth evaluation."""

    policy_config = load_policy_config()
    if not policy_config.api_token:
        raise RuntimeError(
            "HF_TOKEN is required to compare the official LLM baseline against the heuristic policy."
        )
    summarize(list_task_names())


if __name__ == "__main__":
    main()

