"""Iterative LLM-as-judge prompt optimizer for the surveillance agent.

This script runs a closed-loop optimization on the system prompt used by the
surveillance LLM (``meverse/policy.py::llm_action``). It is deliberately a
thin wrapper: it does NOT reimplement the environment or the OpenAI client
configuration. Instead, it re-uses:

  - ``MarketSurveillanceEnvironment`` for the deterministic episode rollout
  - ``meverse.policy.load_policy_config`` / ``build_llm_client`` for the
    OpenAI-compatible client (same env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN)
  - ``meverse.baseline_policy.choose_surveillance_action`` as a per-step
    fallback when the LLM call errors mid-episode (so a single bad response
    does not invalidate the entire trajectory's score signal)

Loop, per task:

  1. Run an episode on a fixed seed with the current system prompt and
     collect the trajectory.
  2. Send {prompt, trajectory summary, score breakdown} to the same LLM as a
     "judge" and ask it for a revised system prompt.
  3. Run the same episode again on the same seed with the revised prompt and
     compare scores. Keep the higher-scoring prompt.
  4. Repeat for ``--iterations`` rounds (default 5).
  5. Validate the winner on 3 seeds (42, 123, 999) — only accept it if its
     mean score across those 3 seeds beats the starting prompt's mean.

Curriculum mode (``--curriculum``) runs the optimizer sequentially on
burst_detection -> pattern_manipulation_detection -> full_market_surveillance,
threading each task's winning prompt into the next task's starting prompt.

CLI:
    python prompt_optimizer.py --task burst_detection --iterations 5 --seed 42
    python prompt_optimizer.py --curriculum --iterations 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from meverse.baseline_policy import choose_surveillance_action
from meverse.env import load_repo_env
from meverse.models import SurveillanceAction, SurveillanceObservation
from meverse.policy import (
    VALID_ACTIONS,
    build_llm_client,
    load_policy_config,
)
from meverse.server.meverse_environment import MarketSurveillanceEnvironment

load_repo_env()


# Mirrors the static system prompt currently embedded in ``meverse/policy.py``
# (``llm_action``). Kept here as a constant so the optimizer can use it as the
# starting point without having to parse policy.py's source.
INITIAL_SYSTEM_PROMPT = (
    "You are a market surveillance controller in a simulated AMM market. "
    "Choose exactly one action from ALLOW, FLAG, BLOCK, MONITOR. "
    "Return JSON only: {\"action\": \"ALLOW\"}."
)


JUDGE_SYSTEM_INSTRUCTION = (
    "You are an expert prompt engineer optimizing the system prompt of a "
    "market surveillance LLM agent.\n\n"
    "The agent reads a structured market observation (trade_frequency, "
    "time_gap_min, recent_price_impacts, suspiciousness_score, "
    "manipulation_score, etc.) and must output exactly one of: "
    "ALLOW, FLAG, BLOCK, MONITOR.\n\n"
    "Episodes are graded with this weighted reward function:\n"
    "  detection_score        weight 0.50  (catch suspicious activity; "
    "BLOCK > FLAG > MONITOR > ALLOW on suspicious steps)\n"
    "  false_positive_score   weight 0.20  (penalty for FLAG/BLOCK on "
    "normal trades)\n"
    "  false_negative_score   weight 0.15  (penalty for ALLOW on suspicious)\n"
    "  health_score           weight 0.10  (rewards ALLOW/MONITOR on normal)\n"
    "  overblocking_score     weight 0.05  (penalty for BLOCK on normal)\n\n"
    "CRITICAL: Do not suggest BLOCK for everything. The reward function "
    "penalizes false positives and overblocking. The winning prompt MUST "
    "balance detection against precision — being trigger-happy with BLOCK "
    "on normal flow tanks the score via false_positive_score, "
    "overblocking_score, and health_score.\n\n"
    "You will be given:\n"
    "  - the current system prompt being used by the agent\n"
    "  - a per-step trajectory (step | action | reward | label_hint) from "
    "one full episode on a fixed seed\n"
    "  - the final score breakdown for that episode\n\n"
    "Output ONLY the revised system prompt as plain text. No preamble, no "
    "explanation, no markdown fences, no surrounding quotes. The revised "
    "prompt MUST:\n"
    "  - tell the agent how to map raw observation features to actions\n"
    "  - explicitly cover ALLOW for normal flow (to protect health and "
    "precision scores)\n"
    "  - end with a JSON-output contract line, exactly: "
    "Return JSON only: {\"action\": \"ALLOW\"}\n"
    "  - be self-contained (the agent only sees this system prompt + the "
    "observation JSON)\n"
    "  - stay under ~600 words"
)


# ----------------------------------------------------------------------------
# LLM call (mirrors meverse.policy.llm_action but takes the system prompt as
# an argument so the optimizer can sweep over candidates).
# ----------------------------------------------------------------------------


def _observation_to_prompt_payload(observation: SurveillanceObservation) -> Dict[str, Any]:
    return {
        "task": observation.task_name,
        "step_num": observation.step_num,
        "max_steps": observation.max_steps,
        "current_amm_price": observation.current_amm_price,
        "liquidity_snapshot": observation.liquidity_snapshot,
        "recent_trade_count": observation.recent_trade_count,
        "trades_in_window": observation.trades_in_window,
        "trade_frequency": observation.trade_frequency,
        "average_trade_size": observation.average_trade_size,
        "maximum_trade_size": observation.maximum_trade_size,
        "recent_slippage_impact": observation.recent_slippage_impact,
        "time_gap_mean": observation.time_gap_mean,
        "time_gap_min": observation.time_gap_min,
        "recent_time_gaps": observation.recent_time_gaps,
        "recent_price_impacts": observation.recent_price_impacts,
        "suspiciousness_score": observation.suspiciousness_score,
        "manipulation_score": observation.manipulation_score,
    }


def llm_action_with_prompt(
    client: OpenAI,
    observation: SurveillanceObservation,
    model_name: str,
    system_prompt: str,
) -> str:
    """LLM action selection driven by an arbitrary system prompt."""

    user_payload = _observation_to_prompt_payload(observation)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        max_tokens=12,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(content)
    action = str(parsed.get("action", "")).strip().upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid LLM action: {action!r}")
    return action


# ----------------------------------------------------------------------------
# Episode rollout
# ----------------------------------------------------------------------------


def _label_hint_from_score(observation: SurveillanceObservation) -> str:
    """Coarse label hint derived from the agent-visible scores.

    The true ground-truth label is hidden from the agent (and from us, by
    design — we are not allowed to peek at ``debug_snapshot``). The judge
    only needs a hint about which steps "looked" suspicious so it can reason
    about whether the agent under- or over-reacted.
    """

    if observation.manipulation_score >= 0.5 or observation.suspiciousness_score >= 0.5:
        return "suspicious"
    if observation.manipulation_score >= 0.3 or observation.suspiciousness_score >= 0.3:
        return "borderline"
    return "normal"


def run_episode(
    env: MarketSurveillanceEnvironment,
    system_prompt: str,
    client: OpenAI,
    model_name: str,
    seed: int,
    task_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one full episode and return (trajectory, grade)."""

    observation = env.reset(task=task_name, seed=seed)
    trajectory: List[Dict[str, Any]] = []
    step_num = 0
    fallbacks = 0
    max_steps = max(1, int(getattr(observation, "max_steps", 1)))

    while not observation.done and step_num < max_steps:
        decision_obs = observation
        try:
            action = llm_action_with_prompt(client, decision_obs, model_name, system_prompt)
        except Exception:
            action = choose_surveillance_action(decision_obs)
            fallbacks += 1

        observation = env.step(SurveillanceAction(action_type=action))
        reward = float(observation.reward or 0.0)
        trajectory.append(
            {
                "step_num": step_num,
                "observation": {
                    "trade_frequency": decision_obs.trade_frequency,
                    "time_gap_min": decision_obs.time_gap_min,
                    "average_trade_size": decision_obs.average_trade_size,
                    "maximum_trade_size": decision_obs.maximum_trade_size,
                    "recent_slippage_impact": decision_obs.recent_slippage_impact,
                    "suspiciousness_score": decision_obs.suspiciousness_score,
                    "manipulation_score": decision_obs.manipulation_score,
                    "recent_trade_count": decision_obs.recent_trade_count,
                },
                "action": action,
                "reward": reward,
                "label_hint_from_score": _label_hint_from_score(decision_obs),
            }
        )
        step_num += 1

    grade = env.grade()
    grade["_fallbacks"] = fallbacks
    return trajectory, grade


# ----------------------------------------------------------------------------
# Judge call
# ----------------------------------------------------------------------------


def summarize_trajectory(trajectory: List[Dict[str, Any]]) -> str:
    """Token-efficient one-line-per-step summary for the judge."""

    lines = ["step | action  | reward | label_hint"]
    for t in trajectory:
        lines.append(
            f"{t['step_num']:>4} | {t['action']:<7} | {t['reward']:+.3f} | "
            f"{t['label_hint_from_score']}"
        )
    return "\n".join(lines)


def _strip_judge_wrapping(text: str) -> str:
    """Remove markdown fences / leading 'Here is the revised prompt:' chatter."""

    cleaned = text.strip()

    if cleaned.startswith("```"):
        # Drop the first line (e.g. ``` or ```text) and the trailing fence.
        parts = cleaned.split("\n")
        if parts and parts[0].startswith("```"):
            parts = parts[1:]
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        cleaned = "\n".join(parts).strip()

    # Trim quotation-mark wrapping if the model returned a quoted string.
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()

    return cleaned


def call_judge(
    client: OpenAI,
    model_name: str,
    current_prompt: str,
    trajectory: List[Dict[str, Any]],
    grade: Dict[str, Any],
    task_name: str,
) -> str:
    """Ask the judge LLM for a revised system prompt."""

    summary = summarize_trajectory(trajectory)
    grade_view = {
        "score": grade.get("score"),
        "detection_score": grade.get("detection_score"),
        "false_positive_score": grade.get("false_positive_score"),
        "false_negative_score": grade.get("false_negative_score"),
        "health_score": grade.get("health_score"),
        "overblocking_score": grade.get("overblocking_score"),
    }

    user_payload = (
        f"=== TASK ===\n{task_name}\n\n"
        f"=== CURRENT SYSTEM PROMPT ===\n{current_prompt}\n\n"
        f"=== TRAJECTORY (step | action | reward | label_hint) ===\n{summary}\n\n"
        f"=== EPISODE SCORE BREAKDOWN ===\n{json.dumps(grade_view, indent=2)}\n\n"
        "Output the revised system prompt now. Plain text only, no markdown, "
        "no preamble."
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.5,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_payload},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    return _strip_judge_wrapping(raw)


# ----------------------------------------------------------------------------
# Optimization loop
# ----------------------------------------------------------------------------


def optimize_prompt_for_task(
    task_name: str,
    iterations: int,
    seed: int,
    starting_prompt: str,
    client: OpenAI,
    model_name: str,
) -> Tuple[str, float, List[Dict[str, Any]]]:
    """Run the iterative judge-loop on a single task. Returns (best_prompt, best_score, history)."""

    bar = "=" * 72
    print(f"\n{bar}\n[TASK] {task_name}  iterations={iterations}  seed={seed}\n{bar}", flush=True)

    env = MarketSurveillanceEnvironment(task=task_name, eval_mode=True, demo_mode=False)

    print("[BASELINE] running episode with starting prompt...", flush=True)
    base_traj, base_grade = run_episode(env, starting_prompt, client, model_name, seed, task_name)
    best_prompt = starting_prompt
    best_score = float(base_grade["score"])
    best_traj = base_traj
    best_grade = base_grade
    print(
        f"[BASELINE] score={best_score:.4f} "
        f"detection={base_grade['detection_score']:.3f} "
        f"fp={base_grade['false_positive_score']:.3f} "
        f"overblock={base_grade['overblocking_score']:.3f} "
        f"fallbacks={base_grade.get('_fallbacks', 0)}",
        flush=True,
    )

    history: List[Dict[str, Any]] = [
        {"iteration": 0, "score": best_score, "accepted": True, "grade": base_grade}
    ]

    for it in range(1, iterations + 1):
        new_prompt = call_judge(
            client=client,
            model_name=model_name,
            current_prompt=best_prompt,
            trajectory=best_traj,
            grade=best_grade,
            task_name=task_name,
        )
        if not new_prompt or len(new_prompt) < 30:
            print(
                f"[ITER {it}] judge returned empty/too-short prompt; skipping",
                flush=True,
            )
            history.append({"iteration": it, "accepted": False, "reason": "empty_judge_output"})
            continue

        try:
            new_traj, new_grade = run_episode(
                env, new_prompt, client, model_name, seed, task_name
            )
        except Exception as exc:
            print(f"[ITER {it}] new prompt errored during rollout: {exc}", flush=True)
            history.append({"iteration": it, "accepted": False, "reason": "episode_error"})
            continue

        new_score = float(new_grade["score"])
        accepted = new_score > best_score
        verdict = "accepted" if accepted else "rejected"
        print(
            f"[ITER {it}] old_score={best_score:.4f} new_score={new_score:.4f} "
            f"detection={new_grade['detection_score']:.3f} "
            f"fp={new_grade['false_positive_score']:.3f} "
            f"overblock={new_grade['overblocking_score']:.3f} -> {verdict}",
            flush=True,
        )
        history.append(
            {
                "iteration": it,
                "old_score": best_score,
                "new_score": new_score,
                "accepted": accepted,
                "grade": new_grade,
            }
        )
        if accepted:
            best_prompt = new_prompt
            best_score = new_score
            best_traj = new_traj
            best_grade = new_grade

    return best_prompt, best_score, history


# ----------------------------------------------------------------------------
# Multi-seed validation (anti-reward-hacking guard)
# ----------------------------------------------------------------------------


def validate_on_seeds(
    task_name: str,
    system_prompt: str,
    client: OpenAI,
    model_name: str,
    seeds: Iterable[int],
) -> List[float]:
    """Run the prompt on each seed (deterministic) and return per-seed scores."""

    env = MarketSurveillanceEnvironment(task=task_name, eval_mode=True, demo_mode=False)
    scores: List[float] = []
    for s in seeds:
        _, grade = run_episode(env, system_prompt, client, model_name, s, task_name)
        scores.append(float(grade["score"]))
    return scores


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ----------------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------------


CURRICULUM_TASKS = [
    "burst_detection",
    "pattern_manipulation_detection",
    "full_market_surveillance",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterative LLM-as-judge prompt optimizer for the meverse "
            "surveillance agent."
        )
    )
    parser.add_argument(
        "--task",
        default="burst_detection",
        choices=CURRICULUM_TASKS,
        help="Task to optimize when --curriculum is not set.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of judge-and-rerun rounds per task (default 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the inner optimization loop (default 42).",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Run on burst_detection -> pattern_manipulation_detection -> "
            "full_market_surveillance, threading the winning prompt forward."
        ),
    )
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[42, 123, 999],
        help="Seeds used for the multi-seed validation guard (default 42 123 999).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    config = load_policy_config()
    client = build_llm_client(config)
    if client is None:
        print(
            "[ERROR] HF_TOKEN is not set in the environment. The optimizer "
            "requires a working OpenAI-compatible client. Aborting.",
            flush=True,
        )
        return 2

    print(
        f"[CONFIG] model={config.model_name} api_base={config.api_base_url} "
        f"task={args.task} curriculum={args.curriculum} "
        f"iterations={args.iterations} seed={args.seed}",
        flush=True,
    )

    tasks = CURRICULUM_TASKS if args.curriculum else [args.task]

    final_prompt = INITIAL_SYSTEM_PROMPT

    for task in tasks:
        task_starting_prompt = final_prompt

        winning_prompt, winning_score, _history = optimize_prompt_for_task(
            task_name=task,
            iterations=args.iterations,
            seed=args.seed,
            starting_prompt=task_starting_prompt,
            client=client,
            model_name=config.model_name,
        )

        print(
            f"\n[VALIDATE] task={task} seeds={args.validation_seeds} "
            "comparing winner against the prompt this task started with...",
            flush=True,
        )
        original_scores = validate_on_seeds(
            task, task_starting_prompt, client, config.model_name, args.validation_seeds
        )
        winning_scores = validate_on_seeds(
            task, winning_prompt, client, config.model_name, args.validation_seeds
        )
        original_mean = _mean(original_scores)
        winning_mean = _mean(winning_scores)
        print(
            f"[VALIDATE] starting prompt: mean={original_mean:.4f} "
            f"per_seed={['%.4f' % s for s in original_scores]}",
            flush=True,
        )
        print(
            f"[VALIDATE] winning  prompt: mean={winning_mean:.4f} "
            f"per_seed={['%.4f' % s for s in winning_scores]}",
            flush=True,
        )

        if winning_mean > original_mean:
            print(
                f"[VALIDATE] task={task} PASSED (mean improved by "
                f"{winning_mean - original_mean:+.4f}); promoting winning prompt.",
                flush=True,
            )
            final_prompt = winning_prompt
        else:
            print(
                f"[VALIDATE] task={task} FAILED (winner did not improve mean "
                "across validation seeds; suspected reward hacking on the inner "
                "seed). Keeping the previous prompt.",
                flush=True,
            )

    bar = "=" * 72
    print(f"\n{bar}\n[FINAL OPTIMIZED SYSTEM PROMPT]\n{bar}", flush=True)
    # repr(...) emits a Python string literal (escaped newlines, escaped quotes)
    # that the developer can paste directly into ``meverse/policy.py``.
    print(repr(final_prompt))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[STOP] reason=keyboard_interrupt", flush=True)
        sys.exit(130)
