"""Competition-style inference runner for the surveillance benchmark."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from meverse.env import load_repo_env
from meverse import SurveillanceAction
from meverse.server.meverse_environment import MarketSurveillanceEnvironment

load_repo_env()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TASK_NAME = os.getenv("MEVERSE_TASK") or os.getenv("TASK_NAME") or "full_market_surveillance"
BENCHMARK = "amm-market-surveillance"
TRAIN_EPISODES = int(os.getenv("TRAIN_EPISODES", "300"))
TRAIN_BASE_SEED = int(os.getenv("TRAIN_BASE_SEED", "42"))
MAX_EPISODE_STEPS_OVERRIDE = int(os.getenv("MEVERSE_MAX_EPISODE_STEPS", "0") or "0")
API_TIMEOUT_S = float(os.getenv("API_TIMEOUT_S", "20"))
API_MAX_ATTEMPTS = max(1, int(os.getenv("API_MAX_ATTEMPTS", "1")))
_OPENAI_CLIENT: Optional[OpenAI] = None


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


TASK_DIFFICULTY = {
    "burst_detection": ("EASY", "\033[92m"),  # green
    "pattern_manipulation_detection": ("MEDIUM", "\033[93m"),  # yellow
    "full_market_surveillance": ("HARD", "\033[91m"),  # red
}
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"


def _tag(task: str) -> str:
    diff, color = TASK_DIFFICULTY.get(task, ("?", ""))
    return f"{color}{BOLD}[{diff:<6}]{RESET}"


def log_start(task: str, env: str, model: str) -> None:
    diff, color = TASK_DIFFICULTY.get(task, ("?", ""))
    print(
        f"{color}{BOLD}[{diff:<6}]{RESET} task={task} {DIM}env={env} model={model}{RESET}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    if not env_flag("VERBOSE_STEPS", False):
        return
    error_value = error if error else "null"
    print(
        f"  step={step:>3} action={action:<7} reward={reward:+.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: list[float], action_counts: dict[str, int]) -> None:
    diff, color = TASK_DIFFICULTY.get(task, ("?", ""))
    avg = (sum(rewards) / len(rewards)) if rewards else 0.0
    total = sum(rewards) if rewards else 0.0
    hist = " ".join(f"{a}={action_counts.get(a, 0)}" for a in ("ALLOW", "FLAG", "BLOCK", "MONITOR"))
    status = f"{color}OK{RESET}" if success else f"{DIM}--{RESET}"
    print(
        f"{color}{BOLD}[{diff:<6}]{RESET} {status} "
        f"steps={steps:>3} score={score:.3f} avg_r={avg:+.2f} sum_r={total:+.2f} | {hist}",
        flush=True,
    )


def log_phase_banner(task: str) -> None:
    diff, color = TASK_DIFFICULTY.get(task, ("?", ""))
    bar = "=" * 72
    print(f"\n{color}{bar}{RESET}", flush=True)
    print(f"{color}{BOLD}  {diff} TASK :: {task}{RESET}", flush=True)
    print(f"{color}{bar}{RESET}", flush=True)


def build_signal_snapshot(observation) -> dict[str, Any]:
    return {
        "task_name": observation.task_name,
        "step_num": observation.step_num,
        "max_steps": observation.max_steps,
        "done": observation.done,
        "reward": float(observation.reward or 0.0),
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
        "metadata": {
            "episode_id": observation.metadata.get("episode_id"),
            "seed": observation.metadata.get("seed"),
            "eval_mode": observation.metadata.get("eval_mode"),
            "demo_mode": observation.metadata.get("demo_mode"),
            "scenario_note": observation.metadata.get("scenario_note"),
            "amm_price": observation.metadata.get("amm_price"),
            "amm_liquidity": observation.metadata.get("amm_liquidity"),
            "bot_confidence": observation.metadata.get("bot_confidence"),
            "last_action_error": observation.metadata.get("last_action_error"),
            # Internal hint scores kept on metadata only — not part of the
            # agent-facing observation but preserved for telemetry/debug.
            "burst_indicator": observation.metadata.get("burst_indicator"),
            "pattern_indicator": observation.metadata.get("pattern_indicator"),
        },
    }


class DebugTelemetryWriter:
    """Write detailed episode telemetry to JSONL without touching stdout logs."""

    def __init__(self, enabled: bool, task_name: str):
        self.enabled = enabled
        self.path: Optional[Path] = None
        if not enabled:
            return
        configured_path = os.getenv("DEBUG_TELEMETRY_PATH", "").strip()
        if configured_path:
            self.path = Path(configured_path)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.path = Path("telemetry") / f"{task_name}-{stamp}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            # Telemetry must never interfere with the benchmark run.
            pass


def llm_action(client: OpenAI, observation) -> str:
    prompt = {
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
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=12,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a market surveillance controller in a simulated AMM market. "
                    "Your job: detect suspicious bot-like activity and respond decisively, "
                    "but NEVER penalize normal healthy trading.\n"
                    "Actions: ALLOW, FLAG, BLOCK, MONITOR.\n"
                    "Reason from raw signals: trades_in_window, recent_time_gaps, "
                    "time_gap_min, trade_frequency, recent_price_impacts, "
                    "suspiciousness_score, manipulation_score.\n"
                    "Acting earlier when truly suspicious is rewarded more than acting late, "
                    "but acting on noise is penalized — balance speed against evidence.\n"
                    "Rules (apply in order, stop at first match):\n"
                    "1. manipulation_score >= 0.55 → BLOCK\n"
                    "2. trade_frequency >= 7.0 AND time_gap_min < 0.5 → BLOCK\n"
                    "3. suspiciousness_score >= 0.55 AND any recent_price_impacts > 0.03 → BLOCK\n"
                    "4. suspiciousness_score >= 0.45 AND time_gap_min < 0.8 → FLAG\n"
                    "5. trade_frequency >= 5.0 AND average_trade_size > 18 → FLAG\n"
                    "6. suspiciousness_score >= 0.35 → MONITOR\n"
                    "7. Otherwise → ALLOW\n"
                    "Return JSON only: {\"action\": \"ALLOW\"}"
                ),
            },
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(content)
    action = str(parsed.get("action", "")).strip().upper()
    if action in {"ALLOW", "FLAG", "BLOCK", "MONITOR"}:
        return action
    raise ValueError(f"LLM returned invalid action: {action!r}")


def heuristic_action(observation) -> str:
    """Local fallback policy mirroring the baseline LLM rule chain."""
    if observation.manipulation_score >= 0.55:
        return "BLOCK"
    if observation.trade_frequency >= 7.0 and observation.time_gap_min < 0.5:
        return "BLOCK"
    if observation.suspiciousness_score >= 0.55 and any(x > 0.03 for x in observation.recent_price_impacts):
        return "BLOCK"
    if observation.suspiciousness_score >= 0.45 and observation.time_gap_min < 0.8:
        return "FLAG"
    if observation.trade_frequency >= 5.0 and observation.average_trade_size > 18:
        return "FLAG"
    if observation.suspiciousness_score >= 0.35:
        return "MONITOR"
    return "ALLOW"


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
            timeout=API_TIMEOUT_S,
            max_retries=0,
        )
    return _OPENAI_CLIENT


def select_action(observation) -> str:
    if not HF_TOKEN:
        return heuristic_action(observation)
    client = _get_openai_client()
    last_err = None
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            return llm_action(client, observation)
        except Exception as exc:
            last_err = exc
            err_str = str(exc)
            if "402" in err_str or "401" in err_str or "403" in err_str:
                break
    return heuristic_action(observation)


def run_task(task_name: str, seed: Optional[int] = None) -> dict[str, Any]:
    """Run a single task episode: reset, step through, grade, and log."""
    demo_mode = env_flag("DEMO_MODE", False)
    eval_mode = False if demo_mode else env_flag("EVAL_MODE", True)
    env = MarketSurveillanceEnvironment(task=task_name, eval_mode=eval_mode, demo_mode=demo_mode)
    observation = env.reset(task=task_name, seed=seed)
    telemetry = DebugTelemetryWriter(enabled=env_flag("DEBUG_TELEMETRY", False), task_name=task_name)
    rewards: list[float] = []
    action_counts: dict[str, int] = {"ALLOW": 0, "FLAG": 0, "BLOCK": 0, "MONITOR": 0}
    steps = 0
    score = 0.0
    success = False
    final_grade: Optional[dict[str, float]] = None
    declared_max = max(1, int(getattr(observation, "max_steps", 1)))
    max_episode_steps = (
        min(declared_max, MAX_EPISODE_STEPS_OVERRIDE)
        if MAX_EPISODE_STEPS_OVERRIDE > 0
        else declared_max
    )

    log_start(task_name, BENCHMARK, MODEL_NAME)
    telemetry.write(
        "episode_start",
        {
            "task": task_name,
            "benchmark": BENCHMARK,
            "model": MODEL_NAME,
            "initial_observation": build_signal_snapshot(observation),
            "environment": env.debug_snapshot(),
            "seed": seed,
        },
    )

    try:
        while not observation.done:
            # Safety guard: never allow an episode to run beyond its declared horizon.
            if steps >= max_episode_steps:
                break
            decision_observation = observation
            pre_action_debug = env.debug_snapshot()
            final_action = select_action(observation)
            observation = env.step(SurveillanceAction(action_type=final_action))
            steps += 1
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            action_counts[final_action] = action_counts.get(final_action, 0) + 1
            telemetry.write(
                "step",
                {
                    "step": steps,
                    "action": final_action,
                    "reward": reward,
                    "done": observation.done,
                    "decision_observation": build_signal_snapshot(decision_observation),
                    "returned_observation": build_signal_snapshot(observation),
                    "pre_action_environment": pre_action_debug,
                    "post_action_environment": env.debug_snapshot(),
                },
            )
            log_step(step=steps, action=final_action, reward=reward, done=observation.done, error=observation.metadata.get("last_action_error"))
        final_grade = env.grade()
        score = final_grade["score"]
        success = bool(score >= 0.6)
    except KeyboardInterrupt:
        success = False
        telemetry.write(
            "episode_error",
            {
                "steps_completed": steps,
                "rewards": rewards,
                "reason": "keyboard_interrupt",
            },
        )
        raise
    except BaseException:
        success = False
        telemetry.write(
            "episode_error",
            {
                "steps_completed": steps,
                "rewards": rewards,
            },
        )
        raise
    finally:
        try:
            final_grade = env.grade() if steps > 0 else final_grade
        except Exception:
            final_grade = None
        if final_grade:
            score = final_grade["score"]
        telemetry.write(
            "episode_end",
            {
                "success": success,
                "steps": steps,
                "rewards": rewards,
                "grade": final_grade,
                "telemetry_path": str(telemetry.path) if telemetry.path else None,
            },
        )
        log_end(task=task_name, success=success, steps=steps, score=score, rewards=rewards, action_counts=action_counts)
    return {
        "task": task_name,
        "seed": seed,
        "steps": steps,
        "score": score,
        "grade": final_grade,
    }


def get_task(episode: int) -> str:
    """Curriculum scheduler: easy -> medium -> hard (100 episodes each)."""
    if episode < 100:
        return "burst_detection"
    if episode < 200:
        return "pattern_manipulation_detection"
    return "full_market_surveillance"


def run_training_curriculum(total_episodes: int = TRAIN_EPISODES, base_seed: int = TRAIN_BASE_SEED) -> None:
    """Run 300 episodes distributed evenly across task difficulty levels."""
    task_counts = {
        "burst_detection": 0,
        "pattern_manipulation_detection": 0,
        "full_market_surveillance": 0,
    }
    episode_budget = max(1, int(total_episodes))
    print(f"[TRAINING] total_episodes={episode_budget} base_seed={base_seed}", flush=True)
    print(
        "[CURRICULUM] "
        "episodes 0-99=burst_detection, "
        "100-199=pattern_manipulation_detection, "
        "200+=full_market_surveillance",
        flush=True,
    )
    completed = 0
    try:
        for episode in range(episode_budget):
            if episode < 100:
                task_name = "burst_detection"
                phase = "easy"
            elif episode < 200:
                task_name = "pattern_manipulation_detection"
                phase = "medium"
            else:
                task_name = "full_market_surveillance"
                phase = "hard"

            if episode in {0, 100, 200}:
                print(f"[PHASE] episode={episode} difficulty={phase} task={task_name}", flush=True)

            seed = base_seed + episode
            result = run_task(task_name=task_name, seed=seed)
            task_counts[task_name] += 1
            completed = episode + 1
            print(
                f"[EPISODE] idx={episode} phase={phase} task={task_name} seed={seed} "
                f"steps={result['steps']} score={result['score']:.4f}",
                flush=True,
            )
    except KeyboardInterrupt:
        print(
            f"[STOP] reason=keyboard_interrupt completed_episodes={completed} "
            f"budget={episode_budget}",
            flush=True,
        )
        return
    print(
        "[TRAINING_SUMMARY] "
        f"total_episodes={episode_budget} "
        f"burst_detection={task_counts['burst_detection']} "
        f"pattern_manipulation_detection={task_counts['pattern_manipulation_detection']} "
        f"full_market_surveillance={task_counts['full_market_surveillance']}",
        flush=True,
    )
    print(f"[STOP] reason=budget_reached completed_episodes={completed}", flush=True)


def main() -> None:
    """OpenEnv evaluation entrypoint - 50 episodes per task."""
    tasks = [
        "burst_detection",
        "pattern_manipulation_detection",
        "full_market_surveillance",
    ]
    episodes_per_task = 50

    aggregate: dict[str, list[float]] = {t: [] for t in tasks}
    for task in tasks:
        log_phase_banner(task)
        for i in range(episodes_per_task):
            seed = TRAIN_BASE_SEED + i
            result = run_task(task_name=task, seed=seed)
            aggregate[task].append(float(result.get("score", 0.0)))

    print(f"\n{BOLD}=== SUMMARY ==={RESET}", flush=True)
    for task in tasks:
        scores = aggregate[task]
        diff, color = TASK_DIFFICULTY.get(task, ("?", ""))
        mean = (sum(scores) / len(scores)) if scores else 0.0
        best = max(scores) if scores else 0.0
        worst = min(scores) if scores else 0.0
        print(
            f"{color}{BOLD}[{diff:<6}]{RESET} {task:<35} "
            f"n={len(scores)} mean={mean:.3f} min={worst:.3f} max={best:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[STOP] reason=keyboard_interrupt", flush=True)
