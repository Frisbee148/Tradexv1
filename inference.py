"""Competition-style inference runner for the surveillance benchmark."""

from __future__ import annotations

import os
from typing import Optional

from meverse import SurveillanceAction, build_llm_client, list_task_names, load_policy_config, policy_label, select_action
from meverse.server.meverse_environment import MarketSurveillanceEnvironment

TASK_NAME = os.getenv("MEVERSE_TASK") or os.getenv("TASK_NAME") or "full_market_surveillance"
BENCHMARK = "amm-market-surveillance"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}", flush=True)


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={reward_text}", flush=True)


def main() -> None:
    task_name = TASK_NAME if TASK_NAME in list_task_names() else "full_market_surveillance"
    demo_mode = env_flag("DEMO_MODE", False)
    eval_mode = False if demo_mode else env_flag("EVAL_MODE", True)
    policy_config = load_policy_config()
    llm_client = build_llm_client(policy_config)
    env = MarketSurveillanceEnvironment(task=task_name, eval_mode=eval_mode, demo_mode=demo_mode)
    observation = env.reset(task=task_name)
    rewards: list[float] = []
    steps = 0

    log_start(task_name, BENCHMARK, policy_label(client=llm_client, config=policy_config))

    try:
        while not observation.done:
            final_action = select_action(
                observation,
                client=llm_client,
                config=policy_config,
                allow_fallback=True,
            )
            observation = env.step(SurveillanceAction(action_type=final_action))
            steps += 1
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            log_step(step=steps, action=final_action, reward=reward, done=observation.done, error=observation.metadata.get("last_action_error"))
        grade = env.grade()
        success = bool(grade["score"] >= 0.6)
    except Exception:
        success = False
        raise
    finally:
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    main()
