# Bot-aware Market Surveillance in Simulated AMM Trading

This project is a simulation environment for reinforcement learning and decision intelligence. It is not a trading bot, DeFi product, wallet, liquidity manager, or blockchain integration demo.

The environment models an AMM-style market and asks an agent to act as a market surveillance controller. At each step, the agent reviews structured signals about recent trading activity and chooses one of four responses:

- `ALLOW`
- `FLAG`
- `BLOCK`
- `MONITOR`

The goal is to identify suspicious bot-like behavior while minimizing harm to normal users and preserving healthy market behavior.

## What The Benchmark Measures

This benchmark is designed as a real-world surveillance and anomaly-detection task:

- detect suspicious bursts of trading activity
- detect repeated manipulation patterns
- avoid false positives on normal activity
- avoid false negatives on harmful activity
- preserve healthy market participation

The benchmark is reward-shaped for partial progress. It does not optimize for profit.

## Observation Space

Each step returns a fixed-size structured observation with surveillance signals:

- `current_amm_price`
- `liquidity_snapshot`
- `recent_trade_count`
- `trades_in_window`
- `trade_frequency`
- `average_trade_size`
- `maximum_trade_size`
- `recent_slippage_impact`
- `time_gap_mean`
- `time_gap_min`
- `recent_time_gaps`
- `recent_price_impacts`
- `burst_indicator`
- `pattern_indicator`
- `suspiciousness_score`
- `manipulation_score`

## Action Space

Only these actions are valid:

- `ALLOW`
- `FLAG`
- `BLOCK`
- `MONITOR`

Legacy trading and liquidity-management actions have been removed from the environment logic.

## Reward Logic

Reward combines:

- positive reward for correctly detecting suspicious behavior
- positive reward for correctly allowing normal activity
- false positive penalties
- false negative penalties
- severity bonuses on harmful suspicious activity
- overblocking penalties to protect healthy market behavior

## Tasks

The repo includes three deterministic tasks with distinct difficulty levels:

1. `burst_detection`
2. `pattern_manipulation_detection`
3. `full_market_surveillance`

- `burst_detection`: learn to catch abrupt high-frequency bursts.
- `pattern_manipulation_detection`: learn repeated timing and size signatures.
- `full_market_surveillance`: balance burst detection, pattern detection, and false-positive control in mixed traffic.

## Baseline Policies

The official competition baseline is the LLM-driven policy in [inference.py](/home/casp1an/Code/TradeX1/inference.py). It uses the OpenAI client together with the required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

The rule-based heuristic in [meverse/baseline_policy.py](/home/casp1an/Code/TradeX1/meverse/baseline_policy.py) is kept for two narrower jobs:

- crash fallback if the LLM call fails
- benchmark comparison to test whether the environment has headroom beyond simple thresholds

The heuristic policy follows simple surveillance rules:

- if pattern score is high and slippage is high, `BLOCK`
- elif manipulation score is high, `BLOCK`
- elif burst score or trade frequency is high, `FLAG`
- elif suspiciousness is moderate, `MONITOR`
- else `ALLOW`

## Running The Environment

Serve the OpenEnv app from the repo root:

```bash
python app.py
```

Validate the environment package directly:

```bash
cd meverse
openenv validate
```

## Running Inference

The root inference runner is [inference.py](/home/casp1an/Code/TradeX1/inference.py). It loads the surveillance environment, runs the LLM baseline, and prints clean competition-style logs. If the upstream model call fails, it falls back to the heuristic policy so the run can still complete.

```bash
python inference.py
```

Optional task selection:

```powershell
$env:MEVERSE_TASK="full_market_surveillance"
python inference.py
```

## Required Environment Variables

`inference.py` reads these variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `EVAL_MODE`
- `DEMO_MODE`

Example PowerShell setup:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your-token"
python inference.py
```

If `HF_TOKEN` is not set, the script cannot use the primary LLM baseline and will label the run as `heuristic-fallback`. For submission, the required API variables should be present. The fallback exists to reduce crash risk, not as the intended submitted baseline.

Mode behavior:

- `EVAL_MODE=true` keeps runs reproducible for competition-style evaluation.
- `DEMO_MODE=true` uses the episode seed to introduce small bounded observation variation for manual testing.
- If `DEMO_MODE=true`, it takes precedence over `EVAL_MODE`.

## Validation And Graders

Validation logic is implemented in [meverse/validation.py](/d:/TradeX/meverse/validation.py). It:

- enumerates all tasks
- runs each task independently
- runs the deterministic grader independently
- prints task-wise scores
- asserts every score satisfies `0.0 <= score <= 1.0`

Run it with:

```bash
python -m meverse.validation
```

To compare the heuristic policy against the official LLM baseline and check whether the environment is too shallow, run:

```bash
python compare_policies.py
```

That script is analysis tooling for benchmark quality. It is not the competition inference entrypoint.

## Verifying Score Range

When you run the validation suite, each task prints a normalized score and the script asserts the range check `0.0 <= score <= 1.0`.

## OpenEnv Metadata

Project metadata for OpenEnv lives in [meverse/openenv.yaml](/d:/TradeX/meverse/openenv.yaml). It now describes the repository as a market surveillance benchmark rather than a trading or liquidity-management environment.
