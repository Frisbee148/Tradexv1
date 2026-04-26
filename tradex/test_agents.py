"""Run with: python tradex/test_agents.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tradex.agents import (
    AgentPool,
    ArbitrageAgent,
    LiquidityProvider,
    ManipulatorBot,
    NormalTrader,
)


def test_standalone_signals():
    _ = NormalTrader  # keep explicit import exercised
    pool = AgentPool(episode=0)
    pool.reset(episode=0, seed=42)

    sig = pool.get_signals(price=100.0, step_num=1, is_suspicious_step=True)
    assert sig["burst_boost"] > 0.5, "Stage 1 ManipulatorBot: burst must be high"
    assert sig["manipulation_boost"] > 0.6, "Stage 1: manipulation must be high"
    assert "ManipulatorBot" in sig["active_agents"]
    print("PASS: Stage 1 ManipulatorBot signals correct")

    sig = pool.get_signals(price=100.0, step_num=2, is_suspicious_step=False)
    assert sig["manipulation_boost"] == 0.0, "Normal step: no manipulation"
    assert "NormalTrader" in sig["active_agents"]
    print("PASS: Normal step signals correct")


def test_stage3_anti_hacking():
    bot = ManipulatorBot(episode=250)
    signal = bot.step(price=100.0, step_num=5)
    assert signal.manipulation_contribution >= 0.50, (
        "Stage 3 must keep manipulation_contribution >= 0.50"
    )
    assert signal.time_gap > 1.0, "Stage 3 must mimic normal timing gaps (> 1.0s)"
    print("PASS: Stage 3 anti-hacking properties correct")


def test_false_positive_trap():
    lp = LiquidityProvider()
    lp.reset(seed=42)
    signal = lp.step(price=100.0, step_num=1)
    assert signal.manipulation_contribution == 0.0, (
        "LiquidityProvider: manipulation must always be 0.0"
    )
    assert signal.pattern_contribution > 0.40, (
        "LiquidityProvider: pattern must be elevated (the trap)"
    )
    print("PASS: LiquidityProvider false positive trap correct")


def test_stage3_vs_lp_distinguishable():
    bot = ManipulatorBot(episode=250)
    lp = LiquidityProvider()
    lp.reset(seed=42)

    bot_signal = bot.step(price=100.0, step_num=5)
    lp_signal = lp.step(price=100.0, step_num=5)

    assert bot_signal.manipulation_contribution >= 0.50
    assert lp_signal.manipulation_contribution == 0.0
    gap = bot_signal.manipulation_contribution - lp_signal.manipulation_contribution
    assert gap >= 0.50, f"Gap must be >= 0.50, got {gap:.3f}"
    print(f"PASS: Stage3 vs LP manipulation gap = {gap:.3f} (must be >= 0.50)")


def test_agent_interaction():
    arb = ArbitrageAgent()
    arb.reset(seed=42)

    from tradex.agents import AgentSignal

    manip_signal = AgentSignal(
        agent_id=1,
        agent_name="ManipulatorBot",
        is_malicious=True,
        trade_size=50.0,
        time_gap=0.2,
        price_impact=0.05,
        burst_contribution=0.9,
        pattern_contribution=0.3,
        manipulation_contribution=0.85,
    )
    normal_signal = AgentSignal(
        agent_id=0,
        agent_name="NormalTrader",
        is_malicious=False,
        trade_size=10.0,
        time_gap=2.0,
        price_impact=0.008,
        burst_contribution=0.0,
        pattern_contribution=0.0,
        manipulation_contribution=0.0,
    )

    arb_after_manip = arb.step(price=100.0, step_num=1, last_signal=manip_signal)
    arb_after_normal = arb.step(price=100.0, step_num=2, last_signal=normal_signal)
    assert arb_after_manip.trade_size > arb_after_normal.trade_size, (
        "ArbitrageAgent must trade larger after ManipulatorBot"
    )
    print(
        f"PASS: ArbitrageAgent interaction - "
        f"after manip: {arb_after_manip.trade_size:.1f}, "
        f"after normal: {arb_after_normal.trade_size:.1f}"
    )


def test_env_still_works():
    try:
        from meverse import SurveillanceAction
        from meverse.server.meverse_environment import MarketSurveillanceEnvironment

        env = MarketSurveillanceEnvironment(task="burst_detection", eval_mode=True)
        _ = env.reset(task="burst_detection", seed=42)
        for _ in range(5):
            _ = env.step(SurveillanceAction(action_type="BLOCK"))
        grade = env.grade()
        assert 0.0 <= grade["score"] <= 1.0, "Score must be in [0, 1]"
        print(f"PASS: Env works - score={grade['score']:.4f}")
    except ImportError:
        print("SKIP: meverse not importable in this context")


def test_active_agents_in_metadata():
    try:
        from meverse import SurveillanceAction
        from meverse.server.meverse_environment import MarketSurveillanceEnvironment

        env = MarketSurveillanceEnvironment(task="burst_detection", eval_mode=True)
        _ = env.reset(task="burst_detection", seed=42)
        obs = env.step(SurveillanceAction(action_type="ALLOW"))
        active = obs.metadata.get("active_agents", "NOT_INJECTED")
        stage = obs.metadata.get("manipulator_stage", "NOT_INJECTED")
        print(f"Active agents in metadata: {active}")
        print(f"Manipulator stage in metadata: {stage}")
        if active != "NOT_INJECTED":
            print("PASS: Agent metadata injection working")
        else:
            print("INFO: Agent metadata not yet surfaced - check amm.py injection")
    except ImportError:
        print("SKIP: meverse not importable in this context")


if __name__ == "__main__":
    print("=" * 50)
    print("TradeX Multi-Agent Test Suite")
    print("=" * 50)
    test_standalone_signals()
    test_stage3_anti_hacking()
    test_false_positive_trap()
    test_stage3_vs_lp_distinguishable()
    test_agent_interaction()
    test_env_still_works()
    test_active_agents_in_metadata()
    print("=" * 50)
    print("All tests complete.")
