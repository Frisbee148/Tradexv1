from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentSignal:
    agent_id: int
    agent_name: str
    is_malicious: bool
    trade_size: float
    time_gap: float
    price_impact: float
    burst_contribution: float
    pattern_contribution: float
    manipulation_contribution: float


class NormalTrader:
    agent_id = 0
    agent_name = "NormalTrader"
    is_malicious = False

    def __init__(self):
        self.position = 0.0
        self.entry_price = 100.0
        self._rng = random.Random()

    def reset(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.position = 0.0
        self.entry_price = 100.0

    def step(
        self,
        price: float,
        step_num: int,
        last_signal: Optional[AgentSignal] = None,
    ) -> AgentSignal:
        if last_signal and last_signal.price_impact > 0.03:
            trade_size = max(1.0, self._rng.gauss(18.0, 4.0))
        else:
            trade_size = max(1.0, self._rng.gauss(10.0, 3.0))

        time_gap = self._rng.uniform(1.5, 4.0)
        price_impact = trade_size * 0.0008 * self._rng.uniform(0.8, 1.2)
        return AgentSignal(
            agent_id=0,
            agent_name="NormalTrader",
            is_malicious=False,
            trade_size=trade_size,
            time_gap=time_gap,
            price_impact=price_impact,
            burst_contribution=0.0,
            pattern_contribution=0.0,
            manipulation_contribution=0.0,
        )


class ManipulatorBot:
    agent_id = 1
    agent_name = "ManipulatorBot"
    is_malicious = True

    def __init__(self, episode: int = 0):
        self.episode = episode
        self.stage = 1 if episode < 100 else (2 if episode < 200 else 3)
        self._rng = random.Random()

    def reset(self, episode: int, seed: Optional[int] = None):
        self.episode = episode
        self.stage = 1 if episode < 100 else (2 if episode < 200 else 3)
        self._rng = random.Random(seed)

    def step(
        self,
        price: float,
        step_num: int,
        last_signal: Optional[AgentSignal] = None,
    ) -> AgentSignal:
        arb_correction_boost = 0.0
        if last_signal and last_signal.agent_id == 2:
            arb_correction_boost = 0.20

        if self.stage == 1:
            trade_size = self._rng.uniform(40.0, 60.0)
            time_gap = self._rng.uniform(0.1, 0.3)
            price_impact = trade_size * 0.002
            burst = min(1.0, 0.85 + self._rng.uniform(0.0, 0.15) + arb_correction_boost)
            pattern = 0.30
            manip = 0.80 + self._rng.uniform(0.0, 0.15)
        elif self.stage == 2:
            trade_size = self._rng.uniform(25.0, 45.0)
            if self._rng.random() > 0.4:
                time_gap = self._rng.uniform(0.2, 0.5)
            else:
                time_gap = self._rng.uniform(0.8, 1.5)
            price_impact = trade_size * 0.0015
            burst = min(1.0, 0.55 + self._rng.uniform(0.0, 0.20) + arb_correction_boost)
            pattern = 0.50 + self._rng.uniform(0.0, 0.15)
            manip = 0.60 + self._rng.uniform(0.0, 0.20)
        else:
            trade_size = max(1.0, self._rng.gauss(12.0, 2.0))
            time_gap = max(1.05, self._rng.gauss(2.0, 0.3))
            price_impact = trade_size * 0.0018
            burst = 0.35 + self._rng.uniform(0.0, 0.15)
            pattern = 0.55 + self._rng.uniform(0.0, 0.15)
            manip = max(0.50, 0.50 + self._rng.uniform(0.0, 0.20))

        return AgentSignal(
            agent_id=1,
            agent_name="ManipulatorBot",
            is_malicious=True,
            trade_size=max(1.0, trade_size),
            time_gap=max(0.05, time_gap),
            price_impact=price_impact,
            burst_contribution=burst,
            pattern_contribution=pattern,
            manipulation_contribution=manip,
        )


class ArbitrageAgent:
    agent_id = 2
    agent_name = "ArbitrageAgent"
    is_malicious = False

    def __init__(self):
        self.fair_value = 100.0
        self._rng = random.Random()

    def reset(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def step(
        self,
        price: float,
        step_num: int,
        last_signal: Optional[AgentSignal] = None,
    ) -> AgentSignal:
        manipulation_pressure = 0.0
        if last_signal and last_signal.manipulation_contribution > 0.5:
            manipulation_pressure = last_signal.manipulation_contribution

        deviation = abs(price - self.fair_value) / self.fair_value
        base_size = deviation * 200.0 + self._rng.gauss(8.0, 2.0)
        trade_size = min(30.0, max(5.0, base_size + manipulation_pressure * 15.0))
        time_gap = self._rng.uniform(0.8, 2.5)
        price_impact = trade_size * 0.0010

        return AgentSignal(
            agent_id=2,
            agent_name="ArbitrageAgent",
            is_malicious=False,
            trade_size=trade_size,
            time_gap=time_gap,
            price_impact=price_impact,
            burst_contribution=0.05,
            pattern_contribution=0.10,
            manipulation_contribution=0.0,
        )


class LiquidityProvider:
    agent_id = 3
    agent_name = "LiquidityProvider"
    is_malicious = False

    def __init__(self):
        self._rng = random.Random()

    def reset(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def step(
        self,
        price: float,
        step_num: int,
        last_signal: Optional[AgentSignal] = None,
    ) -> AgentSignal:
        if last_signal and last_signal.price_impact > 0.02:
            trade_size = max(1.0, self._rng.gauss(9.0, 2.0))
        else:
            trade_size = max(1.0, self._rng.gauss(5.0, 1.0))

        time_gap = max(1.0, self._rng.gauss(2.0, 0.15))
        price_impact = trade_size * 0.0005
        pattern = 0.45 + self._rng.uniform(0.0, 0.15)

        return AgentSignal(
            agent_id=3,
            agent_name="LiquidityProvider",
            is_malicious=False,
            trade_size=trade_size,
            time_gap=time_gap,
            price_impact=price_impact,
            burst_contribution=0.0,
            pattern_contribution=pattern,
            manipulation_contribution=0.0,
        )


class AgentPool:
    def __init__(self, episode: int = 0):
        self.normal_trader = NormalTrader()
        self.manipulator = ManipulatorBot(episode=episode)
        self.arbitrage = ArbitrageAgent()
        self.liquidity_provider = LiquidityProvider()
        self.episode = episode
        self._last_signal: Optional[AgentSignal] = None

    def reset(self, episode: int, seed: Optional[int] = None):
        self.episode = episode
        self._last_signal = None
        base = seed if seed is not None else 42
        self.normal_trader.reset(seed=base)
        self.manipulator.reset(episode=episode, seed=base + 1)
        self.arbitrage.reset(seed=base + 2)
        self.liquidity_provider.reset(seed=base + 3)

    def get_signals(self, price: float, step_num: int, is_suspicious_step: bool) -> Dict[str, object]:
        last = self._last_signal

        if is_suspicious_step:
            primary = self.manipulator.step(price, step_num, last_signal=last)
            background: List[AgentSignal] = [
                self.normal_trader.step(price, step_num, last_signal=last),
                self.arbitrage.step(price, step_num, last_signal=last),
            ]
            if self.manipulator.stage == 3:
                background.append(self.liquidity_provider.step(price, step_num, last_signal=last))
            active_agents = ["ManipulatorBot"]

            burst_boost = primary.burst_contribution
            pattern_boost = primary.pattern_contribution
            manipulation_boost = primary.manipulation_contribution
            dominant_trade_size = primary.trade_size
            dominant_time_gap = primary.time_gap
            self._last_signal = primary
        else:
            background = [
                self.normal_trader.step(price, step_num, last_signal=last),
                self.arbitrage.step(price, step_num, last_signal=last),
                self.liquidity_provider.step(price, step_num, last_signal=last),
            ]
            active_agents = ["NormalTrader", "ArbitrageAgent", "LiquidityProvider"]
            burst_boost = 0.0
            pattern_boost = max(s.pattern_contribution for s in background)
            manipulation_boost = 0.0
            dominant_trade_size = sum(s.trade_size for s in background) / float(len(background))
            dominant_time_gap = sum(s.time_gap for s in background) / float(len(background))
            self._last_signal = background[0]

        return {
            "burst_boost": min(1.0, max(0.0, burst_boost)),
            "pattern_boost": min(1.0, max(0.0, pattern_boost)),
            "manipulation_boost": min(1.0, max(0.0, manipulation_boost)),
            "dominant_trade_size": max(0.0, dominant_trade_size),
            "dominant_time_gap": max(0.05, dominant_time_gap),
            "active_agents": active_agents,
            "manipulator_stage": self.manipulator.stage,
            "is_suspicious": bool(is_suspicious_step),
        }
