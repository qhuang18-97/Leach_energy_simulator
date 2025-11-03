# energy_tracking.py
from typing import Dict, List, Iterable, Optional
from dataclasses import dataclass, field

@dataclass
class EnergyTracker:
    """
    Tracks residual energy per agent across rounds.
    Stores:
      - residuals[agent_id] -> list of residual J at each round index
      - ch_per_round[round_idx] -> ch_id
      - learners_per_round[round_idx] -> list of learner ids
    """
    residuals: Dict[int, List[float]] = field(default_factory=dict)
    ch_per_round: Dict[int, int] = field(default_factory=dict)
    learners_per_round: Dict[int, List[int]] = field(default_factory=dict)

    def bootstrap(self, agent_ids: Iterable[int], initial_j: float) -> None:
        for aid in agent_ids:
            self.residuals[aid] = [initial_j]  # round -1 baseline (optional)

    def record_round(
        self,
        round_idx: int,
        agents_residual_after: Dict[int, float],
        ch_id: int,
        learners: List[int]
    ) -> None:
        # Ensure lists are contiguous: append one value per round for each agent
        for aid, rj in agents_residual_after.items():
            series = self.residuals.setdefault(aid, [])
            # If this agent has fewer samples than round_idx, pad with last known value
            while len(series) < round_idx:
                series.append(series[-1] if series else rj)
            if len(series) == round_idx:
                series.append(rj)
            else:
                # overwrite if re-recording same round
                series[round_idx] = rj

        self.ch_per_round[round_idx] = ch_id
        self.learners_per_round[round_idx] = list(learners)

    def get_agent_series(self, agent_id: int) -> List[float]:
        return self.residuals.get(agent_id, [])

    def get_round_snapshot(self, round_idx: int) -> Dict[int, float]:
        return {aid: series[round_idx] for aid, series in self.residuals.items()
                if round_idx < len(series)}

    def to_csv(self, path: str) -> None:
        # Wide format: columns = agent_id, r0, r1, r2, ...
        # For 700 agents × R rounds this stays reasonable.
        # If you prefer long format, let me know.
        # Build header
        max_len = max((len(s) for s in self.residuals.values()), default=0)
        with open(path, "w", encoding="utf-8") as f:
            header = ["agent_id"] + [f"r{r}" for r in range(max_len)]
            f.write(",".join(header) + "\n")
            for aid, series in sorted(self.residuals.items()):
                row = [str(aid)] + [f"{v:.6f}" for v in series]
                f.write(",".join(row) + "\n")

    def participants_csv(self, path: str) -> None:
        # Round-wise role log: round, ch_id, learners...
        with open(path, "w", encoding="utf-8") as f:
            f.write("round_idx,ch_id,learners\n")
            for r in sorted(self.ch_per_round.keys()):
                learners_str = " ".join(map(str, self.learners_per_round.get(r, [])))
                f.write(f"{r},{self.ch_per_round[r]},{learners_str}\n")
