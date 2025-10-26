# leach_cluster_sim.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import math, random
import config as C
from round_driver import round_driver, apply_energy_from_events

# ---------- Agent state ----------
@dataclass
class AgentState:
    agent_id: int
    residual_j: float
    is_ch: bool = False
    last_ch_round: int = -10**9
    lq_to_all: Dict[int, float] = field(default_factory=dict)

# ---------- LEACH ----------
@dataclass
class LeachController:
    p_base: float
    alpha: float = 1.0
    p_min: float = 1e-6
    p_max: float = 0.30
    use_lq: bool = False
    epoch_len: Optional[int] = None

    def _epoch_len(self) -> int:
        return self.epoch_len or max(1, math.ceil(1.0 / max(self.p_base, 1e-9)))

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def elect_ch(self, round_idx: int, agents: Dict[int, AgentState],
                 current_ch: Optional[int], e_floor: float,
                 lq_ref_id: Optional[int] = None) -> int:
        epoch_len = self._epoch_len()
        def eligible(a: AgentState) -> bool:
            return (round_idx - a.last_ch_round) >= epoch_len

        Eavg = sum(a.residual_j for a in agents.values()) / max(len(agents), 1)
        scored: List[Tuple[float, int]] = []
        for i, a in agents.items():
            if a.residual_j < e_floor or not eligible(a):
                continue
            p_i = self.p_base * self._clip(self.alpha * (a.residual_j / max(Eavg, 1e-9)), self.p_min, self.p_max)
            denom = max(1e-9, 1.0 - p_i * (round_idx % epoch_len))
            T_i = p_i / denom
            if self.use_lq and lq_ref_id is not None and lq_ref_id in a.lq_to_all:
                T_i *= self._clip(a.lq_to_all.get(lq_ref_id, 1.0), 0.7, 1.3)
            if random.random() < T_i:
                scored.append((T_i, i))

        if not scored:
            cand = [a for a in agents.values() if a.residual_j >= e_floor]
            if not cand:
                return current_ch if current_ch is not None else max(agents.keys())
            return max(cand, key=lambda a: a.residual_j).agent_id
        scored.sort(reverse=True)
        return scored[0][1]

# ---------- Compact round log (participants only) ----------
@dataclass
class RoundLog:
    records: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, round_idx: int, ch_id: int, learners: List[int],
               events: List[Dict[str, Any]], residual_after: Optional[Dict[int, float]] = None):
        per_agent: Dict[int, Dict[str, float]] = {}
        totals = {"CH_compute": 0.0, "CH_comm": 0.0, "Learners_compute": 0.0, "Learners_comm": 0.0, "Total": 0.0}
        participants = set([ch_id]) | set(learners)
        for ev in events:
            if ev["agent_id"] not in participants:
                continue
            a = per_agent.setdefault(ev["agent_id"], {"compute_J": 0.0, "comm_J": 0.0})
            if ev["kind"] == "compute":
                a["compute_J"] += ev["joules"]
                (totals["CH_compute"] if ev["role"] == "CH" else totals["Learners_compute"])
                totals["CH_compute"] += ev["joules"] if ev["role"] == "CH" else 0.0
                totals["Learners_compute"] += ev["joules"] if ev["role"] != "CH" else 0.0
            else:
                a["comm_J"] += ev["joules"]
                totals["CH_comm"] += ev["joules"] if ev["role"] == "CH" else 0.0
                totals["Learners_comm"] += ev["joules"] if ev["role"] != "CH" else 0.0
            totals["Total"] += ev["joules"]

        per_agent_residual = {}
        if residual_after:
            for aid in participants:
                per_agent_residual[aid] = residual_after.get(aid)

        L = max(1, len(learners))
        self.records.append({
            "round_idx": round_idx,
            "ch_id": ch_id,
            "learners": learners[:],
            "agents": {aid: {"compute_J": vals["compute_J"],
                             "comm_J": vals["comm_J"],
                             "residual_J_after": per_agent_residual.get(aid)}
                       for aid, vals in per_agent.items()},
            "totals": {
                **totals,
                "Learner_avg_compute": totals["Learners_compute"] / L,
                "Learner_avg_comm": totals["Learners_comm"] / L
            },
            "warnings": ([] if totals["Total"] > 0.0 else ["ALL_ZERO_ENERGY"])
        })

    def last(self) -> Optional[Dict[str, Any]]:
        return self.records[-1] if self.records else None

# ---------- Cluster ----------
RoundDriverFn = Callable[[int, List[int], List[int], int, int], List[Dict[str, Any]]]
ApplyEnergyFn = Callable[[List[Dict[str, Any]]], Dict[int, float]]

@dataclass
class Cluster:
    agents: Dict[int, AgentState]
    ch_id: Optional[int]
    leach: LeachController
    max_members: int
    e_floor: float
    e_abort: float
    learner_policy: str = C.LEARNER_POLICY
    collector_policy: str = C.COLLECTOR_POLICY
    run_round_fn: RoundDriverFn = round_driver
    apply_energy_fn: ApplyEnergyFn = apply_energy_from_events
    round_log: RoundLog = field(default_factory=RoundLog)

    def _form_members(self, ch_id: int) -> List[int]:
        cands = [a for a in self.agents.values() if a.agent_id != ch_id]
        cands.sort(key=lambda a: (a.lq_to_all.get(ch_id, 1.0), a.residual_j), reverse=True)
        return [a.agent_id for a in cands[:self.max_members]]

    def _select_learners(self, k: int) -> List[int]:
        pool = [a for a in self.agents.values() if a.agent_id != self.ch_id]
        if not pool or k <= 0: return []
        if self.learner_policy == "battery_lowest":
            pool.sort(key=lambda a: a.residual_j)
        elif self.learner_policy == "battery_highest":
            pool.sort(key=lambda a: -a.residual_j)
        elif self.learner_policy == "lq_best":
            pool.sort(key=lambda a: a.lq_to_all.get(self.ch_id, 1.0), reverse=True)
        else:
            random.shuffle(pool)
        return [a.agent_id for a in pool[:k]]

    def _select_collector(self) -> int:
        pool = [a for a in self.agents.values() if a.agent_id != self.ch_id]
        if self.collector_policy == "battery_lowest":
            return min(pool, key=lambda a: a.residual_j).agent_id
        if self.collector_policy == "battery_highest":
            return max(pool, key=lambda a: a.residual_j).agent_id
        if self.collector_policy == "lq_best":
            return max(pool, key=lambda a: a.lq_to_all.get(self.ch_id, 1.0)).agent_id
        return random.choice(pool).agent_id

    def _elect_new_ch(self, round_idx: int):
        new_ch = self.leach.elect_ch(round_idx, self.agents, self.ch_id, self.e_floor,
                                     lq_ref_id=self.ch_id if self.leach.use_lq else None)
        if self.ch_id is not None:
            self.agents[self.ch_id].is_ch = False
        self.ch_id = new_ch
        self.agents[new_ch].is_ch = True
        self.agents[new_ch].last_ch_round = round_idx

    def check(self, round_idx: int):
        if self.ch_id is None:
            self._elect_new_ch(round_idx); return
        ch = self.agents[self.ch_id]
        epoch_len = self.leach._epoch_len()
        epoch_boundary = (round_idx % epoch_len == 0 and round_idx > 0)
        if ch.residual_j < self.e_floor or epoch_boundary:
            self._elect_new_ch(round_idx)

    def step(self, round_idx: int, k_learners: int) -> List[Dict[str, Any]]:
        self.check(round_idx)
        members  = self._form_members(self.ch_id)
        learners = self._select_learners(k_learners)
        collector = self._select_collector()

        events = self.run_round_fn(self.ch_id, members, learners, collector, round_idx)
        consumed = apply_energy_from_events(events)
        for i, j in consumed.items():
            if i in self.agents:
                self.agents[i].residual_j = max(0.0, self.agents[i].residual_j - j)

        # Mid-round abort guard (for next round)
        if self.agents[self.ch_id].residual_j < self.e_abort:
            self._elect_new_ch(round_idx)

        residual_after = {i: a.residual_j for i, a in self.agents.items()}
        self.round_log.append(round_idx, self.ch_id, learners, events, residual_after)
        return events

# ---------- Per-round summarizer (optional) ----------
def summarize_energy_per_round(
    events: List[Dict[str, Any]],
    track_id: Optional[int] = None,
    return_per_member: bool = False,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    per_round_member: Dict[int, Dict[int, Dict[str, float]]] = {}
    for ev in events:
        r = int(ev["round_idx"]); j = float(ev["joules"])
        role, kind, aid = ev["role"], ev["kind"], int(ev["agent_id"])
        out.setdefault(r, {"CH_compute":0.0,"CH_comm":0.0,"Learners_compute":0.0,"Learners_comm":0.0,"Total":0.0})
        per_round_member.setdefault(r, {})
        if role == "CH":
            out[r]["CH_compute"]  += j if kind=="compute" else 0.0
            out[r]["CH_comm"]     += j if kind!="compute" else 0.0
        else:
            out[r]["Learners_compute"] += j if kind=="compute" else 0.0
            out[r]["Learners_comm"]    += j if kind!="compute" else 0.0
            per_round_member[r].setdefault(aid, {"compute":0.0,"comm":0.0})
            per_round_member[r][aid][kind] += j
        out[r]["Total"] += j

    for r, s in out.items():
        m = per_round_member.get(r, {}); k=len(m)
        s["Learner_avg_compute"] = s["Learners_compute"]/k if k else 0.0
        s["Learner_avg_comm"]    = s["Learners_comm"]/k if k else 0.0
        if track_id is not None and track_id in m:
            s["Learner_track_compute"] = m[track_id]["compute"]
            s["Learner_track_comm"]    = m[track_id]["comm"]
        else:
            s["Learner_track_compute"] = None
            s["Learner_track_comm"]    = None
        if return_per_member:
            s["Learners_detail"] = m
    return out
