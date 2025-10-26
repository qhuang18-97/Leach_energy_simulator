from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
import importlib
import random

# ====================================================
# Configs & Events
# ====================================================

@dataclass
class SimConfig:
    num_agents: int = 700
    num_rounds: int = 700
    ch_index: int = 0

    # Downlink behaviors
    use_broadcast_downlink: bool = False          # for global weights downlink
    broadcast_training_batch: bool = True         # NEW: broadcast for training batches

    ch_trains_too: bool = True
    uav_track_idx: int = 1
    max_learners_per_round: int = 16
    bootstrap_collectors: int = 14

    # --- NEW: batch/trajectory sizing (float32 by default) ---
    obs_size: int = 100                           # your observation size
    batch_size: int = 64                          # your learner minibatch
    bits_per_element: int = 32                    # 32 for fp32, 8 for int8, etc.
    num_batch_updates: int = 50                   # CH pushes 50 batches/round
    traj_steps: int = 11                          # average of [10,12]

    # --- Flags ---
    exclude_departed_agent_tx: bool = True
    model_server_step: bool = True
    log_events: bool = True

    # --- NEW: MAC/PHY efficiency for payload inflation ---
    mac_efficiency: float = 0.85


@dataclass
class EnergyEvent:
    round_idx: int            # FL round index (bootstrap uses -1)
    event_in_round: int       # ordering within the round (0,1,2,...)
    role: str                 # "CH" or "Learner"
    agent_id: int
    phase: str                # "collect", "train", "agg", "distribute"
    kind: str                 # "compute" or "comm"
    direction: str            # "", "tx", "rx"
    bits: int
    joules: float
    note: str

def _log(evts: List[EnergyEvent], **kw):
    evts.append(EnergyEvent(**kw))

# ====================================================
# Link / payload helpers (unchanged)
# ====================================================

class LinkModel:
    """
    Simple radio/link model used by the default energy method.
    """
    def __init__(self, p_tx_w=2.0, p_rx_w=1.0, r_up=1e6, r_dn=1e6):
        self.p_tx_w = p_tx_w  # TX power (W)
        self.p_rx_w = p_rx_w  # RX power (W)
        self.r_up = r_up      # uplink bitrate (b/s)
        self.r_dn = r_dn      # downlink bitrate (b/s)

    def sample_rates(self):
        # Hook for variability; keep constant for now
        return (self.r_up, self.r_dn)

def payload_bits_for_records(records: int, rec_bytes: int, compression_ratio: float) -> int:
    return int(records * rec_bytes * 8 * compression_ratio)

# ====================================================
# Energy method interface + loader
# ====================================================

class EnergyMethod:
    """
    Interface for all energy calculators.

    Implementations MUST provide:
      - compute_train()                                -> J
      - upload_data_member_tx(bits, link)              -> J
      - upload_data_ch_rx(bits, link)                  -> J
      - member_batch_rx(bits, link)                    -> J
      - ch_batch_tx(bits, link, k_members, broadcast)  -> J
      - member_weights_tx(bits, link)                  -> J
      - ch_weights_rx(bits, link, k_members)           -> J
      - ch_to_server_tx(bits, link)                    -> J
      - server_to_ch_rx(bits, link)                    -> J
    """

    # ---- compute ----
    def compute_train(self) -> float:
        raise NotImplementedError

    # ---- comm primitives for data uploads (trajectories) ----
    def upload_data_member_tx(self, bits: int, link: LinkModel) -> float:
        raise NotImplementedError

    def upload_data_ch_rx(self, bits: int, link: LinkModel) -> float:
        raise NotImplementedError

    # ---- comm primitives for per-learner payloads ----
    def member_batch_rx(self, bits: int, link: LinkModel) -> float:
        raise NotImplementedError

    def ch_batch_tx(self, bits: int, link: LinkModel, k_members: int, broadcast: bool) -> float:
        raise NotImplementedError

    # ---- comm primitives for weights (aggregation) ----
    def member_weights_tx(self, bits: int, link: LinkModel) -> float:
        raise NotImplementedError

    def ch_weights_rx(self, bits: int, link: LinkModel, k_members: int) -> float:
        raise NotImplementedError

    def ch_to_server_tx(self, bits: int, link: LinkModel) -> float:
        raise NotImplementedError

    def server_to_ch_rx(self, bits: int, link: LinkModel) -> float:
        raise NotImplementedError

# ---------------------------------------------------------------------
# Device profiles and device-aware compute energy
# ---------------------------------------------------------------------
from dataclasses import dataclass

@dataclass
class DeviceProfile:
    name: str
    fp32_tflops: float   # effective sustained TFLOPS during training (not peak if you have a better measure)
    avg_power_w: float   # average training power (W)
    safety_time_scale: float = 3.0  # kernel/mem overhead cushion applied to FLOP time

# Predefined devices
GTX_1080 = DeviceProfile(name="GTX_1080", fp32_tflops=8.87, avg_power_w=126.0, safety_time_scale=3.0)
AGX_ORIN  = DeviceProfile(name="AGX_ORIN",  fp32_tflops=5.30, avg_power_w=30.0,  safety_time_scale=3.0)

# ---------------------------------------------------------------------
# Parametric energy model + device-aware compute for training/aggregation
# ---------------------------------------------------------------------
class ParametricEnergy(EnergyMethod):
    """
    Adds:
      • device_profile: choose AGX_ORIN (default) or GTX_1080 (legacy) for training/aggregation energy.
      • flops_per_step: per learner optimizer step FLOPs (forward+backward+opt).
      • steps_per_round: local steps per round (epochs * batches).
      • agg_flops_per_param: FLOPs to aggregate one parameter from one member (≈ 1–3 typical).
      • model_num_params: total trainable parameter count (if you want aggregation energy).
      • server_update_flops: optional server-side update FLOPs per round.

    All communication energy APIs remain the same.
    """

    def __init__(
            self,
            *,
            # ---- Device / compute ----
            device_profile: DeviceProfile = AGX_ORIN,  # Orin default
            flops_per_step: float | None = None,
            steps_per_round: int = 0,
            e_train: float | None = None,
            # ---- Aggregation compute ----
            model_num_params: int = 0,
            agg_flops_per_param: float = 1.0,
            server_update_flops: float = 0.0,
            # ---- Radio energy (defaults = 802.11n, R=30 Mbps) ----
            e_tx_per_bit: float | None = 1.28 / 30e6,  # 4.2666667e-08 J/bit
            e_rx_per_bit: float | None = 0.94 / 30e6,  # 3.1333333e-08 J/bit
            e_tx_per_bit_srv: float | None = None,
            e_rx_per_bit_srv: float | None = None,
            ctrl_tx_bits: int = 0,
            ctrl_rx_bits: int = 0,
            mac_efficiency: float = 0.85,  # inflate payload by 1/η
            scale_broadcast: float = 1.0,
    ):
        # compute knobs
        self.device = device_profile
        self.flops_per_step = flops_per_step
        self.steps_per_round = int(steps_per_round)
        self._e_train_override = e_train

        # aggregation knobs
        self.model_num_params = int(model_num_params)
        self.agg_flops_per_param = float(agg_flops_per_param)
        self.server_update_flops = float(server_update_flops)

        # radio knobs
        self.e_tx_per_bit = e_tx_per_bit
        self.e_rx_per_bit = e_rx_per_bit
        self.e_tx_per_bit_srv = e_tx_per_bit_srv
        self.e_rx_per_bit_srv = e_rx_per_bit_srv
        self.ctrl_tx_bits = int(ctrl_tx_bits)
        self.ctrl_rx_bits = int(ctrl_rx_bits)
        self.eta = float(mac_efficiency)
        self.scale_broadcast = float(scale_broadcast)

    # ---- helpers: compute ----------------------------------------------------
    def _compute_energy_from_flops(self, flops: float) -> float:
        if self.device.fp32_tflops <= 0:
            return 0.0
        # time = FLOPs / (TFLOPS * 1e12) * safety
        t = (flops / (self.device.fp32_tflops * 1e12)) * self.device.safety_time_scale
        return self.device.avg_power_w * t  # Joules

    def _training_energy(self) -> float:
        # Priority: explicit override > FLOPs-based > 0
        if self._e_train_override is not None:
            return float(self._e_train_override)
        if self.flops_per_step is None or self.steps_per_round <= 0:
            return 0.0
        total_flops = self.flops_per_step * self.steps_per_round
        return self._compute_energy_from_flops(total_flops)

    def _aggregation_energy(self, k_members: int) -> float:
        if self.model_num_params <= 0 or self.agg_flops_per_param <= 0 or k_members <= 0:
            return 0.0
        # Simple O(P * k) accumulation model at CH
        flops_ch = self.model_num_params * self.agg_flops_per_param * k_members
        e_ch = self._compute_energy_from_flops(flops_ch)

        # Optional server-side update (e.g., FedAvg on server)
        e_srv = self._compute_energy_from_flops(self.server_update_flops) if self.server_update_flops > 0 else 0.0
        return e_ch + e_srv

    # ---- helpers: radio ------------------------------------------------------
    def _tx_energy(self, bits: int, link: LinkModel, *, use_srv: bool = False) -> float:
        payload_bits = int(round(bits / max(self.eta, 1e-9)))
        total_bits = payload_bits + self.ctrl_tx_bits
        if use_srv and self.e_tx_per_bit_srv is not None:
            return total_bits * self.e_tx_per_bit_srv
        if self.e_tx_per_bit is not None:
            return total_bits * self.e_tx_per_bit
        r_up, _ = link.sample_rates()
        time_s = total_bits / max(r_up, 1e-9)
        return link.p_tx_w * time_s

    def _rx_energy(self, bits: int, link: LinkModel, *, use_srv: bool = False) -> float:
        payload_bits = int(round(bits / max(self.eta, 1e-9)))
        total_bits = payload_bits + self.ctrl_rx_bits
        if use_srv and self.e_rx_per_bit_srv is not None:
            return total_bits * self.e_rx_per_bit_srv
        if self.e_rx_per_bit is not None:
            return total_bits * self.e_rx_per_bit
        _, r_dn = link.sample_rates()
        time_s = total_bits / max(r_dn, 1e-9)
        return link.p_rx_w * time_s

    # ---- compute: local training (per learner, per round) --------------------
    def compute_train(self) -> float:
        return self._training_energy()

    # ---- trajectory uploads (unchanged) --------------------------------------
    def upload_data_member_tx(self, bits: int, link: LinkModel) -> float:
        return self._tx_energy(bits, link, use_srv=False)

    def upload_data_ch_rx(self, bits: int, link: LinkModel) -> float:
        return self._rx_energy(bits, link, use_srv=False)

    # ---- per-learner batch payloads (unchanged comm) -------------------------
    def member_batch_rx(self, bits: int, link: LinkModel) -> float:
        return self._rx_energy(bits, link, use_srv=False)

    def ch_batch_tx(self, bits: int, link: LinkModel, k_members: int, broadcast: bool) -> float:
        if broadcast:
            return self._tx_energy(bits, link, use_srv=False) * self.scale_broadcast
        return self._tx_energy(bits * k_members, link, use_srv=False)

    # ---- weight upload (member -> CH) + CH aggregation compute ----------------
    def member_weights_tx(self, bits: int, link: LinkModel) -> float:
        # member TX only (comm); its local compute already accounted in compute_train()
        return self._tx_energy(bits, link, use_srv=False)

    def ch_weights_rx(self, bits: int, link: LinkModel, k_members: int) -> float:
        # CH RX comm + aggregation compute at CH (+ optional server compute later)
        e_comm = self._rx_energy(bits * k_members, link, use_srv=False)
        e_agg  = self._aggregation_energy(k_members)
        return e_comm + e_agg

    # ---- CH ↔ Server (comm) + optional server compute already counted above ---
    def ch_to_server_tx(self, bits: int | None, link: LinkModel) -> float:
        b = 0 if bits is None else int(bits)
        return self._tx_energy(b, link, use_srv=True)

    def server_to_ch_rx(self, bits: int | None, link: LinkModel) -> float:
        b = 0 if bits is None else int(bits)
        return self._rx_energy(b, link, use_srv=True)

class SimpleAssumptionEnergy(EnergyMethod):
    """
    Drop-in replacement that reproduces your current "simple/manual" assumptions.
    Parameters:
      - e_train: fixed compute energy per local training (J)
      - weight_bits: model weight payload size (bits)
    Uses LinkModel's p_tx_w/p_rx_w and r_up/r_dn for comm calculations.
    """
    def __init__(self, e_train: float = 25.0, weight_bits: int = 8_000_000):
        self._e_train = float(e_train)
        self._weight_bits = int(weight_bits)

    # ---- compute ----
    def compute_train(self) -> float:
        return self._e_train

    # ---- data upload (trajectory) ----
    def upload_data_member_tx(self, bits: int, link: LinkModel) -> float:
        r_up, _ = link.sample_rates()
        return bits * (link.p_tx_w / r_up)

    def upload_data_ch_rx(self, bits: int, link: LinkModel) -> float:
        r_up, _ = link.sample_rates()
        return bits * (link.p_rx_w / r_up)

    # ---- payloads (optional trainer data) ----
    def member_batch_rx(self, bits: int, link: LinkModel) -> float:
        _, r_dn = link.sample_rates()
        return bits * (link.p_rx_w / r_dn)

    def ch_batch_tx(self, bits: int, link: LinkModel, k_members: int, broadcast: bool) -> float:
        r_up, _ = link.sample_rates()
        count = 1 if broadcast else k_members
        return bits * count * (link.p_tx_w / r_up)

    # ---- weights (aggregation) ----
    def member_weights_tx(self, bits: int, link: LinkModel) -> float:
        r_up, _ = link.sample_rates()
        return bits * (link.p_tx_w / r_up)

    def ch_weights_rx(self, bits: int, link: LinkModel, k_members: int) -> float:
        r_up, _ = link.sample_rates()
        return bits * k_members * (link.p_rx_w / r_up)

    def ch_to_server_tx(self, bits: Optional[int], link: LinkModel) -> float:
        bits = self._weight_bits if bits is None else bits
        r_up, _ = link.sample_rates()
        return bits * (link.p_tx_w / r_up)

    def server_to_ch_rx(self, bits: Optional[int], link: LinkModel) -> float:
        bits = self._weight_bits if bits is None else bits
        _, r_dn = link.sample_rates()
        return bits * (link.p_rx_w / r_dn)


def load_energy_method(spec: Optional[str] = None, **kwargs) -> EnergyMethod:
    """
    Load an energy method by string spec.

    Examples:
      - None or "simple" -> SimpleAssumptionEnergy(**kwargs)
      - "pkg.module:ClassName" -> dynamic import and instantiate with **kwargs

    kwargs pass directly to the class constructor.
    """
    if spec is None or spec.strip().lower() == "simple":
        return SimpleAssumptionEnergy(**kwargs)

    # dynamic import: "module.submodule:ClassName"
    if ":" not in spec:
        raise ValueError("Energy method spec must be 'module.path:ClassName' or 'simple'.")
    module_path, class_name = spec.split(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)

def bits_for_batch_updates(batch: int, obs: int, bits_per_elem: int, eta: float, num_updates: int) -> int:
    # total bits for all updates in the round per learner
    C = (2 * bits_per_elem) / max(eta, 1e-9)   # 2*(obs+1)*bits, inflated by 1/η later
    per_update = int(batch * C * (obs + 1))
    return num_updates * per_update

def bits_for_trajectory(steps: int, obs: int, bits_per_elem: int, eta: float) -> int:
    C = (2 * bits_per_elem) / max(eta, 1e-9)
    return int(steps * C * (obs + 1))


# ====================================================
# Simulation
# ====================================================

def simulate_and_report(sim: SimConfig,
                        link: LinkModel,
                        tdata,
                        # You can pick the energy method:
                        energy_method: Optional[EnergyMethod] = None,
                        # Or request loading by name and kwargs:
                        energy_method_spec: Optional[str] = None,
                        energy_method_kwargs: Optional[Dict[str, Any]] = None,
                        rec_bytes: int = 1024,
                        # Back-compat knobs for the default method:
                        weight_bits: int = 8_000_000,
                        e_train: float = 25.0) -> Dict[str, Any]:
    """
    Simulate FL comm/compute with per-round event logs.
    All energy math is delegated to `energy_method`.
    """
    # Prepare energy method
    if energy_method is None:
        kwargs = energy_method_kwargs or {}
        # If caller didn't ask for a specific method, use the simple/manual one
        if energy_method_spec is None:
            # Pass e_train and weight_bits so defaults match your previous numbers
            kwargs.setdefault("e_train", e_train)
            kwargs.setdefault("weight_bits", weight_bits)
            energy_method = load_energy_method("simple", **kwargs)
        else:
            energy_method = load_energy_method(energy_method_spec, **kwargs)

    events: List[EnergyEvent] = []
    ch = sim.ch_index
    all_members = list(range(sim.num_agents))
    learners: List[int] = []

    # -------- Bootstrap (pre-round collection) --------
    bootstrap_queue = all_members[:sim.bootstrap_collectors]
    event_in_round = 0
    for departing in bootstrap_queue:
        data_bits = bits_for_trajectory(
            steps=sim.traj_steps,
            obs=sim.obs_size,
            bits_per_elem=sim.bits_per_element,
            eta=sim.mac_efficiency,
        )
        m_tx = energy_method.upload_data_member_tx(data_bits, link)
        ch_rx = energy_method.upload_data_ch_rx(data_bits, link)

        if not sim.exclude_departed_agent_tx and sim.log_events:
            _log(events, round_idx=-1, event_in_round=event_in_round,
                 role="Learner", agent_id=departing,
                 phase="collect", kind="comm", direction="tx",
                 bits=data_bits, joules=m_tx,
                 note="Trajectory upload to CH")
            event_in_round += 1

        if sim.log_events:
            _log(events, round_idx=-1, event_in_round=event_in_round,
                 role="CH", agent_id=ch,
                 phase="collect", kind="comm", direction="rx",
                 bits=data_bits, joules=ch_rx,
                 note=f"Receive trajectory from learner {departing}")
            event_in_round += 1

    # After bootstrap, pick initial learners
    learners = all_members[1:1 + sim.max_learners_per_round]

    # -------- Federated Rounds --------
    for rnd in range(sim.num_rounds):
        k = len(learners)
        if k == 0:
            break

        event_in_round = 0  # reset per round

        # === CH -> Learners: distribute training batches (50 updates) ===
        batch_bits_total = bits_for_batch_updates(
            batch=sim.batch_size,
            obs=sim.obs_size,
            bits_per_elem=sim.bits_per_element,
            eta=sim.mac_efficiency,
            num_updates=sim.num_batch_updates,
        )
        ch_tx_batches = energy_method.ch_batch_tx(
            batch_bits_total, link, k_members=k, broadcast=sim.broadcast_training_batch
        )
        learner_rx_batches = energy_method.member_batch_rx(batch_bits_total, link)

        if sim.log_events:
            _log(events,
                 round_idx=rnd, event_in_round=event_in_round,
                 role="CH", agent_id=ch,
                 phase="distribute", kind="comm", direction="tx",
                 bits=batch_bits_total if sim.broadcast_training_batch else batch_bits_total * k,
                 joules=ch_tx_batches,
                 note=f"Distribute {sim.num_batch_updates} training batches")
        event_in_round += 1

        for i in learners:
            if sim.log_events:
                _log(events,
                     round_idx=rnd, event_in_round=event_in_round,
                     role="Learner", agent_id=i,
                     phase="distribute", kind="comm", direction="rx",
                     bits=batch_bits_total, joules=learner_rx_batches,
                     note=f"Receive {sim.num_batch_updates} training batches")
            event_in_round += 1
        # === Local Training (learners) ===
        e_local = energy_method.compute_train()
        for i in learners:
            if sim.log_events:
                _log(events,
                     round_idx=rnd, event_in_round=event_in_round,
                     role="Learner", agent_id=i,
                     phase="train", kind="compute", direction="",
                     bits=0, joules=e_local, note="Local training")
            event_in_round += 1

        # === Upload local weights to CH ===
        member_tx_J = energy_method.member_weights_tx(weight_bits, link)
        ch_rx_J = energy_method.ch_weights_rx(weight_bits, link, k_members=k)

        for i in learners:
            if not sim.exclude_departed_agent_tx and sim.log_events:
                _log(events,
                     round_idx=rnd, event_in_round=event_in_round,
                     role="Learner", agent_id=i,
                     phase="agg", kind="comm", direction="tx",
                     bits=weight_bits, joules=member_tx_J,
                     note="Upload local weights to CH")
            event_in_round += 1

        if sim.log_events:
            _log(events,
                 round_idx=rnd, event_in_round=event_in_round,
                 role="CH", agent_id=ch,
                 phase="agg", kind="comm", direction="rx",
                 bits=weight_bits * k, joules=ch_rx_J,
                 note=f"Receive {k} local weights from learners")
        event_in_round += 1

        # === CH local training (optional) ===
        if sim.ch_trains_too and sim.log_events:
            e_ch = energy_method.compute_train()
            _log(events,
                 round_idx=rnd, event_in_round=event_in_round,
                 role="CH", agent_id=ch,
                 phase="train", kind="compute", direction="",
                 bits=0, joules=e_ch, note="CH local training")
            event_in_round += 1

        # === CH ↔ Server aggregation (1 up, 1 down) ===
        if sim.model_server_step:
            ch_to_srv = energy_method.ch_to_server_tx(weight_bits, link)
            srv_to_ch = energy_method.server_to_ch_rx(weight_bits, link)

            if sim.log_events:
                _log(events,
                     round_idx=rnd, event_in_round=event_in_round,
                     role="CH", agent_id=ch,
                     phase="agg", kind="comm", direction="tx",
                     bits=weight_bits, joules=ch_to_srv,
                     note="Upload averaged weights to Server")
            event_in_round += 1

            if sim.log_events:
                _log(events,
                     round_idx=rnd, event_in_round=event_in_round,
                     role="CH", agent_id=ch,
                     phase="agg", kind="comm", direction="rx",
                     bits=weight_bits, joules=srv_to_ch,
                     note="Download global weights from Server")
            event_in_round += 1

        # === CH → Learners: distribute global weights ===
        ch_tx_global = energy_method.ch_batch_tx(
            weight_bits, link, k_members=k, broadcast=sim.use_broadcast_downlink
        )
        learner_rx_global = energy_method.member_batch_rx(weight_bits, link)

        if sim.log_events:
            _log(events,
                 round_idx=rnd, event_in_round=event_in_round,
                 role="CH", agent_id=ch,
                 phase="distribute", kind="comm", direction="tx",
                 bits=weight_bits if sim.use_broadcast_downlink else weight_bits * k,
                 joules=ch_tx_global,
                 note="Downlink global weights to learners")
        event_in_round += 1

        for i in learners:
            if sim.log_events:
                _log(events,
                     round_idx=rnd, event_in_round=event_in_round,
                     role="Learner", agent_id=i,
                     phase="distribute", kind="comm", direction="rx",
                     bits=weight_bits, joules=learner_rx_global,
                     note="Receive global weights from CH")
            event_in_round += 1

    # Return only per-round events (no cluster totals)
    return {
        "events": [asdict(e) for e in events],
        "ch_index": ch,
        "num_agents": sim.num_agents,
    }

# ====================================================
# Optional: per-round summary helper (unchanged)
# ====================================================

# def summarize_energy_per_round(events: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
#     """
#     Returns: { round_idx: { 'CH_compute': x, 'CH_comm': y, 'Learners_compute': a, 'Learners_comm': b, 'Total': t } }
#     """
#     out: Dict[int, Dict[str, float]] = {}
#     for ev in events:
#         r = ev["round_idx"]
#         if r not in out:
#             out[r] = {
#                 "CH_compute": 0.0, "CH_comm": 0.0,
#                 "Learners_compute": 0.0, "Learners_comm": 0.0,
#                 "Total": 0.0
#             }
#         key_role = "CH" if ev["role"] == "CH" else "Learners"
#         key_kind = "compute" if ev["kind"] == "compute" else "comm"
#         out[r][f"{key_role}_{key_kind}"] += ev["joules"]
#         out[r]["Total"] += ev["joules"]
#         # out[r]["Learners_compute"] /= 16
#         # out[r]["Learners_comm"] /= 16
#
#     return out
def summarize_energy_per_round(
    events: List[Dict[str, Any]],
    track_id: Optional[int] = None,
    return_per_member: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Returns a dict keyed by round_idx. Each value includes:
      - CH_compute, CH_comm
      - Learners_compute, Learners_comm (totals)
      - Learner_avg_compute, Learner_avg_comm (average per learner in that round)
      - Learner_track_compute, Learner_track_comm (for a specific agent_id if provided and present)
      - (optional) Learners_detail: {agent_id: {'compute': x, 'comm': y}}
    """
    out: Dict[int, Dict[str, Any]] = {}
    # per-round, per-learner accumulator
    per_round_member: Dict[int, Dict[int, Dict[str, float]]] = {}

    for ev in events:
        r = ev["round_idx"]
        out.setdefault(r, {
            "CH_compute": 0.0, "CH_comm": 0.0,
            "Learners_compute": 0.0, "Learners_comm": 0.0,
            "Total": 0.0
        })
        per_round_member.setdefault(r, {})

        role = ev["role"]
        kind = ev["kind"]            # "compute" or "comm"
        j = float(ev["joules"])
        agent_id = ev["agent_id"]

        if role == "CH":
            if kind == "compute":
                out[r]["CH_compute"] += j
            else:
                out[r]["CH_comm"] += j
        else:  # Learner
            if kind == "compute":
                out[r]["Learners_compute"] += j
            else:
                out[r]["Learners_comm"] += j

            # keep per-learner breakdown
            per_round_member[r].setdefault(agent_id, {"compute": 0.0, "comm": 0.0})
            per_round_member[r][agent_id][kind] += j

        out[r]["Total"] += j

    # post-process: averages and tracked learner
    for r, summary in out.items():
        learners_map = per_round_member.get(r, {})
        k = len(learners_map)

        if k > 0:
            summary["Learner_avg_compute"] = summary["Learners_compute"] / k
            summary["Learner_avg_comm"] = summary["Learners_comm"] / k
        else:
            summary["Learner_avg_compute"] = 0.0
            summary["Learner_avg_comm"] = 0.0

        if track_id is not None and track_id in learners_map:
            summary["Learner_track_compute"] = learners_map[track_id]["compute"]
            summary["Learner_track_comm"] = learners_map[track_id]["comm"]
        else:
            summary["Learner_track_compute"] = None
            summary["Learner_track_comm"] = None

        if return_per_member:
            summary["Learners_detail"] = learners_map

    return out


# ====================================================
# Standalone demo
# ====================================================

if __name__ == "__main__":

    energy_method_spec = "energy_log_recorder:ParametricEnergy"  # class path

    # ---- Derive compute from AGX Orin (no manual e_train) ----
    # Param count from your fp32 model bits:
    model_params = 4_440_128 // 32  # = 138,754

    # One training step FLOPs:
    #   flops_per_step ≈ coef * (#params) * (batch_size)
    # Rule of thumb:
    #   forward ≈ 2P, backward ≈ 4P  => coef ≈ 6 per sample (SGD-like).
    # If you use Adam/LayerNorm/etc., set coef higher (e.g., 10–30).
    coef_train_per_param_per_sample = 12.0  # tweak 6–30 to calibrate vs measurements

    flops_per_step = coef_train_per_param_per_sample * model_params * 64  # batch_size=64
    steps_per_round = 50  # 50 policy updates per round

    energy_method_kwargs = dict(
        # === Compute from FLOPs on AGX Orin ===
        device_profile=AGX_ORIN,  # uses Orin’s TFLOPS, power, safety scale
        flops_per_step=flops_per_step,
        steps_per_round=steps_per_round,

        # === Aggregation compute at CH (enable non-zero CH_compute for aggregation) ===
        model_num_params=model_params,
        agg_flops_per_param=1.0,  # 1–3 typical; raise to make aggregation more visible
        server_update_flops=0.0,

        # === Radio: 802.11n-ish @ 30 Mbps with η=0.85 (our agreed defaults) ===
        # e_tx_per_bit=1.28/30e6, e_rx_per_bit=0.94/30e6,  # defaults in class
        mac_efficiency=0.85,
        scale_broadcast=1.0,
    )


    class DummyTData:  # kept only for compatibility; unused now
        records_per_member_per_round = 5
        compression_ratio = 0.8


    link = LinkModel()
    cfg = SimConfig(
        obs_size=12,
        batch_size=64,
        num_batch_updates=50,
        traj_steps=11,
        mac_efficiency=0.85,
        broadcast_training_batch=True,
        use_broadcast_downlink=True,
        # If the CH should NOT do local training, set this False:
        # ch_trains_too=False,
    )

    result = simulate_and_report(
        cfg, link, DummyTData(),
        energy_method_spec=energy_method_spec,
        energy_method_kwargs=energy_method_kwargs,
        weight_bits=4_440_128,
    )

    print("First 8 events:")
    for ev in result["events"][:8]:
        print(ev)

    summaries = summarize_energy_per_round(result["events"])
    print("\nPer-round energy summary (J):")
    for r in sorted(summaries.keys()):
        label = "bootstrap" if r == -1 else f"round {r}"
        print(label, summaries[r])
