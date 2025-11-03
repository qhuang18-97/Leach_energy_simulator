# round_driver.py (legacy-compatible version)
from typing import List, Dict, Any, Optional
from energy_log_recorder import load_energy_method, LinkModel
from energy_log_recorder import ParametricEnergy, AGX_ORIN, GTX_1080  # if you want explicit profiles
# If you already had a dict of kwargs for ParametricEnergy, reuse it here.

def mk_legacy_energy_and_link(
    *,
    # Fill these with the same values you used before (placeholders shown)
    device_profile=AGX_ORIN,
    flops_per_step: float = 1.2e11,   # example FLOPs per learner step (forward+backward+opt)
    steps_per_round: int = 1,         # local steps per round per learner
    model_num_params: int = 1_110_032, # to compute CH aggregation FLOPs
    agg_flops_per_param: float = 2.0,  # CH agg flops per param per member
    server_update_flops: float = 0.0,  # if you modeled server compute
    e_tx_per_bit: Optional[float] = None,  # if you used per-bit energy (else time*power)
    e_rx_per_bit: Optional[float] = None,
    e_tx_per_bit_srv: Optional[float] = None,
    e_rx_per_bit_srv: Optional[float] = None,
    ctrl_tx_bits: int = 0,
    ctrl_rx_bits: int = 0,
    mac_efficiency: float = 0.85,
    scale_broadcast: float = 1.0,
    # Link (radio) you used before:
    p_tx_w: float = 2.0, p_rx_w: float = 1.0, r_up: float = 1e6, r_dn: float = 1e6,
):
    energy = ParametricEnergy(
        device_profile=device_profile,
        flops_per_step=flops_per_step,
        steps_per_round=steps_per_round,
        model_num_params=model_num_params,
        agg_flops_per_param=agg_flops_per_param,
        server_update_flops=server_update_flops,
        e_tx_per_bit=e_tx_per_bit,
        e_rx_per_bit=e_rx_per_bit,
        e_tx_per_bit_srv=e_tx_per_bit_srv,
        e_rx_per_bit_srv=e_rx_per_bit_srv,
        ctrl_tx_bits=ctrl_tx_bits,
        ctrl_rx_bits=ctrl_rx_bits,
        mac_efficiency=mac_efficiency,
        scale_broadcast=scale_broadcast,
    )
    link = LinkModel(p_tx_w=p_tx_w, p_rx_w=p_rx_w, r_up=r_up, r_dn=r_dn)
    return energy, link

# def round_driver(
#     ch_id: int,
#     members: List[int],
#     learners: List[int],
#     collector: int,
#     round_idx: int,
#     *,
#     energy: ParametricEnergy,
#     link: LinkModel,
#     # Use the exact payload sizes you used before. 4_440_128 was your previous default.
#     weight_bits: int = 4_440_128,
#     # If you also modeled batch exchanges and/or trajectory uploads, include their bits:
#     batch_bits: Optional[int] = None,
#     traj_bits: Optional[int] = None,
#     broadcast_weights: bool = True,
# ) -> List[Dict[str, Any]]:
#     """
#     Returns events with joules computed by your legacy ParametricEnergy:
#       - Learner local compute
#       - Learner -> CH weights upload
#       - CH RX + aggregation compute
#       - CH <-> Server (optional)
#       - CH -> learners broadcast (optionally counted once, like before)
#     """
#     events: List[Dict[str, Any]] = []
#     k = len(learners)
#
#     # 1) Learners local training compute
#     for lid in learners:
#         j = energy.compute_train()
#         events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "compute", "joules": j})
#
#     # 2) (Optional) batch/trajectory traffic if you modeled it previously
#     if traj_bits:
#         for lid in learners:
#             j_tx = energy.upload_data_member_tx(traj_bits, link)
#             events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "comm", "joules": j_tx})
#         j_rx = energy.upload_data_ch_rx(traj_bits * k, link)
#         events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_rx})
#
#     if batch_bits:
#         # learners receive batches; CH transmits
#         j_ch_tx = energy.ch_batch_tx(batch_bits, link, k_members=k, broadcast=True)
#         events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_ch_tx})
#         for lid in learners:
#             j_rx = energy.member_batch_rx(batch_bits, link)
#             events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "comm", "joules": j_rx})
#
#     # 3) Learners -> CH weights upload
#     for lid in learners:
#         j_tx = energy.member_weights_tx(weight_bits, link)
#         events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "comm", "joules": j_tx})
#
#     # 4) CH RX + aggregation compute (and optional server compute inside)
#     j_ch_rx_agg = energy.ch_weights_rx(weight_bits, link, k_members=k)
#     events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_ch_rx_agg})
#
#     # 5) CH <-> Server comm (if you modeled it)
#     #    Pass bits=None if your previous model treated it as pure control/keepalive.
#     j_up = energy.ch_to_server_tx(weight_bits, link)
#     events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_up})
#     j_dn = energy.server_to_ch_rx(weight_bits, link)
#     events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_dn})
#
#     # 6) CH broadcast of new model to learners (if previously counted)
#     if broadcast_weights:
#         j_bcast = energy.ch_batch_tx(weight_bits, link, k_members=k, broadcast=True)
#         events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_bcast})
#         # If you also charged learners for RX, uncomment:
#         # for lid in learners:
#         #     j_rx = energy.member_batch_rx(weight_bits, link)
#         #     events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "comm", "joules": j_rx})
#
#     return events
# round_driver.py
from typing import List, Dict, Any
import config as C
from energy_log_recorder import ParametricEnergy, LinkModel

def round_driver(
    ch_id: int,
    members: List[int],
    learners: List[int],
    collector: int,
    round_idx: int,
    *,
    energy: ParametricEnergy,
    link: LinkModel,
    weight_bits: int,
    batch_bits: int | None = None,
    traj_bits: int | None = None,          # kept for backward compat; not used if EXP pipeline on
    broadcast_weights: bool = True,
) -> List[Dict[str, Any]]:
    """
    Per-round comm/compute events with the added 'experience' pipeline:

    Pipeline order:
      0) (optional) Collector -> CH : upload NEW EXPERIENCE (EXP_BITS_PER_UPDATE)
         CH -> Learners : broadcast NEW EXPERIENCE ; Learners RX new data
      1) Learners compute locally
      2) Learners -> CH : upload weights (member_weights_tx)
         CH RX all weights (ch_weights_rx)
      3) CH <-> Server : (optional) push/pull global model
      4) CH -> Learners : broadcast UPDATED WEIGHTS ; Learners RX updated weights
    """
    events: List[Dict[str, Any]] = []

    # ---------------- 0) Experience ingress & distribution ----------------
    if C.ENABLE_EXPERIENCE_PIPELINE:
        exp_bits = C.EXP_BITS_PER_UPDATE

        # Collector uploads new experience to CH
        # TX at collector
        j_tx_col = energy.upload_data_member_tx(exp_bits, link)
        events.append({"round_idx": round_idx, "agent_id": collector, "role": "Learner", "kind": "comm", "joules": j_tx_col})

        # RX at CH
        j_rx_ch = energy.upload_data_ch_rx(exp_bits, link)
        events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_rx_ch})

        # CH broadcasts the new experience to learners (one-to-many broadcast counted once at CH)
        j_bcast_exp = energy.ch_batch_tx(exp_bits, link, k_members=len(learners), broadcast=True)
        events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_bcast_exp})

        # Each learner receives the new experience (count learner RX individually)
        for lid in learners:
            j_rx_l = energy.member_batch_rx(exp_bits, link)
            events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "comm", "joules": j_rx_l})

    # ---------------- 1) Learners local compute ----------------
    for lid in learners:
        j_compute = energy.compute_train()
        events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "compute", "joules": j_compute})

    # ---------------- 2) Learners upload weights ; CH RX+aggregate ----------------
    for lid in learners:
        j_tx = energy.member_weights_tx(weight_bits, link)
        events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "comm", "joules": j_tx})

    # CH receives all learners' weights (RX + aggregation compute inside your energy class)
    j_ch_rx_agg = energy.ch_weights_rx(weight_bits, link, k_members=len(learners))
    events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_ch_rx_agg})

    # ---------------- 3) CH <-> Server exchange (if modeled) ----------------
    j_up = energy.ch_to_server_tx(weight_bits, link)
    events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_up})
    j_dn = energy.server_to_ch_rx(weight_bits, link)
    events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_dn})

    # ---------------- 4) CH broadcasts UPDATED weights ; learners RX ----------------
    if broadcast_weights:
        # CH one-shot broadcast of new global model
        j_bcast_w = energy.ch_batch_tx(weight_bits, link, k_members=len(learners), broadcast=True)
        events.append({"round_idx": round_idx, "agent_id": ch_id, "role": "CH", "kind": "comm", "joules": j_bcast_w})

        if C.ENABLE_WEIGHT_BROADCAST_RX:
            # Each learner receives the updated weights
            for lid in learners:
                j_rx = energy.member_batch_rx(weight_bits, link)
                events.append({"round_idx": round_idx, "agent_id": lid, "role": "Learner", "kind": "comm", "joules": j_rx})

    return events


def apply_energy_from_events(events: List[Dict[str, Any]]) -> Dict[int, float]:
    consumed = {}
    for ev in events:
        i = int(ev["agent_id"])
        consumed[i] = consumed.get(i, 0.0) + float(ev.get("joules", 0.0))
    return consumed
