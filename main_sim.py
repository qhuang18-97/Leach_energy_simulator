# main_sim.py
from leach_cluster_sim import AgentState, LeachController, Cluster
from round_driver import mk_legacy_energy_and_link, round_driver, apply_energy_from_events
import config as C
from energy_log_recorder import AGX_ORIN
# --- Values mirrored from /mnt/data/energy_log_recorder.py demo & defaults ---

# Model payload used previously
WEIGHT_BITS = 4_440_128

# Param count derived exactly the same way as in the demo:
MODEL_NUM_PARAMS = WEIGHT_BITS // 32                 # = 138_754

# Same training FLOPs recipe as your demo:
#   flops_per_step = coef * params * batch_size
COEF_TRAIN_PER_PARAM_PER_SAMPLE = 12.0               # (you tweaked 6–30 there)
BATCH_SIZE = 64                                       # SimConfig.batch_size default
FLOPS_PER_STEP = COEF_TRAIN_PER_PARAM_PER_SAMPLE * MODEL_NUM_PARAMS * BATCH_SIZE
STEPS_PER_ROUND = 50                                  # SimConfig.num_batch_updates default

# Radio defaults used when per-bit energy is not provided
# (these are LinkModel defaults in your file)
P_TX_W, P_RX_W = 2.0, 1.0
R_UP, R_DN = 1e6, 1e6

# MAC/PHY efficiency & broadcast scale exactly as in your file/demo
MAC_EFF = 0.85
SCALE_BCAST = 1.0

# Build energy + link using your ParametricEnergy/LinkModel API
energy, link = mk_legacy_energy_and_link(
    device_profile=AGX_ORIN,               # leave None to use ParametricEnergy default (AGX_ORIN)
    flops_per_step=FLOPS_PER_STEP,
    steps_per_round=STEPS_PER_ROUND,
    model_num_params=MODEL_NUM_PARAMS,
    agg_flops_per_param=1.0,              # same as demo
    server_update_flops=0.0,              # same as demo
    e_tx_per_bit=None,                    # use power/rate model (same as your defaults)
    e_rx_per_bit=None,
    e_tx_per_bit_srv=None,
    e_rx_per_bit_srv=None,
    ctrl_tx_bits=0,
    ctrl_rx_bits=0,
    mac_efficiency=MAC_EFF,
    scale_broadcast=SCALE_BCAST,
    p_tx_w=P_TX_W, p_rx_w=P_RX_W, r_up=R_UP, r_dn=R_DN,
)

def build_agents(n: int, init_j: float):
    agents = {i: AgentState(agent_id=i, residual_j=init_j) for i in range(n)}
    # If you flip USE_LQ=True in config, you can also populate lq_to_all here.
    return agents

if __name__ == "__main__":
    agents = build_agents(C.N_AGENTS, C.INIT_BATTERY_J)

    leach = LeachController(
        p_base=C.P_BASE,
        alpha=1.0,
        use_lq=C.USE_LQ,
    )

    cluster = Cluster(
        agents=agents,
        ch_id=None,
        leach=leach,
        max_members=C.MAX_MEMBERS,
        e_floor=C.E_FLOOR,
        e_abort=C.E_ABORT,
        learner_policy=C.LEARNER_POLICY,
        collector_policy=C.COLLECTOR_POLICY,
        # Curry the driver so it uses the SAME energy/link math as before
        run_round_fn=lambda ch, members, learners, collector, r: round_driver(
            ch, members, learners, collector, r,
            energy=energy, link=link,
            weight_bits=WEIGHT_BITS,
            batch_bits=None,            # set if you previously used batch pushes
            traj_bits=None,             # set if you previously uploaded trajectories
            broadcast_weights=True      # matches your prior behavior
        ),
        apply_energy_fn=apply_energy_from_events,
    )

    all_events = []
    for r in range(C.N_ROUNDS):
        ev = cluster.step(round_idx=r, k_learners=C.K_LEARNERS_PER_ROUND)
        all_events.extend(ev)
        rec = cluster.round_log.last()
        print(f"round {rec['round_idx']} | CH {rec['ch_id']} | #learners {len(rec['learners'])}")
        print("  totals:", rec["totals"])
        for aid, a in rec["agents"].items():
            role = "CH" if aid == rec["ch_id"] else "Learner"
            print(f"   {role} {aid}: compute {a['compute_J']:.6f} J, comm {a['comm_J']:.6f} J, residual {a['residual_J_after']:.2f} J")
        if rec["warnings"]:
            print("  WARN:", rec["warnings"])
