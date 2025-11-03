# main_sim.py
from leach_cluster_sim import AgentState, LeachController, Cluster
from round_driver import mk_legacy_energy_and_link, round_driver, apply_energy_from_events
from energy_log_recorder import AGX_ORIN       # device profile for ParametricEnergy
import config as C


def build_agents(n: int, init_j: float):
    """Create agent states."""
    agents = {i: AgentState(agent_id=i, residual_j=init_j) for i in range(n)}
    return agents


if __name__ == "__main__":
    # --------------------------------------------------------------
    # 1.  Build agents and LEACH controller
    # --------------------------------------------------------------
    agents = build_agents(C.N_AGENTS, C.INIT_BATTERY_J)
    leach = LeachController(p_base=C.P_BASE, alpha=1.0, use_lq=C.USE_LQ)

    # --------------------------------------------------------------
    # 2.  Build energy + link models (use mk_legacy_energy_and_link)
    # --------------------------------------------------------------

    # ---- Model configuration (mirrors energy_log_recorder defaults) ----
    WEIGHT_BITS = 4_440_128                         # same as before
    MODEL_NUM_PARAMS = WEIGHT_BITS // 32            # 138,754 parameters
    COEF_TRAIN_PER_PARAM_PER_SAMPLE = 12.0
    BATCH_SIZE = 64
    FLOPS_PER_STEP = COEF_TRAIN_PER_PARAM_PER_SAMPLE * MODEL_NUM_PARAMS * BATCH_SIZE
    STEPS_PER_ROUND = 50

    # ---- Radio & per-bit settings ----
    # Option A: use per-bit energy constants (recommended small-network config)
    energy, link = mk_legacy_energy_and_link(
        device_profile=AGX_ORIN,
        flops_per_step=FLOPS_PER_STEP,
        steps_per_round=STEPS_PER_ROUND,
        model_num_params=MODEL_NUM_PARAMS,
        agg_flops_per_param=1.0,
        server_update_flops=0.0,
        # Use your calibrated per-bit values
        e_tx_per_bit=C.J_PER_BIT_TX,
        e_rx_per_bit=C.J_PER_BIT_RX,
        e_tx_per_bit_srv=C.J_PER_BIT_TX,
        e_rx_per_bit_srv=C.J_PER_BIT_RX,
        # Remaining radio parameters (still required, but ignored when per-bit is given)
        p_tx_w=2.0, p_rx_w=1.0, r_up=1e6, r_dn=1e6,
        mac_efficiency=0.85, scale_broadcast=1.0,
    )

    # Option B (alternative): if you prefer power/rate model, comment the e_* lines above
    # and set realistic Wi-Fi speeds: r_up=r_dn=30e6 (30 Mbps)

    # --------------------------------------------------------------
    # 3.  Build cluster with a run_round_fn using this energy/link
    # --------------------------------------------------------------
    cluster = Cluster(
        agents=agents,
        ch_id=None,
        leach=leach,
        max_members=C.MAX_MEMBERS,
        e_floor=C.E_FLOOR,
        e_abort=C.E_ABORT,
        learner_policy=C.LEARNER_POLICY,
        collector_policy=C.COLLECTOR_POLICY,
        # Curry the driver so it always uses our energy + link models
        run_round_fn=lambda ch, members, learners, collector, r: round_driver(
            ch, members, learners, collector, r,
            energy=energy,
            link=link,
            weight_bits=WEIGHT_BITS,
            batch_bits=None,
            traj_bits=None,
            broadcast_weights=True,
        ),
        apply_energy_fn=apply_energy_from_events,
    )

    # --------------------------------------------------------------
    # 4.  Simulation loop
    # --------------------------------------------------------------
    all_events = []
    for r in range(C.N_ROUNDS):
        ev = cluster.step(round_idx=r, k_learners=C.K_LEARNERS_PER_ROUND)
        all_events.extend(ev)

        rec = cluster.round_log.last()
        print(f"round {rec['round_idx']} | CH {rec['ch_id']} | #learners {len(rec['learners'])}")
        print("  totals:", rec["totals"])
        for aid, a in rec["agents"].items():
            role = "CH" if aid == rec["ch_id"] else "Learner"
            print(f"   {role} {aid}: compute {a['compute_J']:.6f} J, "
                  f"comm {a['comm_J']:.6f} J, residual {a['residual_J_after']:.2f} J")
        if rec["warnings"]:
            print("  WARN:", rec["warnings"])

    print("\nSimulation complete.")
