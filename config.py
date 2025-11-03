# config.py
N_AGENTS = 700
INIT_BATTERY_J = 10_000.0

# LEACH / cluster knobs
P_BASE = 1.0 / N_AGENTS
USE_LQ = False
MAX_MEMBERS = N_AGENTS - 1
E_FLOOR = 0.10 * INIT_BATTERY_J
E_ABORT = 0.05 * INIT_BATTERY_J
LEARNER_POLICY = "battery_lowest"   # battery_lowest | battery_highest | random | lq_best
COLLECTOR_POLICY = "battery_lowest"
K_LEARNERS_PER_ROUND = 16
N_ROUNDS = 700

# Example payload sizes (change to your real values)
BITS_UPLINK_PER_LEARNER = 8_000_000   # 1 MB
BITS_DOWNLINK_MODEL     = 8_000_000   # 1 MB

# Example compute placeholders (replace with your numbers)
LEARNER_COMPUTE_SEC = 0.02
LEARNER_POWER_W     = 30.0
CH_AGG_SEC          = 0.01
CH_POWER_W          = 35.0

# Radio energy model (example; replace with your calibrated figures)
J_PER_BIT_TX = 2.5e-8   # 25 nJ / bit
J_PER_BIT_RX = 1.0e-8   # 10 nJ / bit


# Size of one "new experience" push from the collector to CH (bits)
# Set this to your actual per-round experience payload. Example: 2 MB
EXP_BITS_PER_UPDATE = 2_000_000 * 8   # 2 MB → 16,000,000 bits (adjust!)

# Whether to include the new experience pipeline this round
ENABLE_EXPERIENCE_PIPELINE = True

# Whether learners should explicitly receive the global model after aggregation
ENABLE_WEIGHT_BROADCAST_RX = True  # we’ll now count learner RX for weights

