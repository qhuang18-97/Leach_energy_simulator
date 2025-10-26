# energy_models.py
def energy_compute(power_w: float, seconds: float) -> float:
    """Joules = Watts * Seconds."""
    return max(0.0, power_w * seconds)

def energy_tx(bits: int, j_per_bit_tx: float) -> float:
    return max(0.0, bits * j_per_bit_tx)

def energy_rx(bits: int, j_per_bit_rx: float) -> float:
    return max(0.0, bits * j_per_bit_rx)
