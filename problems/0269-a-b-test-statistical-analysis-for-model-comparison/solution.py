import numpy as np
import math

def analyze_ab_test(control_outcomes: list, treatment_outcomes: list, confidence_level: float = 0.95, min_detectable_effect: float = 0.02) -> dict:
    nc, nt = len(control_outcomes), len(treatment_outcomes)
    if not nc or not nt:
        return {}

    control_rate = np.mean(control_outcomes)
    treatment_rate = np.mean(treatment_outcomes)
    alpha = 1 - confidence_level
    
    # 1. Lift calculations
    absolute_lift = treatment_rate - control_rate
    relative_lift_pct = (absolute_lift / control_rate * 100) if control_rate != 0 else 0.0
    
    # 2. Z-Score calculation (Pooled Variance)
    pool_rate = (nc * control_rate + nt * treatment_rate) / (nc + nt)
    standard_error = math.sqrt(pool_rate * (1 - pool_rate) * (1/nc + 1/nt))
    z_statistic = absolute_lift / standard_error if standard_error != 0 else 0.0
    
    # 3. P-Value calculation (using math.erf instead of scipy)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    def norm_cdf(z):
    