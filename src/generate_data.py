"""
================================================
 Predictive Maintenance - Data Generation
 Generates a realistic synthetic sensor dataset
================================================
"""

import numpy as np
import pandas as pd
import os

def generate_dataset(n_samples=5000, random_state=42, save_path="data/sensor_data.csv"):
    """
    Generate a synthetic industrial sensor dataset.
    
    Features:
        - temperature   : Machine temperature in °C
        - vibration     : Vibration level in mm/s
        - pressure      : Pressure in bar
        - runtime_hours : Cumulative machine runtime in hours
        - failure       : Target variable (1 = Failure, 0 = No Failure)
    """
    np.random.seed(random_state)

    # ── Normal operating conditions ─────────────────────────────────────────
    temperature   = np.random.normal(loc=75,  scale=10,  size=n_samples)   # °C
    vibration     = np.random.normal(loc=0.5, scale=0.1, size=n_samples)   # mm/s
    pressure      = np.random.normal(loc=30,  scale=5,   size=n_samples)   # bar
    runtime_hours = np.random.uniform(0, 5000, size=n_samples)             # hours

    # ── Inject realistic failure conditions (≈10 % of samples) ──────────────
    failure_idx = np.random.choice(n_samples, size=int(0.10 * n_samples), replace=False)

    temperature[failure_idx]   += np.random.normal(20, 5,  size=len(failure_idx))
    vibration[failure_idx]     += np.random.normal(0.4, 0.1, size=len(failure_idx))
    pressure[failure_idx]      += np.random.normal(10, 3,  size=len(failure_idx))
    runtime_hours[failure_idx] *= np.random.uniform(1.2, 1.5, size=len(failure_idx))

    failure = np.zeros(n_samples, dtype=int)
    failure[failure_idx] = 1

    # ── Introduce ≈2 % missing values to simulate real-world noise ──────────
    for col_arr in [temperature, vibration, pressure, runtime_hours]:
        missing_idx = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        col_arr[missing_idx] = np.nan

    df = pd.DataFrame({
        "temperature":   temperature,
        "vibration":     vibration,
        "pressure":      pressure,
        "runtime_hours": runtime_hours,
        "failure":       failure
    })

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[✓] Dataset saved → {save_path}  |  Shape: {df.shape}")
    print(f"    Failure rate: {df['failure'].mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    generate_dataset(save_path="data/sensor_data.csv")