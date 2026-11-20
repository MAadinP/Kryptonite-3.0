"""Logging how long experiments take (for the sustainability section) and saving results."""
import json
import os
import platform
import time
from contextlib import contextmanager

import numpy as np
import sklearn
import torch

from .config import RESULTS

CPU_POWER_W = 45.0      # rough laptop power draw with every core busy (W)
CORES = 12              # logical cores of the laptop most experiments ran on
# Fixed, not os.cpu_count(): the estimate must not depend on which machine redraws the figure
CORE_POWER_W = CPU_POWER_W / CORES  # 3.75 W per busy logical core
GRID_INTENSITY = 0.124  # kg CO2e per kWh, GB grid average


def energy_kwh(cpu_s):
    # CPU time x the power of one busy logical core. We use CPU time, not wall time, because time spent
    # asleep or sharing the machine with another run adds to wall time but not to CPU time.
    return cpu_s * CORE_POWER_W / 3.6e6


@contextmanager
def track(name):
    # time a block of code and add its runtime, energy and CO2 to results/runtime_log.json
    t0, c0 = time.perf_counter(), time.process_time()
    yield
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    kwh = energy_kwh(cpu)

    RESULTS.mkdir(exist_ok=True)
    log_file = RESULTS / "runtime_log.json"
    log = json.loads(log_file.read_text()) if log_file.exists() else {}
    log[name] = {"wall_s": round(wall, 1), "cpu_s": round(cpu, 1), "energy_kWh": kwh,
                 "kgCO2e": kwh * GRID_INTENSITY, "threads": torch.get_num_threads(),
                 "machine": platform.processor() or platform.machine(),
                 # bit-identical reruns also need the same library versions
                 "versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "sklearn": sklearn.__version__}}
    log_file.write_text(json.dumps(log, indent=2))
    print(f"[{name}] {wall:.0f}s, ~{kwh * 1000:.2f} Wh, ~{kwh * GRID_INTENSITY * 1000:.2f} gCO2e")


def save_json(obj, name):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2))


def load_json(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None
