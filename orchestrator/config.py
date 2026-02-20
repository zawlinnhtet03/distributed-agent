import os
import time

import ray


def _get_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value}") from exc


def init_ray():
    if ray.is_initialized():
        return

    address = os.getenv("RAY_ADDRESS")
    init_args = {"ignore_reinit_error": True}

    num_cpus = _get_optional_int("RAY_NUM_CPUS")
    num_gpus = _get_optional_int("RAY_NUM_GPUS")

    if num_cpus is not None:
        init_args["num_cpus"] = num_cpus
    if num_gpus is not None:
        init_args["num_gpus"] = num_gpus

    if address:
        last_error = None
        for attempt in range(1, 11):
            try:
                ray.init(address=address, **init_args)
                print(f"Ray connected to cluster at {address}")
                return
            except Exception as exc:  # pragma: no cover - depends on runtime cluster state
                last_error = exc
                time.sleep(2)
        raise RuntimeError(f"Could not connect to Ray at {address}") from last_error
    else:
        ray.init(**init_args)
        print("Ray initialized in local mode")
