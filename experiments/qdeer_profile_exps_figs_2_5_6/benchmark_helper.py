"""
Helper functions for memory and time profiling
"""

from typing import Tuple, Callable, Sequence, List
import jax

import time
import traceback

# for memory profiling
import GPUtil
import gc
import threading


def many_function_benchmark(
    func_dict,
    args: Sequence,
    with_jit: bool = True,
    nwarmups: int = 5,
    nreps: int = 5,
):
    """
    Helper function to report the timing of multiple functions
    that all takes args

    Args:
        func_dict: dictionary of name and function
    """
    results = {}

    if with_jit:
        for name, func in func_dict.items():
            func_dict[name] = jax.jit(func)

    for key in func_dict.keys():
        func1 = func_dict[key]
        # warmup
        for _ in range(nwarmups):
            x1 = func1(*args)
            jax.block_until_ready(x1)

        # benchmark func1
        t0 = time.time()
        for _ in range(nreps):
            x1 = func1(*args)
            jax.block_until_ready(x1)
        t1 = time.time()
        time1_tots = (t1 - t0) / nreps
        print(f"{key} time: {time1_tots:.3e} s")
        results[key] = time1_tots

    return results


def many_fxn_args_benchmark_timing(
    func_arg_dict,
    with_jit: bool = True,
    nwarmups: int = 5,
    nreps: int = 5,
):
    """
    Helper function to report timing of multiple (function, arg) pairs
    with robust error handling to manage exceptions like Out of Memory (OOM).

    Args:
        func_arg_dict: dictionary with keys as an identifier, values are dicts
                       with keys 'func' and 'args'
        with_jit: whether to JIT compile the functions
        nwarmups: number of warmup iterations to perform
        nreps: number of repetitions to time
    """
    results = {}

    if with_jit:
        for name in func_arg_dict.keys():
            try:
                func_arg_dict[name]["func"] = jax.jit(func_arg_dict[name]["func"])
            except Exception as e:
                results[name] = {"error": str(e)}
                print(f"JIT compilation failed for {name}: {str(e)}")
                continue  # Skip JIT compilation if it fails, log the error, and continue with other functions

    for key, value in func_arg_dict.items():
        func = value["func"]
        args = value["args"]
        try:
            # Warm-up phase
            for _ in range(nwarmups):
                x = func(*args)
                jax.block_until_ready(x)

            # Benchmark phase
            t0 = time.time()
            for _ in range(nreps):
                _, newton_iters = func(*args)
                jax.block_until_ready(x)
            t1 = time.time()
            time_elapsed = (t1 - t0) / nreps
            print(f"time: {time_elapsed:.3e} s")
            results[key] = time_elapsed
            results["newton_iters"] = newton_iters
        except Exception as e:
            # Handle exceptions during function execution
            error_message = f"Error during benchmarking {key}: {str(e)}"
            results[key] = {"error": error_message}
            traceback_str = traceback.format_exc()
            print(error_message)
            print(traceback_str)

    return results


"""
------------------------------------------------------------------------------------------------------------------------

Memory profiling helper functions

------------------------------------------------------------------------------------------------------------------------
"""



class MemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        self.keep_monitoring = False
        self.monitor_thread = threading.Thread(target=self.monitor_gpu)

    def start(self):
        self.peak_memory = 0
        self.keep_monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_gpu)
        self.monitor_thread.start()

    def stop(self):
        self.keep_monitoring = False
        self.monitor_thread.join()

    def monitor_gpu(self):
        while self.keep_monitoring:
            current_memory = self.memory_usage()
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            time.sleep(0.1)

    @staticmethod
    def memory_usage():
        GPUs = GPUtil.getGPUs()
        return sum(gpu.memoryUsed for gpu in GPUs)


def many_fxn_args_benchmark_memory(
    func_arg_dict,
    with_jit: bool = True,
    nwarmups: int = 5,
    nreps: int = 5,
):
    """
    Helper function to report memory usage of multiple (function, arg) pairs
    with robust error handling to manage exceptions like Out of Memory (OOM).

    Args:
        func_arg_dict: dictionary with keys as an identifier, values are dicts
                       with keys 'func' and 'args'
        with_jit: whether to JIT compile the functions
        nwarmups: number of warmup iterations to perform
        nreps: number of repetitions to time
    """
    results = {}


    if with_jit:
        for name in func_arg_dict.keys():
            try:
                func_arg_dict[name]["func"] = jax.jit(func_arg_dict[name]["func"])
            except Exception as e:
                results[name] = {"error": str(e)}
                print(f"JIT compilation failed for {name}: {str(e)}")
                continue  # Skip JIT compilation if it fails, log the error, and continue with other functions


    for key, value in func_arg_dict.items():
        func = value["func"]
        args = value["args"]
        try:
            jax.clear_caches()
            gc.collect()
            time.sleep(3)

            monitor = MemoryMonitor()
            monitor.start()

            # Benchmark phase
            gc.collect()
            for _ in range(nreps):
                x = func(*args)
                jax.block_until_ready(x)
                time.sleep(0.1)
            gc.collect()

            monitor.stop()

            print(f"memory used: {monitor.peak_memory:.3f} MB")
            results[key] = monitor.peak_memory
        except Exception as e:
            # Handle exceptions during function execution
            error_message = f"Error during benchmarking {key}: {str(e)}"
            results[key] = {"error": error_message}
            traceback_str = traceback.format_exc()
            print(error_message)
            print(traceback_str)

    return results
