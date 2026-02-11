import time
import subprocess
import shutil
import os
import psutil
import threading
import sys

def monitor_resources(pid, stop_event, results):
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    peak_memory = 0
    while not stop_event.is_set():
        try:
            # Check children as well since the pipeline might spawn processes
            children = process.children(recursive=True)
            total_rss = process.memory_info().rss
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            rss = total_rss / (1024 * 1024) # MB
            if rss > peak_memory:
                peak_memory = rss
            time.sleep(0.1)
        except psutil.NoSuchProcess:
            break
    results['peak_memory'] = peak_memory

def run_benchmark():
    start_time = time.time()

    # Paths (relative to tests/)
    # We assume this script is run from repo root, but if run from tests/, paths need adjustment.
    # Best to run from repo root: python tests/benchmark_pipeline.py

    output_dir = "./.github/workflows/tests/gistTutorial/results/NGC0000Example"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "ngistPipeline",
        "--config=./.github/workflows/tests/gistTutorial/configFiles/MasterConfig.yaml",
        "--default-dir=./.github/workflows/tests/gistTutorial/configFiles/defaultDir_ubuntu"
    ]

    print("Running benchmark...")
    process = subprocess.Popen(cmd)

    stop_event = threading.Event()
    results = {}
    monitor_thread = threading.Thread(target=monitor_resources, args=(process.pid, stop_event, results))
    monitor_thread.start()

    process.wait()
    stop_event.set()
    monitor_thread.join()

    end_time = time.time()
    duration = end_time - start_time

    print(f"Execution Time: {duration:.2f} seconds")
    print(f"Peak Memory: {results.get('peak_memory', 0):.2f} MB")

    if process.returncode != 0:
        print("Pipeline failed!")
        sys.exit(1)

    print("Benchmark completed successfully.")

if __name__ == "__main__":
    run_benchmark()
