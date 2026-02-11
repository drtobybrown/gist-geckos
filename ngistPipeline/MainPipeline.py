#!/usr/bin/env python

# ==================================================================================================================== #
#                                                                                                                      #
#                                          T H E   G I S T   P I P E L I N E                                           #
#                                                                                                                      #
# ==================================================================================================================== #


import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings

import numpy as np

warnings.filterwarnings("ignore")
import importlib.util
import logging
import optparse
import sys
import time
import psutil
import threading

import matplotlib

matplotlib.use("pdf")

from printStatus import printStatus

from ngistPipeline._version import __version__
from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.continuumCube import _continuumCube
from ngistPipeline.emissionLines import _emissionLines
from ngistPipeline.initialise import _initialise
from ngistPipeline.lineStrengths import _lineStrengths
from ngistPipeline.prepareSpectra import _prepareSpectra
from ngistPipeline.readData import _readData
from ngistPipeline.spatialBinning import _spatialBinning
from ngistPipeline.spatialMasking import _spatialMasking
from ngistPipeline.starFormationHistories import _starFormationHistories
from ngistPipeline.stellarKinematics import _stellarKinematics
from ngistPipeline.userModules import _userModules


class ResourceMonitor:
    def __init__(self):
        self.stats = {}
        self._stop_event = threading.Event()
        self._thread = None
        self._peak_memory = 0

    def start_monitoring(self):
        self._stop_event.clear()
        self._peak_memory = 0
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()

    def stop_monitoring(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        return self._peak_memory

    def _monitor(self):
        process = psutil.Process()
        while not self._stop_event.is_set():
            try:
                # Include children in memory calculation
                mem = process.memory_info().rss
                try:
                    children = process.children(recursive=True)
                    for child in children:
                        try:
                            mem += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                if mem > self._peak_memory:
                    self._peak_memory = mem
                time.sleep(0.1)
            except Exception:
                pass

    def record_module(self, name, func, *args, **kwargs):
        self.start_monitoring()
        start_time = time.time()
        start_cpu = psutil.Process().cpu_times()

        try:
            result = func(*args, **kwargs)
        finally:
            peak_mem = self.stop_monitoring()
            end_time = time.time()
            end_cpu = psutil.Process().cpu_times()

            wall_time = end_time - start_time
            cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)

            self.stats[name] = {
                "wall_time": wall_time,
                "cpu_time": cpu_time,
                "peak_memory_mb": peak_mem / (1024 * 1024)
            }

        return result

    def log_report(self):
        logging.info("\n" + "="*60)
        logging.info("RESOURCE USAGE REPORT")
        logging.info(f"{'Module':<25} {'Wall Time (s)':<15} {'CPU Time (s)':<15} {'Peak Mem (MB)':<15}")
        logging.info("-" * 70)

        total_wall = 0
        total_cpu = 0
        max_mem = 0

        for name, data in self.stats.items():
            logging.info(f"{name:<25} {data['wall_time']:<15.2f} {data['cpu_time']:<15.2f} {data['peak_memory_mb']:<15.2f}")
            total_wall += data['wall_time']
            total_cpu += data['cpu_time']
            max_mem = max(max_mem, data['peak_memory_mb'])

        logging.info("-" * 70)
        logging.info(f"{'TOTAL':<25} {total_wall:<15.2f} {total_cpu:<15.2f} {max_mem:<15.2f} (Max Peak)")
        logging.info("="*60 + "\n")


def skipGalaxy(config):
    # _auxiliary.addGISTHeaderComment(config)
    printStatus.module("The nGIST pipeline")
    printStatus.failed("Galaxy is skipped!")
    logging.critical("Galaxy is skipped!")


def runGIST(dirPath, galindex):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - -  I N I T I A L I S E   T H E   G I S T  - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    monitor = ResourceMonitor()

    # - - - - - INITIALISE MODULE - - - - -
    # Read config
    config = _initialise.readMasterConfig(dirPath.configFile, galindex)
    config = _initialise.addPathsToConfig(config, dirPath)

    # Pre-flight resource check
    if not _initialise.checkResources(config):
        skipGalaxy(config)
        return None

    # Staging logic
    use_scratch = False
    original_input = config["GENERAL"]["INPUT"]
    original_output = config["GENERAL"]["OUTPUT"]
    staging_dir = None

    scratch_dir = os.environ.get("SCRATCH_DIR", "/scratch")
    if os.path.exists(scratch_dir) and os.access(scratch_dir, os.W_OK):
        try:
            import tempfile
            import shutil
            use_scratch = True
            run_id = config["GENERAL"]["RUN_ID"]
            staging_dir = tempfile.mkdtemp(prefix=f"ngist_{run_id}_", dir=scratch_dir)
            printStatus.module("Staging")
            printStatus.running(f"Staging input data to {staging_dir}")

            # Copy input file
            input_filename = os.path.basename(original_input)
            staged_input = os.path.join(staging_dir, input_filename)
            shutil.copy2(original_input, staged_input)
            config["GENERAL"]["INPUT"] = staged_input

            # Copy mask file if present
            if config.get("SPATIAL_MASKING") and config["SPATIAL_MASKING"].get("MASK"):
                 mask_filename = config["SPATIAL_MASKING"]["MASK"]
                 original_mask_path = os.path.join(os.path.dirname(original_input), mask_filename)
                 if os.path.exists(original_mask_path):
                     shutil.copy2(original_mask_path, os.path.join(staging_dir, mask_filename))

            # Setup staged output
            staged_output = os.path.join(staging_dir, "output", run_id)
            if not os.path.exists(os.path.dirname(staged_output)):
                os.makedirs(os.path.dirname(staged_output))
            # MainPipeline expects RUN_ID logic in output path usually, handled by addPathsToConfig.
            # Here config["GENERAL"]["OUTPUT"] is full path to RUN_ID folder.
            config["GENERAL"]["OUTPUT"] = staged_output

            printStatus.updateDone(f"Staging complete. Running in {staging_dir}")
        except Exception as e:
            logging.error(f"Staging failed: {e}")
            printStatus.warning("Staging failed, falling back to direct I/O")
            if staging_dir and os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
            use_scratch = False
            config["GENERAL"]["INPUT"] = original_input
            config["GENERAL"]["OUTPUT"] = original_output

    try:
        # Print configurations
        _initialise.printConfig(config)

        # Check output directory
        _initialise.checkOutputDirectory(config)

        # Setup logfile
        _initialise.setupLogfile(config)
        sys.excepthook = _initialise.handleUncaughtException

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - - - -  P R E P A R A T I O N   M O D U L E S  - - - - - - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # - - - - - READ_DATA MODULE - - - - -

        cube = monitor.record_module("readData", _readData.readData_Module, config)
        if cube == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - SPATIAL MASKING MODULE - - - - -

        _ = monitor.record_module("spatialMasking", _spatialMasking.spatialMasking_Module, config, cube)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - SPATIAL BINNING MODULE - - - - -

        _ = monitor.record_module("spatialBinning", _spatialBinning.spatialBinning_Module, config, cube)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - PREPARE SPECTRA MODULE - - - - -

        _ = monitor.record_module("prepareSpectra", _prepareSpectra.prepareSpectra_Module, config, cube)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        del cube

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - - - - - -   A N A L Y S I S   M O D U L E S   - - - - - - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # - - - - - STELLAR KINEMATICS MODULE - - - - -

        _ = monitor.record_module("stellarKinematics", _stellarKinematics.stellarKinematics_Module, config)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - CONTINUUM CUBE MODULE - - - - -

        _ = monitor.record_module("continuumCube", _continuumCube.continuumCube_Module, config)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - EMISSION LINES MODULE - - - - -

        _ = monitor.record_module("emissionLines", _emissionLines.emissionLines_Module, config)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - STAR FORMATION HISTORIES MODULE - - - - -

        _ = monitor.record_module("starFormationHistories", _starFormationHistories.starFormationHistories_Module, config)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - LINE STRENGTHS MODULE - - - - -

        _ = monitor.record_module("lineStrengths", _lineStrengths.lineStrengths_Module, config)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

        # - - - - - USERS  MODULE - - - - -

        _ = monitor.record_module("userModules", _userModules.user_Modules, config)
        if _ == "SKIP":
            skipGalaxy(config)
            return None

    finally:
        # Copy back results if staging was used
        if use_scratch and os.path.exists(config["GENERAL"]["OUTPUT"]):
            try:
                printStatus.running(f"Copying results back to {original_output}")
                import shutil
                if not os.path.exists(original_output):
                    os.makedirs(original_output)

                # Copy contents
                for item in os.listdir(config["GENERAL"]["OUTPUT"]):
                    s = os.path.join(config["GENERAL"]["OUTPUT"], item)
                    d = os.path.join(original_output, item)
                    if os.path.isdir(s):
                        if os.path.exists(d): shutil.rmtree(d)
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)

                printStatus.updateDone("Results copied back.")
                # Only clean up if copy back succeeded
                if staging_dir and os.path.exists(staging_dir):
                    shutil.rmtree(staging_dir)
            except Exception as e:
                printStatus.updateFailed(f"Failed to copy results back: {e}")
                logging.error(f"Failed to copy results back: {e}")
                printStatus.warning(f"Data remains in staging directory: {staging_dir}")

            # Restore config paths so finalization steps (and any future logic) use the persistent locations
            config["GENERAL"]["INPUT"] = original_input
            config["GENERAL"]["OUTPUT"] = original_output


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  F I N A L I S E   T H E   A N A L Y S I S  - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Branding
    # _auxiliary.addGISTHeaderComment(config)

    # Resource Report
    monitor.log_report()

    # Goodbye
    printStatus.module("nGIST pipeline")
    printStatus.done("nGIST completed successfully.")
    logging.info("nGIST completed successfully.")


# ============================================================================ #
#                           M A I N   F U N C T I O N                          #
# ============================================================================ #
def main(args=None):
    # Capture command-line arguments
    parser = optparse.OptionParser(usage="%ngistPipeline [options] arg")
    jls_extract_var = "configFile"
    parser.add_option(
        "--config",
        dest=jls_extract_var,
        type="string",
        help="State the path of the config file.",
    )
    parser.add_option(
        "--default-dir",
        dest="defaultDir",
        type="string",
        help="File defining default directories for input, output, configuration files, and spectral templates.",
    )
    (dirPath, args) = parser.parse_args()

    # Check if required command-line argument is given
    if dirPath.configFile == None:
        printStatus.failed(
            "Please specify the path of the config file to be used. Exit!"
        )
        exit(1)

    # Check if Config-file exists
    if os.path.isfile(dirPath.configFile) == False:
        printStatus.failed("Config file at " + dirPath.configFile + " not found. Exit!")
        exit(1)

    # Single-galaxy YAML config (one run per config file)
    ngalaxies = 1
    if ngalaxies <= 0:
        message = "The number of runs defined in the config file seems to be 0. Exit."
        printStatus.failed(message)
        exit(1)
    for galindex in range(ngalaxies):
        runGIST(dirPath, galindex)
        print("\n")


if __name__ == "__main__":
    # Call the main function
    main()
