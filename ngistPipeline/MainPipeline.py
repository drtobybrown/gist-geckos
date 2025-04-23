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
from astropy.io import ascii, fits
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
import importlib.util
import logging
import optparse
import sys
import time

import matplotlib

matplotlib.use("pdf")

from printStatus import printStatus

from ngistPipeline._version import __version__
from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.auxiliary import _performance
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


def skipGalaxy(config):
    # _auxiliary.addGISTHeaderComment(config)
    printStatus.module("The nGIST pipeline")
    printStatus.failed("Galaxy is skipped!")
    logging.critical("Galaxy is skipped!")


def numberOfGalaxies(filename):
    """
    Returns the number of galaxies to be analysed, as stated in the config file.
    """
    i = 0
    for line in open(filename):
        if not line.startswith("#"):
            i = i + 1
    return i


def runGIST(dirPath, galindex):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - -  I N I T I A L I S E   T H E   G I S T  - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - INITIALISE MODULE - - - - -
    # Read config
    config = _initialise.readMasterConfig(dirPath.configFile, galindex)
    config = _initialise.addPathsToConfig(config, dirPath)

    # Print configurations
    _initialise.printConfig(config)

    # Check output directory
    _initialise.checkOutputDirectory(config)

    # Setup logfile
    _initialise.setupLogfile(config)
    sys.excepthook = _initialise.handleUncaughtException

    # Setup performance monitor
    perf_monitor = _performance.setup_performance_monitor(config)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  P R E P A R A T I O N   M O D U L E S  - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - READ_DATA MODULE - - - - -
    perf_monitor.start_module("read_data")
    cube = _readData.readData_Module(config)
    perf_monitor.end_module()
    if cube == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - SPATIAL MASKING MODULE - - - - -
    perf_monitor.start_module("spatial_masking")
    _ = _spatialMasking.spatialMasking_Module(config, cube)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - SPATIAL BINNING MODULE - - - - -
    perf_monitor.start_module("spatial_binning")
    _ = _spatialBinning.spatialBinning_Module(config, cube)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - PREPARE SPECTRA MODULE - - - - -
    perf_monitor.start_module("prepare_spectra")
    _ = _prepareSpectra.prepareSpectra_Module(config, cube)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    del cube

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - -   A N A L Y S I S   M O D U L E S   - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - STELLAR KINEMATICS MODULE - - - - -
    perf_monitor.start_module("stellar_kinematics")
    _ = _stellarKinematics.stellarKinematics_Module(config)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - CONTINUUM CUBE MODULE - - - - -
    perf_monitor.start_module("continuum_cube")
    _ = _continuumCube.continuumCube_Module(config)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - EMISSION LINES MODULE - - - - -
    perf_monitor.start_module("emission_lines")
    _ = _emissionLines.emissionLines_Module(config)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - STAR FORMATION HISTORIES MODULE - - - - -
    perf_monitor.start_module("star_formation_histories")
    _ = _starFormationHistories.starFormationHistories_Module(config)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - LINE STRENGTHS MODULE - - - - -
    perf_monitor.start_module("line_strengths")
    _ = _lineStrengths.lineStrengths_Module(config)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # - - - - - USERS  MODULE - - - - -
    perf_monitor.start_module("user_modules")
    _ = _userModules.user_Modules(config)
    perf_monitor.end_module()
    if _ == "SKIP":
        skipGalaxy(config)
        return None

    # Save performance metrics
    perf_monitor.save_benchmark()
    
    # Print performance summary
    summary = perf_monitor.get_summary()
    logging.info("Performance Summary:")
    logging.info(f"Total Duration: {summary['total_duration']:.2f} seconds")
    logging.info(f"Parallel Mode: {'Enabled' if summary['parallel_mode'] else 'Disabled'}")
    logging.info(f"Total Cores: {summary['total_cores']}")
    logging.info(f"Threads Available: {summary['num_threads']}")
    
    for module, metrics in summary['modules'].items():
        logging.info(f"\nModule: {module}")
        logging.info(f"  Duration: {metrics['duration_seconds']:.2f} seconds")
        if summary['parallel_mode']:
            logging.info(f"  Max Cores Used: {metrics['max_cores_used']:.1f} of {metrics['max_cores_available']}")
            logging.info(f"  Avg CPU Usage: {metrics['avg_cpu_percent']:.1f}% ({metrics['avg_cpu_percent_per_core']:.1f}% per core)")
            logging.info(f"  Max CPU Usage: {metrics['max_cpu_percent']:.1f}% ({metrics['max_cpu_percent_per_core']:.1f}% per core)")
        else:
            logging.info(f"  Avg CPU Usage: {metrics['avg_cpu_percent']:.1f}%")
            logging.info(f"  Max CPU Usage: {metrics['max_cpu_percent']:.1f}%")
        logging.info(f"  Avg Memory: {metrics['avg_memory_usage_mb']:.1f} MB")
        logging.info(f"  Max Memory: {metrics['max_memory_usage_mb']:.1f} MB")
        if 'total_disk_read_mb' in metrics and 'total_disk_write_mb' in metrics:
            logging.info(f"  Disk Read: {metrics['total_disk_read_mb']:.1f} MB")
            logging.info(f"  Disk Write: {metrics['total_disk_write_mb']:.1f} MB")
        else:
            logging.info("  Disk I/O metrics not available")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - -  F I N A L I S E   T H E   A N A L Y S I S  - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Branding
    # _auxiliary.addGISTHeaderComment(config)

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

    # Iterate over galaxies in Config-file
    ngalaxies = 1  # numberOfGalaxies(dirPath.configFile) - 2 # Amelia changed this because she changed the format of the config file. We can revisit should we ever need to run more than one galaxy per config file
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
