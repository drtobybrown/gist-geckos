import logging
import os
import shutil
import sys
import time

import yaml
from printStatus import printStatus

from ngistPipeline._version import __version__

"""
PURPOSE:
  This file contains a collection of functions necessary to initialise the
  pipeline. This includes the creation of the LOGFILE, functions to read, save,
  and check the MasterConfig file, and print the configurations to stdout.

  The functions in this file do not interfere with subsequent modules or their
  configuration parameters provided in MasterConfig.

  02 Mar 2023: AFM editing to change configFile to a .yaml to make more user friendly
"""


def setupLogfile(config):
    """Initialise the LOGFILE."""
    welcomeString = "\n\n# ============================================== #\n#{:^48}#\n#{:^48}#\n# ============================================== #\n".format(
        "THE GIST PIPELINE", "Version " + __version__
    )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(config["GENERAL"]["OUTPUT"], "LOGFILE"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)-8s - %(module)s: %(message)s",
        datefmt="%m/%d/%y %H:%M:%S",
    )
    logging.Formatter.converter = time.gmtime
    logging.info(welcomeString)


def handleUncaughtException(exceptionType, exceptionValue, exceptionTraceback):
    """Write error message of uncaught exceptions in the logfile."""
    logging.error(
        "Uncaught Exception",
        exc_info=(exceptionType, exceptionValue, exceptionTraceback),
    )
    sys.__excepthook__(exceptionType, exceptionValue, exceptionTraceback)
    print("\n")
    printStatus.failed("FATAL ERROR! The execution of the pipeline is terminated.")
    print("")


def readMasterConfig(filename, galindex):
    """
    Read the MasterConfig file and return all parameters as a config dictionary.
    galindex is accepted for API compatibility but not used (single-galaxy YAML).
    """
    # Amelia edited this module to instead of reading in the old MasterConfig, to read in MasterConfig.yaml
    with open(filename, "r") as f:
        configs = yaml.safe_load(f)

    return configs


def addPathsToConfig(
    config, dirPath
):  # Amelia: input is the config dictionary
    """
    Combine the configuration parameters from MasterConfig with the paths specified as command line arguments.

    Naturally, this function cannot account for paths in user-defined parameters and/or user-defined modules.
    """
    if os.path.isfile(dirPath.defaultDir) == True:
        for line in open(dirPath.defaultDir, "r"):

        #for line in open('configFiles/defaultDir', "r"): #Amelia uncomment for testing only. Need to be in gistTutorial folder
            if not line.startswith('#'):
                line = line.split('=')
                line = [x.strip() for x in line]
                if len(line) < 2:
                    continue
                if os.path.isdir(line[1]) == True:
                    if line[0] == "inputDir":
                        config["GENERAL"]["INPUT"] = os.path.join(
                            line[1], config["GENERAL"]["INPUT"]
                        )
                    elif line[0] == "outputDir":
                        config["GENERAL"]["OUTPUT"] = os.path.join(
                            line[1],
                            config["GENERAL"]["OUTPUT"],
                            config["GENERAL"]["RUN_ID"],
                        )
                    elif line[0] == "configDir":
                        config["GENERAL"]["CONFIG_DIR"] = line[1]
                    elif line[0] == "templateDir":
                        config["GENERAL"]["TEMPLATE_DIR"] = line[1]
                elif line[1] == "outputDir" and line[0] == "configDir":
                    config["GENERAL"]["CONFIG_DIR"] = config["GENERAL"]["OUTPUT"]
                else:
                    print(
                        "WARNING! "
                        + line[1]
                        + " specified as default "
                        + line[0]
                        + " is not a directory!"
                    )
    else:
        print("WARNING! " + dirPath.defaultDir + " is not a file!")

    return config


def ensureNCPU(config):
    """
    If NCPU is not set or invalid in config["GENERAL"], set it to the number of
    CPUs available (os.cpu_count()). Default: use all available cores for faster runs.
    """
    general = config["GENERAL"]
    ncpu = general.get("NCPU")
    try:
        ncpu = int(ncpu) if ncpu is not None else None
    except (TypeError, ValueError):
        ncpu = None
    if ncpu is None or ncpu < 1:
        ncpu = os.cpu_count()
        if ncpu is None or ncpu < 1:
            ncpu = 1
    general["NCPU"] = ncpu
    return config


def checkOutputDirectory(config):
    """
    Create output directory if it does not exist yet.
    """
    if os.path.isdir(config["GENERAL"]["OUTPUT"]) == False:
        os.mkdir(config["GENERAL"]["OUTPUT"])
        saveConfig(config)
    else:
        if config["GENERAL"]["OW_CONFIG"] == True:
            saveConfig(config)
        else:
            loadedConfig = loadConfig(config["GENERAL"]["OUTPUT"])
            checkConfig(config, loadedConfig)

    return None


def saveConfig(config):
    """
    Save configurations from MasterConfig in the output directory of the current run.
    """
    with open(os.path.join(config["GENERAL"]["OUTPUT"], "CONFIG"), "w") as file:
        yaml.dump(config, file, sort_keys=False)
    return None


def loadConfig(outdir):
    """
    Load configurations from a saved CONFIG file in the output directory of the current run.
    """
    with open(os.path.join(outdir, "CONFIG"), "r") as file:
        loadedConfig = yaml.safe_load(file)
    return loadedConfig


def checkConfig(config, loadedConfig):
    """
    Compare to config dictionaries.
    """
    if config != loadedConfig:
        message = "The configurations set in MasterConfig and those saved in the output directory are not identical. Please double-check your configurations. The analysis will continue with the configurations from MasterConfig, however, this does not imply that any previous results are compatible with these configurations."
        printStatus.warning(message)
    return None


def checkResources(config):
    """
    Check if resources (disk space, memory) are sufficient.
    """
    printStatus.running("Checking resources")

    # Check input existence
    if not os.path.exists(config["GENERAL"]["INPUT"]):
        printStatus.failed(f"Input file not found: {config['GENERAL']['INPUT']}")
        return False

    # Check disk space in output directory
    output_dir = config["GENERAL"]["OUTPUT"]
    # If output dir doesn't exist, check parent
    check_dir = output_dir
    while not os.path.exists(check_dir):
        parent = os.path.dirname(check_dir)
        if not parent or parent == check_dir: # reached root or invalid
            check_dir = "." # fall back to current dir
            break
        check_dir = parent

    if os.path.exists(check_dir):
        try:
            total, used, free = shutil.disk_usage(check_dir)
            # Estimate required space: Input size * 10?
            if os.path.exists(config["GENERAL"]["INPUT"]):
                input_size = os.path.getsize(config["GENERAL"]["INPUT"])
                required = input_size * 10
                if free < required:
                    printStatus.warning(f"Low disk space! Free: {free/1024**3:.2f} GB, Estimated required: {required/1024**3:.2f} GB")
            else:
                 printStatus.warning("Input file missing during resource check.")
        except Exception as e:
            logging.warning(f"Failed to check disk usage: {e}")

    printStatus.updateDone("Resources checked")
    return True


def printConfig(config):
    """
    Print an overview of the configuration parameters to stdout and in the logfile.
    """
    os.system("clear")
    headerString = (
        "\n"
        "\033[0;37m"
        + "********************************************************************"
        + "\033[0;39m\n"

        "\033[0;37m"
        + "*     The      ____ ___ ____ _____                                 *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*       _ __  / ___|_ _/ ___|_   _|                                *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*      | '_ \| |  _ | |\___ \ | |                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*      | | | | |_| || | ___) || |                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*      |_| |_|\____|___|____/ |_|  Pipeline                        *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*                                                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*                                                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*                                                                  *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "* .    _     *         |     .       .     <==X==>           +     *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*    .' \\\`.     +    -*-     *   .                .   *           *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "* .  |__''_|  .        |   +         .    +       .                *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*    |     | .                                        .     -*-    *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*    |     |           `  .    '      *     . *   .    +    '      *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*  _.'-----'-._     *                  .                           *"
        + "\033[0;39m\n"
        "\033[0;37m"
        + "*/             \__.__.--._______________                           *"
        + "\033[0;39m\n"

        "\033[0;37m"
        + "********************************************************************"
        + "\033[0;39m\n"
    )
    infoString = ""
    for mK in config.keys():
        infoString = infoString + "\n"
        infoString = infoString + mK + "\n"
        for pK in config[mK].keys():
            infoString = infoString + "    {:13}{}\n".format(
                str(pK) + ":", str(config[mK][pK])
            )

    print(headerString + infoString + "\n")
