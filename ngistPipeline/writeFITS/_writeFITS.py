import logging
import os
import traceback

from printStatus import printStatus

from ngistPipeline.writeFITS import save_maps_fits


def generateFITS(config, module):
    """
    generateFITS _summary_

    Args:
        config (_type_): _description_
        module (_type_): _description_

    Returns:
        _type_: _description_
    """

    outputPrefix = os.path.join(
        config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]
    )


    # - - - - - TABLES MODULE - - - - -
    if module == "SPATIAL_BINNING":
        outdir_sb = config["GENERAL"]["OUTPUT"]
        run_id_sb = config["GENERAL"]["RUN_ID"]
        table_path_sb = os.path.join(outdir_sb, run_id_sb) + "_table.fits"
        try:
            printStatus.running("Producing table binned maps in FITS format")
            logging.info(
                "FITS maps SPATIAL_BINNING: OUTPUT=%s RUN_ID=%s table exists=%s",
                outdir_sb, run_id_sb, os.path.isfile(table_path_sb),
            )
            save_maps_fits.savefitsmaps("SPATIAL_BINNING", config["SPATIAL_BINNING"]["METHOD"], outdir_sb)
            printStatus.updateDone("Producing table binned maps in FITS format")
            logging.info("Produced table binned maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing table binned maps in FITS format")
            msg = "Producing table binned maps failed: %s: %s" % (type(e).__name__, str(e))
            logging.error(msg)
            print(msg, flush=True)
            print("OUTPUT=%s table_path=%s exists=%s" % (outdir_sb, table_path_sb, os.path.isfile(table_path_sb)), flush=True)
            logging.error("OUTPUT=%s table_path=%s exists=%s", outdir_sb, table_path_sb, os.path.isfile(table_path_sb))
            logging.exception("Traceback for table binned maps failure")
            traceback.print_exc()

    # - - - - - STELLAR KINEMATICS MODULE - - - - -
    if module == "KIN":
        outdir_kin = config["GENERAL"]["OUTPUT"]
        run_id = config["GENERAL"]["RUN_ID"]
        kin_fits = os.path.join(outdir_kin, run_id) + "_kin.fits"
        maps_fits = os.path.join(outdir_kin, run_id) + "_KIN_maps.fits"
        try:
            printStatus.running("Producing stellar kinematics maps in FITS format")
            logging.info(
                "FITS maps KIN: OUTPUT=%s RUN_ID=%s _kin.fits exists=%s _KIN_maps.fits exists=%s",
                outdir_kin, run_id, os.path.isfile(kin_fits), os.path.isfile(maps_fits),
            )
            print("FITS maps KIN: OUTPUT=%s RUN_ID=%s _kin.fits exists=%s _KIN_maps.fits exists=%s" % (outdir_kin, run_id, os.path.isfile(kin_fits), os.path.isfile(maps_fits)), flush=True)
            save_maps_fits.savefitsmaps("KIN", config["KIN"]["METHOD"], outdir_kin)
            printStatus.updateDone("Producing stellar kinematics maps in FITS format")
            logging.info("Produced stellar kinematics maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing stellar kinematics maps in FITS format")
            msg = "Producing stellar kinematics maps failed: %s: %s" % (type(e).__name__, str(e))
            logging.error(msg)
            print(msg, flush=True)
            print("OUTPUT=%s RUN_ID=%s kin_fits=%s exists=%s" % (outdir_kin, run_id, kin_fits, os.path.isfile(kin_fits)), flush=True)
            logging.error("OUTPUT=%s RUN_ID=%s kin_fits=%s exists=%s", outdir_kin, run_id, kin_fits, os.path.isfile(kin_fits))
            logging.exception("Traceback for stellar kinematics maps failure")
            traceback.print_exc()

    # - - - - - CONTINUUM CUBE MODULE - - - - -
    if module == "CONT":
        try:
            printStatus.running(
                "Producing continuum-only and line-only cubes in FITS format"
            )
            save_maps_fits.saveContLineCube(config)
            printStatus.updateDone(
                "Producing continuum-only and line-only cubes in FITS format"
            )
            logging.info("Produced continuum-only and line-only cubes in FITS format")
        except Exception as e:
            printStatus.updateFailed(
                "Producing continuum-only and line-only cubes in FITS format"
            )
            logging.error(e, exc_info=True)
            logging.error(
                "Failed to produce continuum-only and line-only cubes in FITS format"
            )

    # - - - - - EMISSION LINES MODULE - - - - -
    if module == "GAS":
        try:
            printStatus.running("Producing FITS maps from the emission-line analysis")
            if os.path.isfile(outputPrefix + "_gas_BIN.fits") == True:
                if config["GAS"]["LEVEL"] == "BIN": #And we aren't running in BOTH mode
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL=config["GAS"]["LEVEL"],
                        AoNThreshold=4,
                    )
            if os.path.isfile(outputPrefix + "_gas_SPAXEL.fits") == True:
                if config["GAS"]["LEVEL"] == 'SPAXEL': #And we aren't running in BOTH mode
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL=config["GAS"]["LEVEL"],
                        AoNThreshold=4,
                    )
            if os.path.isfile(outputPrefix + "_gas_BIN.fits") == True and os.path.isfile(outputPrefix + "_gas_SPAXEL.fits") == True:
                if config["GAS"]["LEVEL"] == 'BOTH': # Special case for running in BOTH mode
                    # Run first to create the _BIN maps
                    print('First run-through to save bin results')
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL='BIN',
                        AoNThreshold=4,
                    )
                    # Then run to create the SPAXEL maps
                    print('second run through to save SPAXEL results')
                    save_maps_fits.savefitsmaps_GASmodule(
                        "gas",
                        config["GENERAL"]["OUTPUT"],
                        LEVEL='SPAXEL',
                        AoNThreshold=4,
                    )

            printStatus.updateDone(
                "Producing FITS maps from the emission-line analysis"
            )
            logging.info("Producing FITS maps from the emission-line analysis")
        except Exception as e:
            printStatus.updateFailed(
                "Producing FITS maps from the emission-line analysis"
            )
            logging.error(e, exc_info=True)
            logging.error("Failed to produce maps from the emission-line analysis.")

    # - - - - - STAR FORMATION HISTORIES MODULE - - - - -
    if module == "SFH":
        try:
            printStatus.running("Producing SFH maps in FITS format")
            save_maps_fits.savefitsmaps("SFH", config["SFH"]["METHOD"], config["GENERAL"]["OUTPUT"])
            printStatus.updateDone("Producing SFH maps in FITS format")
            logging.info("Produced SFH maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing SFH maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce SFH maps.")

    # - - - - - LINE STRENGTHS MODULE - - - - -
    if module == "LS":
        try:
            printStatus.running("Producing line strength maps in FITS format")
            save_maps_fits.savefitsmaps_LSmodule(
                "LS", config["GENERAL"]["OUTPUT"], "ORIGINAL"
            )
            save_maps_fits.savefitsmaps_LSmodule(
                "LS", config["GENERAL"]["OUTPUT"], "ADAPTED"
            )
            printStatus.updateDone("Producing line strength maps in FITS format")
            logging.info("Produced line strength maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing line strength maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce line strength maps.")

    # - - - - - USER MODULE - - - - -
    if module == "UMOD":
        try:
            printStatus.running("Producing User Module maps in FITS format")
            save_maps_fits.savefitsmaps("UMOD",  config["UMOD"]["METHOD"],
                                        config["GENERAL"]["OUTPUT"]
                                       )
            printStatus.updateDone("Producing User Module maps in FITS format")
            logging.info("Produced User Module maps in FITS format")
        except Exception as e:
            printStatus.updateFailed("Producing User Module maps in FITS format")
            logging.error(e, exc_info=True)
            logging.error("Failed to produce User Module component maps.")


    return None
