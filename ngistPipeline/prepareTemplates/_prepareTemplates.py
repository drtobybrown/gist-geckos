import importlib.util
import logging
import os

from printStatus import printStatus

from ngistPipeline.prepareTemplates.template_cache import (
    _build_fingerprint,
    get_cache_path,
    read_cache,
    write_cache,
)


def prepareTemplates_Module(
    config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid=False
):
    """
    This function calls the prepareTemplates routine specified by the user.
    When CACHE_TEMPLATES is True (default), prepared templates are read from or
    written to a cache under TEMPLATE_DIR/.prepared to speed up later runs.
    """
    use_cache = config["GENERAL"].get("CACHE_TEMPLATES", True)
    if use_cache:
        fingerprint = _build_fingerprint(
            config, lmin, lmax, velscale, module_used, sortInGrid
        )
        cache_path = get_cache_path(config, fingerprint)
        cached = read_cache(cache_path)
        if cached is not None:
            logging.info("Using cached templates from %s", cache_path)
            return cached

    # Import the chosen prepareTemplates routine
    try:
        spec = importlib.util.spec_from_file_location(
            "",
            os.path.dirname(os.path.realpath(__file__))
            + "/"
            + config[module_used]["TEMPLATE_SET"]
            + ".py",
        )
        logging.info(
            "Using the routine for '" + config[module_used]["TEMPLATE_SET"] + ".py'"
        )
        prepTemplatesModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepTemplatesModule)
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = (
            "Failed to import the routine '"
            + config[module_used]["TEMPLATE_SET"]
            + ".py'"
        )
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    # Execute the chosen prepareTemplates routine
    try:
        (
            templates,
            lamRange_spmod,
            logLam2,
            ntemplates,
            logAge_grid,
            metal_grid,
            alpha_grid,
            ncomb,
            nAges,
            nMetal,
            nAlpha,
        ) = prepTemplatesModule.prepareSpectralTemplateLibrary(
            config,
            lmin,
            lmax,
            velscale,
            LSF_Data,
            LSF_Templates,
            module_used,
            sortInGrid,
        )
    except Exception as e:
        logging.critical(e, exc_info=True)
        message = "Routine '" + config[module_used]["TEMPLATE_SET"] + ".py' failed."
        printStatus.failed(message)
        logging.critical(message)
        return "SKIP"

    result = (
        templates,
        lamRange_spmod,
        logLam2,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    )
    if use_cache:
        write_cache(cache_path, result)
    return result
