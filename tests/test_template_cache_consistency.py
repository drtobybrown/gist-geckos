"""
Test that template cache returns the same 11-tuple as uncached preparation.
Requires template library and config (e.g. gistTutorial); skips if not found.
"""
import os
import sys

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)


def _get_test_config():
    """Load config and paths from workflow gistTutorial if available."""
    config_path = os.path.join(
        PROJ_ROOT,
        ".github/workflows/tests/gistTutorial/configFiles/MasterConfig.yaml",
    )
    default_dir_path = os.path.join(
        PROJ_ROOT,
        ".github/workflows/tests/gistTutorial/configFiles/defaultDir_mac",
    )
    if not os.path.isfile(config_path):
        return None
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if os.path.isfile(default_dir_path):
        with open(default_dir_path, "r") as f:
            for line in f:
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.strip().partition("=")
                key, val = key.strip(), val.strip()
                resolved = os.path.abspath(os.path.join(PROJ_ROOT, val))
                if key == "templateDir" and os.path.isdir(resolved):
                    config["GENERAL"]["TEMPLATE_DIR"] = resolved
                elif key == "configDir" and os.path.isdir(resolved):
                    config["GENERAL"]["CONFIG_DIR"] = resolved
    template_dir = config["GENERAL"].get("TEMPLATE_DIR", "")
    library = config["KIN"].get("LIBRARY", "MILES/")
    lib_path = os.path.join(template_dir, library)
    if not os.path.isdir(lib_path):
        return None
    return config


def test_template_cache_consistency():
    """Cached template load returns same arrays as uncached (allclose)."""
    from ngistPipeline.auxiliary import _auxiliary
    from ngistPipeline.prepareTemplates import _prepareTemplates
    from ngistPipeline.prepareTemplates.template_cache import (
        _build_fingerprint,
        get_cache_path,
    )

    config = _get_test_config()
    if config is None:
        import pytest
        pytest.skip("gistTutorial config/template dir not found")

    config["GENERAL"]["CACHE_TEMPLATES"] = True
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "KIN")
    lmin = config["KIN"]["LMIN"]
    lmax = config["KIN"]["LMAX"]
    velscale = 70.0 / 2  # velscale_ratio 2
    module_used = "KIN"
    sortInGrid = False

    # Run 1: compute and write cache
    result1 = _prepareTemplates.prepareTemplates_Module(
        config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid
    )
    if result1 == "SKIP":
        import pytest
        pytest.skip("prepareTemplates_Module returned SKIP")
    assert len(result1) == 11

    # Run 2: read from cache
    fingerprint = _build_fingerprint(config, lmin, lmax, velscale, module_used, sortInGrid)
    cache_path = get_cache_path(config, fingerprint)
    assert os.path.isfile(cache_path), "Cache file should exist after first run"
    result2 = _prepareTemplates.prepareTemplates_Module(
        config, lmin, lmax, velscale, LSF_Data, LSF_Templates, module_used, sortInGrid
    )
    assert result2 != "SKIP"
    assert len(result2) == 11

    # Compare
    for i in range(11):
        a, b = result1[i], result2[i]
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            np.testing.assert_allclose(a, b, err_msg="item %d" % i)
        elif np.isscalar(a) and np.isscalar(b):
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == b or (np.isfinite(a) and np.isfinite(b) and np.isclose(a, b)), "item %d" % i
        else:
            assert a == b, "item %d" % i


def test_template_cache_read_write_roundtrip():
    """read_cache(write_cache(result)) matches result for a small synthetic payload."""
    try:
        import h5py
    except ImportError:
        import pytest
        pytest.skip("h5py required")
    import tempfile
    import numpy as np
    from ngistPipeline.prepareTemplates.template_cache import read_cache, write_cache

    templates = np.random.randn(100, 10).astype(np.float64)
    logLam = np.linspace(np.log(4800), np.log(7000), 100)
    lamRange = [4800.0, 7000.0]
    ntemplates, ncomb = 10, 10
    nAges, nMetal, nAlpha = 5, 2, 1
    logAge_grid = np.nan
    metal_grid = np.nan
    alpha_grid = np.nan
    result = (
        templates,
        lamRange,
        logLam,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.h5")
        write_cache(path, result)
        assert os.path.isfile(path)
        back = read_cache(path)
        assert back is not None
        np.testing.assert_allclose(back[0], result[0])
        np.testing.assert_allclose(back[2], result[2])
        assert back[3] == result[3]
        assert back[7] == result[7]


if __name__ == "__main__":
    test_template_cache_read_write_roundtrip()
    test_template_cache_consistency()
    print("Template cache tests passed.")
