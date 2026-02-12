import os
import sys

import numpy as np
from astropy.io import fits

output_dir = os.path.join(
    os.path.dirname(__file__), "gistTutorial", "results", "NGC0000Example"
)

if not os.path.isdir(output_dir):
    print(
        f"Output directory not found: {output_dir}\n"
        "The pipeline may have failed before writing output. Check the pipeline step.",
        file=sys.stderr,
    )
    sys.exit(1)

logfile = os.path.join(output_dir, "LOGFILE")
if not os.path.isfile(logfile):
    print(
        f"LOGFILE not found: {logfile}\n"
        "The pipeline may have failed before completing. Check the pipeline step.",
        file=sys.stderr,
    )
    sys.exit(1)

with open(logfile) as f:
    for line in f:
        print(line, end="")



#Check the pipeline creates the correct files
output_files = ["CONFIG","LOGFILE",
				# "NGC0000Example_AllSpectra.fits", # no longer outputted
				"NGC0000Example_BinSpectra_linear.hdf5",
				"NGC0000Example_BinSpectra.hdf5",
				"NGC0000Example_gas_BIN.fits",
				"NGC0000Example_gas-bestfit_BIN.fits",
				"NGC0000Example_gas-cleaned_BIN.fits",
				"NGC0000Example_kin-bestfit.fits",
				"NGC0000Example_kin-optimalTemplates.fits",
				"NGC0000Example_kin.fits",
				"NGC0000Example_ls_AdapRes.fits",
				"NGC0000Example_ls_OrigRes.fits",
				"NGC0000Example_ls-cleaned_linear.fits",
				"NGC0000Example_mask.fits",
				"NGC0000Example_sfh-bestfit.fits",
				"NGC0000Example_sfh-weights.fits",
				"NGC0000Example_sfh.fits",
				"NGC0000Example_table.fits",
				"NGC0000Example_KIN_maps.fits",
	       		"NGC0000Example_gas_BIN_maps.fits",
				"NGC0000Example_SFH_maps.fits",
				"NGC0000Example_LS_ORIGINAL_maps.fits",
				"NGC0000Example_LS_ADAPTED_maps.fits"]

for file in output_files:
    assert os.path.isfile(os.path.join(output_dir, file)), (
        f"{file} not created, probably the module failed"
    )
with open(logfile) as f:
    for line in f:
        print(line, end="")

# Check equivalent width (EW) functionality: gas FITS must contain EW columns
# (EW uses stellar continuum from KIN module when _kin-bestfit.fits is available)
gas_bin_path = f"{output_dir}/NGC0000Example_gas_BIN.fits"
with fits.open(gas_bin_path) as hdul:
    gas_table = hdul[1].data
    colnames = gas_table.columns.names
    ew_cols = [c for c in colnames if c.endswith("_EW")]
    ew_err_cols = [c for c in colnames if c.endswith("_EW_ERR")]
    cont_cols = [c for c in colnames if c.endswith("_CONT")]
assert len(ew_cols) >= 1, (
    f"Gas BIN FITS must have at least one *_EW column; found {ew_cols}"
)
assert len(ew_err_cols) >= 1, (
    f"Gas BIN FITS must have at least one *_EW_ERR column; found {ew_err_cols}"
)
# EW values: at least one column should have at least one finite value
n_finite_ew = 0
for c in ew_cols:
    n_finite_ew += np.sum(np.isfinite(np.asarray(gas_table[c])))
assert n_finite_ew >= 1, (
    "At least one equivalent width value must be finite in gas BIN FITS"
)
# KIN bestfit is required for EW to use stellar kinematics continuum
kin_bestfit_path = f"{output_dir}/NGC0000Example_kin-bestfit.fits"
assert os.path.isfile(kin_bestfit_path), (
    "KIN bestfit file required for EW continuum; pipeline should produce it before gas"
)
with fits.open(kin_bestfit_path) as kin_hdu:
    kin_bestfit = kin_hdu[1].data.BESTFIT
    kin_loglam = kin_hdu[2].data.LOGLAM
assert kin_bestfit.shape[0] == len(gas_table), (
    "KIN bestfit bin count should match gas table rows for EW continuum"
)
print("Equivalent width (EW) and KIN continuum checks passed.")

map_files = ["NGC0000Example_sfh.pdf",
		"NGC0000Example_gas-[NI]_5200.39_A_BIN.pdf",
		"NGC0000Example_gas-[NI]_5200.39_F_BIN.pdf",
		"NGC0000Example_gas-[NI]_5200.39_S_BIN.pdf",
		"NGC0000Example_gas-[NI]_5200.39_V_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_A_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_F_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_S_BIN.pdf",
		"NGC0000Example_gas-[OIII]_5006.77_V_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_A_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_F_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_S_BIN.pdf",
		"NGC0000Example_gas-Hb_4861.32_V_BIN.pdf",
		"NGC0000Example_kin.pdf",
		"NGC0000Example_ls_AdapRes.pdf",
		"NGC0000Example_ls_OrigRes.pdf",
		"NGC0000Example_ls-spp.pdf"]

#for file in map_files:
#	assert os.path.isfile(f"{output_dir}/maps/{file}"), f"{file} not created, probably the module failed"









