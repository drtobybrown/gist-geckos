"""
LazyCube — memory-efficient wrapper for IFU data cubes.

Instead of loading the full (nwave, nspaxels) flux + error arrays into RAM,
LazyCube keeps only the small per-spaxel statistics (x, y, signal, noise, snr,
defunct_mask) and streams the full spectral data on demand via ``tile_iterator``.

This enables processing cubes many times larger than available memory on CANFAR
(e.g. a 64 GB cube on a 16 GB node).

Downstream modules that only need per-spaxel statistics (spatialMasking,
spatialBinning) work unchanged because LazyCube extends ``dict``.  Modules
that need full spectra (prepareSpectra) use ``cube.tile_iterator()`` to
stream spatial tiles.

Accessing ``cube["spec"]`` or ``cube["error"]`` raises ``MemoryError`` with
a helpful message directing callers to ``tile_iterator()``.
"""

import logging
import numpy as np

try:
    import fitsio
except ImportError:
    fitsio = None


class LazyCube(dict):
    """Dict-like cube that streams large spectral arrays from FITS on demand.

    Small keys stored in memory (fits any system):
        x, y, wave, signal, noise, snr, pixelsize, wcshdr, bunit, defunct_mask

    Large keys **not** stored (access via ``tile_iterator()``):
        spec, error

    Parameters
    ----------
    fits_path : str
        Path to the FITS data cube.
    data_ext : int
        FITS extension index for the flux data.
    error_ext : int or None
        FITS extension for the variance/error data, or None if absent.
    wave_start, wave_end : int
        Row indices (along NAXIS3) for the wavelength trim window.
    shape_yx : (int, int)
        Spatial dimensions of the cube (ny, nx).
    extinction_curve : np.ndarray or None
        Multiplicative Galactic extinction correction curve (shape: nwave).
        Applied on-the-fly to tiles (same as dividing spec by curve).
    der_snr_noise : np.ndarray or None
        Per-spaxel noise from DER_SNR algorithm (used when error_ext is None).
    **small_arrays
        All the small dict entries: x, y, wave, signal, noise, snr,
        pixelsize, wcshdr, bunit, defunct_mask.
    """

    _BLOCKED_KEYS = frozenset(("spec", "error"))

    def __init__(
        self,
        fits_path,
        data_ext,
        error_ext,
        wave_start,
        wave_end,
        shape_yx,
        extinction_curve=None,
        der_snr_noise=None,
        **small_arrays,
    ):
        super().__init__(**small_arrays)
        self._fits_path = fits_path
        self._data_ext = data_ext
        self._error_ext = error_ext
        self._wave_start = wave_start
        self._wave_end = wave_end
        self._shape_yx = shape_yx  # (ny, nx)
        self._extinction_curve = extinction_curve
        self._der_snr_noise = der_snr_noise
        # Optional DEBUG restriction: (y_start, y_end) tuple
        self._debug_y_range = None

    # ------------------------------------------------------------------ #
    #  Dict overrides: block access to large keys                         #
    # ------------------------------------------------------------------ #
    def __getitem__(self, key):
        if key in self._BLOCKED_KEYS:
            raise MemoryError(
                f"cube['{key}'] is not loaded in out-of-core mode. "
                f"Use cube.tile_iterator() to stream spatial tiles, "
                f"or set READ_DATA.OOC = False to fall back to in-memory mode."
            )
        return super().__getitem__(key)

    def __contains__(self, key):
        if key in self._BLOCKED_KEYS:
            return False
        return super().__contains__(key)

    # ------------------------------------------------------------------ #
    #  Core streaming interface                                           #
    # ------------------------------------------------------------------ #
    @property
    def nspaxels(self):
        """Number of active spaxels (respects DEBUG restriction)."""
        return len(self["x"])

    @property
    def nwave(self):
        """Number of wavelength pixels in the trimmed range."""
        return self._wave_end - self._wave_start

    def tile_iterator(self, target_spaxels=10_000):
        """Yield ``(indices, spec_tile, error_tile)`` for spatial tiles.

        Each tile contains all wavelengths for a contiguous block of
        spatial pixels (rows of the spatial grid), yielding at most
        ``target_spaxels`` per tile.

        Parameters
        ----------
        target_spaxels : int
            Approximate number of spaxels per tile.  The actual count
            per tile is rounded to whole spatial rows.

        Yields
        ------
        indices : np.ndarray, shape (n_tile,)
            Flat spaxel indices (into the *active* spaxel set).
        spec_tile : np.ndarray, shape (nwave, n_tile)
            Flux spectra for the tile, extinction-corrected.
        error_tile : np.ndarray, shape (nwave, n_tile)
            Error (variance) spectra for the tile, extinction-corrected.
        """
        if fitsio is None:
            raise ImportError("fitsio is required for out-of-core cube streaming")

        ny, nx = self._shape_yx

        # Determine y-range (full grid or DEBUG row)
        if self._debug_y_range is not None:
            y_lo, y_hi = self._debug_y_range
        else:
            y_lo, y_hi = 0, ny

        rows_per_tile = max(1, target_spaxels // nx)

        # Prepare extinction divisor (column vector for broadcasting)
        ext_div = None
        if self._extinction_curve is not None:
            ext_div = self._extinction_curve.reshape(-1, 1)

        flat_offset = 0  # running index into the *active* spaxel array

        with fitsio.FITS(self._fits_path) as f:
            for y_start in range(y_lo, y_hi, rows_per_tile):
                y_end = min(y_start + rows_per_tile, y_hi)
                n_rows = y_end - y_start
                n_tile = n_rows * nx

                # Read flux tile: shape (nwave, n_rows, nx) -> (nwave, n_tile)
                data_3d = f[self._data_ext][
                    self._wave_start : self._wave_end, y_start:y_end, :
                ]
                spec_tile = np.asarray(data_3d, dtype=np.float64).reshape(
                    self.nwave, n_tile
                )

                # Read or synthesise error tile
                if self._error_ext is not None:
                    err_3d = f[self._error_ext][
                        self._wave_start : self._wave_end, y_start:y_end, :
                    ]
                    error_tile = np.asarray(err_3d, dtype=np.float64).reshape(
                        self.nwave, n_tile
                    )
                elif self._der_snr_noise is not None:
                    # Constant per-spaxel noise, broadcast to full tile
                    noise_slice = self._der_snr_noise[flat_offset : flat_offset + n_tile]
                    error_tile = np.broadcast_to(
                        noise_slice.reshape(1, -1), (self.nwave, n_tile)
                    ).copy()
                else:
                    error_tile = np.zeros((self.nwave, n_tile), dtype=np.float64)

                # Apply Galactic extinction correction (divide in-place)
                if ext_div is not None:
                    np.divide(spec_tile, ext_div, out=spec_tile)
                    np.divide(error_tile, ext_div, out=error_tile)

                indices = np.arange(flat_offset, flat_offset + n_tile)
                flat_offset += n_tile

                yield indices, spec_tile, error_tile

    # ------------------------------------------------------------------ #
    #  DEBUG mode                                                         #
    # ------------------------------------------------------------------ #
    def apply_debug(self, ny, nx):
        """Restrict to one central row of spaxels (DEBUG mode).

        Slices the small arrays to the central y-row and sets the internal
        y-range so ``tile_iterator`` only yields data for that row.
        """
        mid_y = ny // 2
        start = mid_y * nx
        end = (mid_y + 1) * nx

        for key in ("x", "y", "snr", "signal", "noise", "defunct_mask"):
            if key in dict.keys(self):
                dict.__setitem__(self, key, dict.__getitem__(self, key)[start:end])

        self._debug_y_range = (mid_y, mid_y + 1)
        logging.info(
            "DEBUG mode: restricted to y-row %d (%d spaxels)", mid_y, nx
        )
