import logging

import numpy as np
import scipy.signal

from openeo import Connection, DataCube

_log = logging.getLogger(__name__)


def makekernel(size: int) -> np.ndarray:
    # TODO move to kernel building module in openeo client. https://github.com/Open-EO/openeo-python-client/issues/106
    assert size % 2 == 1
    kernel_vect = scipy.signal.windows.gaussian(size, std=size / 6.0, sym=True)
    kernel = np.outer(kernel_vect, kernel_vect)
    kernel = kernel / kernel.sum()
    return kernel


def _add_band_dimension_workaround(cube: DataCube) -> DataCube:
    # TODO: avoid "add_dimension" workaround #EP-3402 #EP-3404
    _log.warning("Doing 'add_dimension' workaround")
    return cube.add_dimension("bands", "mask", type="bands").band("mask")


def create_simple_mask(
    connection: Connection,
    s2_collection_id: str,
    size: int = 5,
    class_to_mask=3,
    band_math_workaround=False,
):
    band = "SCENECLASSIFICATION_20M"
    # TODO: drop "bands" argument after https://github.com/Open-EO/openeo-python-driver/issues/38
    classification = connection.load_collection(s2_collection_id, bands=[band]).band(band)
    base_mask = classification == class_to_mask
    fuzzy = base_mask.apply_kernel(makekernel(size))
    if band_math_workaround:
        # TODO: avoid "add_dimension" workaround #EP-3402 #EP-3404
        fuzzy = _add_band_dimension_workaround(fuzzy)
    mask = fuzzy > 0.1
    return mask


def create_advanced_mask(
    start: str,
    end: str,
    connection: Connection,
    s2_collection_id: str,
    band_math_workaround=False,
):
    band = "SCENECLASSIFICATION_20M"
    # TODO: drop "bands" argument after https://github.com/Open-EO/openeo-python-driver/issues/38
    classification = connection.load_collection(s2_collection_id, bands=[band]).band(band)

    # in openEO, 1 means mask (remove pixel) 0 means keep pixel

    # keep useful pixels, so set to 1 (remove) if smaller than threshold
    first_mask = ~ ((classification == 2) | (classification == 4) | (classification == 5) | (classification == 6) | (classification == 7))
    first_mask = first_mask.apply_kernel(makekernel(17))
    # remove pixels smaller than threshold, so pixels with a lot of neighbouring good pixels are retained?
    if band_math_workaround:
        first_mask = _add_band_dimension_workaround(first_mask)
    first_mask = first_mask > 0.057

    # remove cloud pixels so set to 1 (remove) if larger than threshold
    second_mask = (classification == 3) | (classification == 8) | (classification == 9) | (classification == 10) | (classification == 11)
    second_mask = second_mask.apply_kernel(makekernel(201))
    if band_math_workaround:
        second_mask = _add_band_dimension_workaround(second_mask)
    second_mask = second_mask > 0.025

    # TODO: the use of filter_temporal is a trick to make cube merging work, needs to be fixed in openeo client
    return first_mask.filter_temporal(start, end) | second_mask.filter_temporal(start, end)
