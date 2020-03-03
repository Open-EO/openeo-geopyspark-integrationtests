# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
import xarray
from openeo_udf.api.datacube import DataCube
from typing import Dict

__license__ = "Apache License, Version 2.0"
__author__ = "Soeren Gebbert"
__copyright__ = "Copyright 2018, Soeren Gebbert"
__maintainer__ = "Soeren Gebbert"
__email__ = "soerengebbert@googlemail.com"


def apply_hypercube(cube: DataCube, context: Dict) -> DataCube:
    """Reduce the time dimension for each tile and compute min, mean, max and sum for each pixel
    over time.
    Each raster tile in the udf data object will be reduced by time. Minimum, maximum, mean and sum are
    computed for each pixel over time.
    Args:
        udf_data (UdfData): The UDF data object that contains raster and vector tiles
    Returns:
        This function will not return anything, the UdfData object "udf_data" must be used to store the resulting
        data.
    """
    # The list of tiles that were created
    tile_results = []
    array: xarray.DataArray = cube.get_array()

    result = xarray.concat([array.min(dim='t'), array.max(dim='t'), array.sum(dim='t'), array.mean(dim='t')], dim='bands')
    return result


