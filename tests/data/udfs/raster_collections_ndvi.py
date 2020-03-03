# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
import xarray
from openeo_udf.api.datacube import DataCube
from typing import Dict


def apply_hypercube(cube: DataCube, context: Dict) -> DataCube:
    """Compute the NDVI based on sentinel2 tiles

    Tiles with ids "red" and "nir" are required. The NDVI computation will be applied
    to all time stamped 2D raster tiles that have equal time stamps.

    """
    array: xarray.DataArray = cube.get_array()
    red = array.sel(bands="4")
    nir = array.sel(bands="8")
    ndvi = (nir-red)/(nir+red)
    return DataCube(ndvi)



