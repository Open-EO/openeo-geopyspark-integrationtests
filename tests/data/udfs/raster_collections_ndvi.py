import xarray

from openeo.udf import XarrayDataCube


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    """Compute the NDVI based on sentinel2 tiles"""
    array: xarray.DataArray = cube.get_array()
    red = array.sel(bands="TOC-B04_10M")
    nir = array.sel(bands="TOC-B08_10M")
    ndvi = (nir - red) / (nir + red)

    import os
    statinfo = os.stat("/data/users/Public")
    print(statinfo)
    return XarrayDataCube(ndvi)
