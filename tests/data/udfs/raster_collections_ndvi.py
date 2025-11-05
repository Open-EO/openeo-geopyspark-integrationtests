import xarray

from openeo.udf import XarrayDataCube


def apply_datacube(array: xarray.DataArray, context: dict) -> xarray.DataArray:
    """Compute the NDVI based on sentinel2 tiles"""
    red = array.sel(bands="B04")
    nir = array.sel(bands="B08")
    ndvi = (nir - red) / (nir + red)

    return ndvi
