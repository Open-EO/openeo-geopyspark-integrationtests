import xarray

from openeo_udf.api.datacube import DataCube


def apply_datacube(cube: DataCube, context: dict) -> DataCube:
    """Compute the NDVI based on sentinel2 tiles

    Tiles with ids "red" and "nir" are required. The NDVI computation will be applied
    to all time stamped 2D raster tiles that have equal time stamps.

    """
    array: xarray.DataArray = cube.get_array()
    red = array.sel(bands="TOC-B04_10M")
    nir = array.sel(bands="TOC-B08_10M")
    ndvi = (nir - red) / (nir + red)
    return DataCube(ndvi)
