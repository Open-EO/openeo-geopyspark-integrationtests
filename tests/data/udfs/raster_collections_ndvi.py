# -*- coding: utf-8 -*-
# Uncomment the import only for coding support
#import numpy
#import pandas
#import torch
#import torchvision
#import tensorflow
#import tensorboard
from openeo_udf.api.base import RasterCollectionTile

#from openeo_udf.api.raster_collection_tile import RasterCollectionTile
#from openeo_udf.api.udf_data import UdfData

__license__ = "Apache License, Version 2.0"
__author__ = "Soeren Gebbert"
__copyright__ = "Copyright 2018, Soeren Gebbert"
__maintainer__ = "Soeren Gebbert"
__email__ = "soerengebbert@googlemail.com"


def rct_ndvi(udf_data):
    """Compute the NDVI based on sentinel2 tiles

    Tiles with ids "red" and "nir" are required. The NDVI computation will be applied
    to all time stamped 2D raster tiles that have equal time stamps.

    Args:
        udf_data (UdfData): The UDF data object that contains raster and vector tiles

    Returns:
        This function will not return anything, the UdfData object "udf_data" must be used to store the resulting
        data.

    """
    red = None
    nir = None

    ids = set()

    # Iterate over each tile
    for tile in udf_data.raster_collection_tiles:
        if "4" in tile.id.lower():
            red = tile
        if "8" in tile.id.lower():
            nir = tile
        ids.add(tile.id)
    if red is None:
        raise Exception("B04 raster collection tile is missing in input, found: " + str(ids))
    if nir is None:
        raise Exception("B08 raster collection tile is missing in input, found: " + str(ids))
    if red.start_times is None or red.start_times.tolist() == nir.start_times.tolist():
        # Compute the NDVI
        ndvi = (nir.data - red.data) / (nir.data + red.data)
        # Create the new raster collection tile
        rct = RasterCollectionTile(id="ndvi", extent=red.extent, data=ndvi,
                                   start_times=red.start_times, end_times=red.end_times)
        # Insert the new tiles as list of raster collection tiles in the input object. The new tiles will
        # replace the original input tiles.
        udf_data.set_raster_collection_tiles([rct,])
    else:
        raise Exception("Time stamps are not equal")


# This function call is the entry point for the UDF.
# The caller will provide all required data in the **data** object.
rct_ndvi(data)