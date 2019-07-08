from unittest import TestCase,skip

from pathlib import Path

from openeo.rest import rest_connection as rest_session
from openeo.connection import Connection
import requests
import os
from shapely.geometry import Polygon
import time
import pytest
import tempfile

def get_test_resource(relative_path):
    dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return dir / relative_path


def load_udf(relative_path):
    import json
    with open(get_test_resource(relative_path), 'r+') as f:
        return f.read()

class Test(TestCase):

    _rest_base = "%s/openeo/0.4.0" % os.environ['ENDPOINT']
    #_rest_base = "%s/openeo/0.4.0" % "http://openeo.vgt.vito.be"
    #_rest_base = "%s/openeo/0.4.0" % "http://localhost:5000"

    def test_reduce_time(self):
        import os, openeo_udf
        import openeo_udf.functions

        product = "CGS_SENTINEL2_RADIOMETRY_V102_001"
        bbox = {
            "left": 6.8371137,
            "top": 50.5647147,
            "right": 6.8566699,
            "bottom": 50.560007,
            "srs": "EPSG:4326"
        }
        time = {
            "start": "2017-03-10",
            "end": "2017-03-30"
        }

        out_format = "GTIFF"

        connection = rest_session.connection(self._rest_base)

        image_collection = connection.imagecollection(product) \
            .date_range_filter(start_date=time["start"], end_date=time["end"]) \
            .bbox_filter(left=bbox["left"], right=bbox["right"], bottom=bbox["bottom"], top=bbox["top"],srs=bbox["srs"])

        udf_code = load_udf("udf_temporal_slope.py")
        print(udf_code)

        trend = image_collection.reduce_tiles_over_time(udf_code,runtime="Python",version="latest")

        trend.download("/tmp/openeo-trend-udf.geotiff", format =  out_format)
