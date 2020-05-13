import os
from pathlib import Path
from unittest import TestCase

import openeo
from .conftest import get_openeo_base_url


def get_test_resource(relative_path):
    dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return dir / relative_path


def load_udf(relative_path):
    with open(str(get_test_resource(relative_path)), 'r+') as f:
        return f.read()


class Test(TestCase):

    _rest_base = get_openeo_base_url()

    def test_reduce_time(self):
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

        connection = openeo.connect(self._rest_base)

        image_collection = connection.imagecollection(product) \
            .date_range_filter(start_date=time["start"], end_date=time["end"]) \
            .bbox_filter(left=bbox["left"], right=bbox["right"], bottom=bbox["bottom"], top=bbox["top"],srs=bbox["srs"])

        udf_code = load_udf("udf_temporal_slope.py")
        print(udf_code)

        trend = image_collection.reduce_tiles_over_time(udf_code,runtime="Python",version="latest")

        trend.download("/tmp/openeo-trend-udf.geotiff", format =  out_format)
