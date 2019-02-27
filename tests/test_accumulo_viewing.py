from .base_test_class import BaseTestClass
BaseTestClass.setup_local_spark()

from unittest import TestCase,skip

import requests
import os
from shapely.geometry import Polygon
from openeogeotrellis.GeotrellisCatalogImageCollection import GeotrellisCatalogImageCollection


class Test(TestCase):

    @skip
    def test_wmts_from_accumulo(self):
        imagecollection = GeotrellisCatalogImageCollection(image_collection_id="CGS_SENTINEL2_RADIOMETRY_V102")
        service = imagecollection.tiled_viewing_service(type="WMTS")
        print(service)
        print(service['url'])