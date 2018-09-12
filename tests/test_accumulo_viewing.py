from tests.base_test_class import BaseTestClass
from unittest import TestCase,skip
from openeo.rest import rest_session
import requests
import os
from shapely.geometry import Polygon
from openeogeotrellis.GeotrellisCatalogImageCollection import GeotrellisCatalogImageCollection


class Test(BaseTestClass):

    @skip
    def test_wmts_from_accumulo(self):
        imagecollection = GeotrellisCatalogImageCollection(image_collection_id="CGS_SENTINEL2_RADIOMETRY_V102")
        service = imagecollection.tiled_viewing_service(type="WMTS")
        print(service)
        print(service['url'])