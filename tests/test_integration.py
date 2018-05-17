from unittest import TestCase
from openeo.rest import rest_session
import requests
import os
from shapely.geometry import Polygon


class Test(TestCase):

    _rest_base = "%s/openeo" % os.environ['ENDPOINT']

    def test_health(self):
        r = requests.get(self._rest_base + "/health")
        self.assertEqual(200, r.status_code)

    def test_imagecollections(self):
        session = rest_session.session(userid=None, endpoint=self._rest_base)
        image_collections = session.list_collections()

        product_ids = [entry["product_id"] for entry in image_collections]
        self.assertIn("PROBAV_L3_S10_TOC_NDVI_333M", product_ids)

    def test_zonal_statistics(self):
        session = rest_session.session(userid=None, endpoint=self._rest_base)

        image_collection = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .date_range_filter(start_date="2017-11-01", end_date="2017-11-21")

        polygon = Polygon(shell=[
            (7.022705078125007, 51.75432477678571),
            (7.659912109375007, 51.74333844866071),
            (7.659912109375007, 51.29289899553571),
            (7.044677734375007, 51.31487165178571),
            (7.022705078125007, 51.75432477678571)
        ])

        timeseries = image_collection.polygonal_mean_timeseries(polygon).execute()

        expected_dates = ["2017-11-01T00:00:00", "2017-11-11T00:00:00", "2017-11-21T00:00:00"]
        actual_dates = timeseries.keys()

        self.assertEqual(sorted(expected_dates), sorted(actual_dates))
        self.assertTrue(type(timeseries["2017-11-01T00:00:00"][0]) == float)
