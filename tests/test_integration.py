from unittest import TestCase
from openeo.rest import rest_session
import requests
import os


class Test(TestCase):

    _rest_base = "%s/openeo" % os.environ['ENDPOINT']

    def test_health(self):
        r = requests.get(self._rest_base + "/health")
        self.assertEqual(200, r.status_code)

    def test_imagecollections(self):
        session = rest_session.session(userid=None, endpoint=self._rest_base)
        image_collections = session.list_collections()

        self.assertTrue(image_collections)
