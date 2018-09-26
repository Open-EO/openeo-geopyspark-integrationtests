from unittest import TestCase,skip
from openeo.rest import rest_session
import requests
import os
from shapely.geometry import Polygon
import time
import pytest
import tempfile
import imghdr


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

    def test_ndvi_udf(self):
        import os,openeo_udf
        import openeo_udf.functions
        dir = os.path.dirname(openeo_udf.functions.__file__)
        file_name = os.path.join(dir, "raster_collections_ndvi.py")
        with open(file_name, "r")  as f:
            udf_code = f.read()
            session = rest_session.session(userid=None, endpoint=self._rest_base)

            image_collection = session \
                .imagecollection('CGS_SENTINEL2_RADIOMETRY_V101') \
                .date_range_filter(start_date="2017-10-15", end_date="2017-10-15") \
                .bbox_filter(left=761104,right=763281,bottom=6543830,top=6544655,srs="EPSG:3857") \
                .apply_tiles(udf_code) \
                .download("/tmp/openeo-ndvi-udf.geotiff","geotiff")

    @skip
    @pytest.mark.timeout(600)
    def test_batch_job(self):
        create_batch_job = requests.post(self._rest_base + "/jobs", json={
            "process_graph": {
                "process_id": "zonal_statistics",
                "args": {
                    "imagery": {
                        "process_id": "filter_daterange",
                        "args": {
                            "imagery": {
                                "product_id": "PROBAV_L3_S10_TOC_NDVI_333M"
                            },
                            "from": "2017-11-01",
                            "to": "2017-11-21"
                        }
                    },
                    "regions": {
                        "type": "Polygon",
                        "coordinates": [[
                            [7.022705078125007, 51.75432477678571], [7.659912109375007, 51.74333844866071],
                            [7.659912109375007, 51.29289899553571], [7.044677734375007, 51.31487165178571],
                            [7.022705078125007, 51.75432477678571]
                        ]]
                    }
                }
            },
            "output": {}
        })

        self.assertEqual(201, create_batch_job.status_code)

        job_url = create_batch_job.headers['Location']
        self.assertIsNotNone(job_url)

        queue_job = requests.post(job_url + "/results")
        self.assertEqual(202, queue_job.status_code)

        job_in_progress = True
        while job_in_progress:
            time.sleep(60)

            get_job_info = requests.get(job_url)
            job_info = get_job_info.json()

            job_in_progress = job_info['status'] not in ['canceled', 'finished', 'error']

        self.assertEqual('finished', job_info['status'])

        get_job_results = requests.get(job_url + "/results")
        self.assertEqual(200, get_job_results.status_code)

        job_result_links = get_job_results.json()["links"]
        self.assertEqual(1, len(job_result_links))

        get_job_result = requests.get(job_result_links[0])
        self.assertEqual(200, get_job_result.status_code)

        zonal_statistics = get_job_result.json()

        self.assertEqual(3, len(zonal_statistics))

    def test_sync_cog(self):
        session = rest_session.session(userid=None, endpoint=self._rest_base)

        with tempfile.TemporaryDirectory() as tempdir:
            output_file = "%s/%s.geotiff" % (tempdir, "test_cog")

            session \
                .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
                .date_range_filter(start_date="2017-11-21", end_date="2017-11-21") \
                .bbox_filter(left=0, right=5, bottom=50, top=55, srs='EPSG:4326') \
                .download(output_file, parameters={"tiled": True})

            self.assertEqual("tiff", imghdr.what(output_file))  # FIXME: check if actually a COG

    @pytest.mark.timeout(600)
    def test_batch_client(self):
        session = rest_session.session(userid=None, endpoint=self._rest_base)

        job = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .date_range_filter(start_date="2017-11-01", end_date="2017-11-21") \
            .bbox_filter(left=0, right=5, bottom=50, top=55, srs='EPSG:4326') \
            .send_job('GTiff')

        self.assertEqual(200, job.queue())  # FIXME: HTTP error should be translated to an exception

        in_progress = True
        while in_progress:
            time.sleep(60)

            status = job.status()
            print("job %s has status %s" % (job.job_id, status))
            in_progress = status not in ['canceled', 'finished', 'error']

        self.assertEqual('finished', job.status())

        results = job.results()

        self.assertEqual(1, len(results))

        with tempfile.TemporaryDirectory() as tempdir:
            output_file = "%s/%s.geotiff" % (tempdir, job.job_id)

            results[0].save_as(output_file)
            self.assertEqual("tiff", imghdr.what(output_file))
