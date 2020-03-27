from pathlib import Path
from unittest import TestCase,skip
import openeo
import rasterio
import requests
import os
from shapely.geometry import shape, Polygon
import time
import pytest
import tempfile
import imghdr


class Test(TestCase):

    _rest_base = "%s/openeo/0.4.0" % os.environ['ENDPOINT']
    #_rest_base =  "http://openeo.vgt.vito.be/openeo/0.4.0"
    #_rest_base = "%s/openeo/0.4.0" % "http://localhost:8080"

    def test_health(self):
        r = requests.get(self._rest_base + "/health")
        self.assertEqual(200, r.status_code)

    def test_imagecollections(self):
        session = openeo.connect(self._rest_base)
        image_collections = session.list_collections()

        product_ids = [entry.get("id") for entry in image_collections]
        self.assertIn("PROBAV_L3_S10_TOC_NDVI_333M", product_ids)

    def testS2FAPAR_download_latlon(self):
        session = openeo.connect(self._rest_base)
        s2_fapar = session.imagecollection("SENTINEL2_NDVI_TERRASCOPE")
        #bounding box:
        #http://bboxfinder.com/#51.197400,5.027000,51.221300,5.043800
        s2_fapar = s2_fapar.filter_daterange(["2018-08-06T00:00:00Z","2018-08-06T00:00:00Z"]) \
            .filter_bbox(west=5.027, east=5.0438, south=51.1974, north=51.2213, crs="EPSG:4326")

        tempfile = "/tmp/s2_fapar_latlon.geotiff"
        try:
            os.remove(tempfile)
        except OSError:
            pass
        s2_fapar.download(tempfile, format="GTIFF")
        self.assertTrue(os.path.exists(tempfile))

    def testS2FAPAR_download_webmerc(self):
        session = openeo.connect(self._rest_base)
        s2_fapar = session.imagecollection("SENTINEL2_NDVI_TERRASCOPE")
        s2_fapar = s2_fapar.filter_daterange(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"]) \
            .filter_bbox(west=561864.7084, east=568853, south=6657846, north=6661080, crs="EPSG:3857")

        tempfile = "/tmp/s2_fapar_webmerc.geotiff"
        try:
            os.remove(tempfile)
        except OSError:
            pass
        s2_fapar.download(tempfile, format="geotiff")
        self.assertTrue(os.path.exists(tempfile))

    def test_zonal_statistics(self):
        session = openeo.connect(self._rest_base)

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
        assert isinstance(timeseries["2017-11-01T00:00:00"], list)
        assert isinstance(timeseries["2017-11-01T00:00:00"][0], list)
        assert isinstance(timeseries["2017-11-01T00:00:00"][0][0], float)

    def test_ndvi_udf(self):
        import os,openeo_udf
        import openeo_udf.functions
        dir = os.path.dirname(openeo_udf.functions.__file__)
        with (Path(__file__).parent / 'data/udfs/raster_collections_ndvi.py').open('r') as f:
            udf_code = f.read()

        session = openeo.connect(self._rest_base)

        image_collection = session \
            .imagecollection('CGS_SENTINEL2_RADIOMETRY_V102_001') \
            .date_range_filter(start_date="2017-10-15", end_date="2017-10-15") \
            .bbox_filter(left=761104,right=763281,bottom=6543830,top=6544655,srs="EPSG:3857") \
            .apply_tiles(udf_code) \
            .download("/tmp/openeo-ndvi-udf.geotiff",format="geotiff")


    def test_ndwi(self):
        import os,openeo_udf
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
            "start": "2017-10-10",
            "end": "2017-10-30"
        }
        ndvi = {
            "red": "red",
            "nir": "nir"
        }
        stretch = {
            "min": -1,
            "max": 1
        }
        out_format = "GTIFF"

        connection = openeo.connect(self._rest_base)

        image_collection = connection.imagecollection(product) \
            .date_range_filter(start_date=time["start"], end_date=time["end"]) \
            .bbox_filter(left=bbox["left"],right=bbox["right"],bottom=bbox["bottom"],top=bbox["top"],srs=bbox["srs"]) \

        red = image_collection.band("4")
        nir = image_collection.band("8")
        ndwi = (red-nir)/(red+nir)

        ndwi.download("/tmp/openeo-ndwi-udf2.geotiff",format=out_format)

    def test_mask(self):
        session = openeo.connect(self._rest_base)

        bbox = {
            "left": 7.0,
            "top": 51.8,
            "right": 7.7,
            "bottom": 51.28,
            "srs": "EPSG:4326"
        }

        image_collection = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .bbox_filter(left=bbox["left"], right=bbox["right"], bottom=bbox["bottom"], top=bbox["top"],
                         srs=bbox["srs"]) \
            .date_range_filter(start_date="2017-11-01", end_date="2017-11-01")

        polygon = Polygon(shell=[
            (7.022705078125007, 51.75432477678571),
            (7.659912109375007, 51.74333844866071),
            (7.659912109375007, 51.29289899553571),
            (7.044677734375007, 51.31487165178571),
            (7.022705078125007, 51.75432477678571)
        ])

        geotiff = image_collection.mask(polygon).download("out.tiff",format="GTIFF")

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
        session = openeo.connect(self._rest_base)

        with tempfile.TemporaryDirectory() as tempdir:
            output_file = "%s/%s.geotiff" % (tempdir, "test_cog")

            session \
                .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
                .date_range_filter(start_date="2017-11-21", end_date="2017-11-21") \
                .bbox_filter(left=0, right=5, bottom=50, top=55, srs='EPSG:4326') \
                .download(output_file, format="GTIFF", options={"tiled": True})

            self._assert_geotiff(output_file)

    @pytest.mark.timeout(600)
    def test_batch_cog(self):
        session = openeo.connect(self._rest_base).authenticate_basic(username='dummy', password='dummy123')

        job = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .date_range_filter(start_date="2017-11-01", end_date="2017-11-21") \
            .bbox_filter(left=0, right=5, bottom=50, top=55, srs='EPSG:4326') \
            .send_job('GTiff', tiled=True)

        job.start_job()

        in_progress = True
        while in_progress:
            time.sleep(10)

            status = job.describe_job()['status']
            print("job %s has status %s" % (job.job_id, status))
            in_progress = status not in ['canceled', 'finished', 'error']

        self.assertEqual('finished', job.describe_job()['status'])

        with tempfile.TemporaryDirectory() as tempdir:
            output_file = "%s/%s.geotiff" % (tempdir, job.job_id)

            job.download_results(output_file)
            self._assert_geotiff(output_file)

        this_job = [user_job for user_job in session.list_jobs() if user_job['job_id'] == job.job_id][0]
        self.assertEqual('finished', this_job['status'])

    @pytest.mark.timeout(600)
    def test_cancel_batch_job(self):
        session = openeo.connect(self._rest_base).authenticate_basic(username='dummy', password='dummy123')

        job = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .date_range_filter(start_date="2017-01-01", end_date="2017-11-21") \
            .zonal_statistics(regions={
                "type": "Polygon",
                "coordinates": [[
                    [7.022705078125007, 51.75432477678571], [7.659912109375007, 51.74333844866071],
                    [7.659912109375007, 51.29289899553571], [7.044677734375007, 51.31487165178571],
                    [7.022705078125007, 51.75432477678571]
                ]]
            }, func='mean') \
            .send_job(out_format="GTIFF")

        job.start_job()

        # await job running
        job_running = False
        while not job_running:
            time.sleep(3)

            status = job.describe_job()['status']
            print("job status %s" % status)

            if status in ['canceled', 'finished', 'error']:
                self.fail(status)

            job_running = status == 'running'

        # cancel it
        job.stop_job()
        print("stopped job")

        # await job canceled
        job_canceled = False
        while not job_canceled:
            time.sleep(10)

            status = job.describe_job()['status']
            print("job status %s" % status)

            if status in ['finished', 'error']:
                self.fail(status)

            job_canceled = status == 'canceled'

        pass  # success

    def _assert_geotiff(self, file, is_cog=None):
        # FIXME: check if actually a COG
        self.assertEqual("tiff", imghdr.what(file))

    # this test requires proxying to work properly
    @skip
    def test_create_wtms_service(self):
        session = openeo.connect(self._rest_base)

        s2_fapar = session \
            .imagecollection('S2_FAPAR_V102_WEBMERCATOR2') \
            .bbox_filter(left=0, right=5, bottom=50, top=55, srs='EPSG:4326') \
            .date_range_filter(start_date="2019-04-01", end_date="2019-04-01") \

        wmts_url = s2_fapar.tiled_viewing_service(type='WMTS')['url']
        #returns a url like: http://tsviewer-rest-test.vgt.vito.be/openeo/services/6d6f0dbc-d3e6-4606-9768-2130ad96b01d/service/wmts

        time.sleep(5)  # seems to take a while before the service is proxied
        get_capabilities = requests.get(wmts_url + '?REQUEST=getcapabilities').text

        self.assertIn(wmts_url, get_capabilities)  # the capabilities document should advertise the proxied URL

    def test_histogram_timeseries(self):

        session = openeo.connect(self._rest_base)

        probav = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .filter_bbox(5, 6, 52, 51, 'EPSG:4326') \
            .filter_temporal(['2017-11-21', '2017-12-21'])

        histograms = probav.polygonal_histogram_timeseries(polygon=shape({
          "type": "Polygon",
          "coordinates": [
            [
              [
                5.0761587693484875,
                51.21222494794898
              ],
              [
                5.166854684377381,
                51.21222494794898
              ],
              [
                5.166854684377381,
                51.268936260927404
              ],
              [
                5.0761587693484875,
                51.268936260927404
              ],
              [
                5.0761587693484875,
                51.21222494794898
              ]
            ]
          ]
        })).execute()

        single_band_index = 0
        buckets = [bucket for bands in histograms.values() for bucket in bands[single_band_index].items()]

        self.assertIsNotNone(buckets)

    #This test depends on a secret uuid that we can not check in EP-3050
    @skip
    def test_ep3048_sentinel1_udf(self):
        session = openeo.connect(self._rest_base)
        N, E, S, W = (-4.740, -55.695, -4.745, -55.7)
        with (Path(__file__).parent / 'data/udfs/smooth_savitsky_golay.py').open('r') as f:
            udf_code = f.read()

        polygon = Polygon(shell=[[W, N], [E, N], [E, S], [W, S]])
        ts = (
            session.imagecollection("SENTINEL1_GAMMA0_SENTINELHUB")
                .filter_temporal(["2019-05-24T00:00:00Z", "2019-05-30T00:00:00Z"])
                .filter_bbox(north=N, east=E, south=S, west=W, crs="EPSG:4326")
                .band_filter([0])
                .apply_dimension(udf_code, runtime="Python")
                .polygonal_mean_timeseries(polygon)
                .execute()
        )
        assert isinstance(ts, dict)
        assert all(k.startswith('2019-05-') for k in ts.keys())

    def test_mask_out_all_data(self):
        session = openeo.connect(self._rest_base)

        date = "2017-12-21"

        probav = session.imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .filter_bbox(west=5, east=6, south=51, north=52) \
            .filter_temporal(date, date)

        opaque_mask = probav != 255  # all ones

        probav_masked = probav.mask(rastermask=opaque_mask)

        probav.download("probav.geotiff", format='GTiff')
        probav_masked.download("probav_masked.geotiff", format='GTiff')

        import rasterio as rio
        from numpy import all, isnan

        with rio.open("probav.geotiff") as probav_geotiff, \
                rio.open("probav_masked.geotiff") as probav_masked_geotiff:
            self.assertTrue(probav_geotiff.width > 0)
            self.assertEqual(probav_geotiff.width, probav_geotiff.height)
            self.assertEqual(1, probav_geotiff.count)

            probav_is_all_data = all(probav_geotiff.read(1) != 255)
            self.assertTrue(probav_is_all_data)

            self.assertEqual(probav_geotiff.width, probav_masked_geotiff.width)
            self.assertEqual(probav_geotiff.height, probav_masked_geotiff.height)
            self.assertEqual(1, probav_masked_geotiff.count)

            probav_masked_is_all_nodata = all(isnan(probav_masked_geotiff.read(1)))
            self.assertTrue(probav_masked_is_all_nodata)

    def test_load_collection_from_disk(self):
        session = openeo.connect(self._rest_base)

        date = "2019-04-24"

        fapar = session.load_disk_collection(
            format='GTiff',
            glob_pattern='/data/MTDA/CGS_S2/CGS_S2_FAPAR/2019/04/24/*/*/10M/*_FAPAR_10M_V102.tif',
            options={
                'date_regex': r".*\/S2._(\d{4})(\d{2})(\d{2})T.*"
            }
        ) \
            .filter_bbox(west=2.59003, south=51.069, east=2.8949, north=51.2206, crs="EPSG:4326") \
            .filter_temporal(date, date)

        with tempfile.TemporaryDirectory() as tempdir:
            output_file = "%s/%s.geotiff" % (tempdir, "fapar_from_disk")
            fapar.download(output_file, format="GTiff")

            self._assert_geotiff(output_file)

    def test_advanced_cloud_masking(self):
        from .cloudmask import create_mask
        # RETIE
        minx, miny, maxx, maxy = (4.996033, 51.258922, 5.091603, 51.282696)
        date = "2018-08-14"

        session = openeo.connect(self._rest_base)
        mask = create_mask(date, date, session) \
            .filter_bbox(west=minx, east=maxx, north=maxy, south=miny, crs="EPSG:4326")

        s2_radiometry = session.imagecollection("CGS_SENTINEL2_RADIOMETRY_V102_001", bands=["2", "3", "4"]) \
            .filter_bbox(west=minx, east=maxx, north=maxy, south=miny, crs="EPSG:4326").filter_temporal(date, date) \
            .mask(rastermask=mask)

        s2_radiometry.download("masked_result.tiff", format='GTIFF')
        from numpy.testing import assert_array_equal
        with rasterio.open("masked_result.tiff") as result_ds:
            with rasterio.open(Path(__file__).parent /'data'/ 'reference' / 'cloud_masked.tiff') as ref_ds:
                assert_array_equal(ref_ds.read(),result_ds.read())

