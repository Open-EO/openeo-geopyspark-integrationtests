import imghdr
import json
import tempfile
import time
from pathlib import Path
from typing import Callable, Union
from unittest import TestCase, skip

import numpy as np
import openeo
import pytest
import rasterio
import requests
import schema
from numpy.testing import assert_array_equal
from openeo.rest.datacube import DataCube
from openeo.rest.imagecollectionclient import ImageCollectionClient
from openeo.rest.job import RESTJob
from shapely.geometry import shape, Polygon

from .cloudmask import create_advanced_mask, create_simple_mask
from .conftest import get_openeo_base_url
from .data import get_path, read_data


def _dump_process_graph(cube: Union[DataCube, ImageCollectionClient], tmp_path: Path, name="process_graph.json"):
    """Dump a cube's process graph as json to a temp file"""
    with (tmp_path / name).open("w") as fp:
        json.dump(cube.graph, fp, indent=2)


def _parse_bboxfinder_com(url: str) -> dict:
    """Parse a bboxfinder.com URL to bbox dict"""
    coords = [float(x) for x in url.split('#')[-1].split(",")]
    return {"south": coords[0], "west": coords[1], "north": coords[2], "east": coords[3], "crs": "EPSG:4326"}


BBOX_MOL = _parse_bboxfinder_com("http://bboxfinder.com/#51.21,5.071,51.23,5.1028")
BBOX_GENT = _parse_bboxfinder_com("http://bboxfinder.com/#51.03,3.7,51.05,3.75")
BBOX_NIEUWPOORT = _parse_bboxfinder_com("http://bboxfinder.com/#51.05,2.60,51.20,2.90")


def test_health(connection):
    r = connection.get("/health")
    assert r.status_code == 200


def test_collections(connection):
    image_collections = connection.list_collections()
    product_ids = [entry.get("id") for entry in image_collections]
    assert "PROBAV_L3_S10_TOC_NDVI_333M" in product_ids


def test_terrascope_download_latlon(connection, tmp_path):
    s2_fapar = (
        connection.load_collection("SENTINEL2_NDVI_TERRASCOPE")
            # bounding box: http://bboxfinder.com/#51.197400,5.027000,51.221300,5.043800
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
            .filter_bbox(west=5.027, east=5.0438, south=51.1974, north=51.2213, crs="EPSG:4326")
    )
    out_file = tmp_path / "s2_fapar_latlon.geotiff"
    s2_fapar.download(out_file, format="GTIFF")
    assert out_file.exists()


def test_terrascope_download_webmerc(connection, tmp_path):
    s2_fapar = (
        connection.load_collection("SENTINEL2_NDVI_TERRASCOPE")
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
            .filter_bbox(west=561864.7084, east=568853, south=6657846, north=6661080, crs="EPSG:3857")
    )
    out_file = tmp_path / "/tmp/s2_fapar_webmerc.geotiff"
    s2_fapar.download(out_file, format="geotiff")
    assert out_file.exists()


def test_aggregate_spatial_polygon(connection):
    polygon = Polygon(shell=[
        (7.022705078125007, 51.75432477678571),
        (7.659912109375007, 51.74333844866071),
        (7.659912109375007, 51.29289899553571),
        (7.044677734375007, 51.31487165178571),
        (7.022705078125007, 51.75432477678571)
    ])
    timeseries = (
        connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal(start_date="2017-11-01", end_date="2017-11-21")
            .polygonal_mean_timeseries(polygon)
            .execute()
    )
    print(timeseries)
    expected_dates = ["2017-11-01T00:00:00", "2017-11-11T00:00:00", "2017-11-21T00:00:00"]
    assert sorted(timeseries.keys()) == sorted(expected_dates)
    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(timeseries)


def test_histogram_timeseries(connection):
    probav = (
        connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_bbox(5, 6, 52, 51, 'EPSG:4326')
            .filter_temporal(['2017-11-21', '2017-12-21'])
    )
    polygon = shape({"type": "Polygon", "coordinates": [[
        [5.0761587693484875, 51.21222494794898],
        [5.166854684377381, 51.21222494794898],
        [5.166854684377381, 51.268936260927404],
        [5.0761587693484875, 51.268936260927404],
        [5.0761587693484875, 51.21222494794898]
    ]]})
    timeseries = probav.polygonal_histogram_timeseries(polygon=polygon).execute()
    print(timeseries)

    expected_schema = schema.Schema({str: [[{str: int}]]})
    assert expected_schema.validate(timeseries)

    for date, histograms in timeseries.items():
        assert len(histograms) == 1
        assert len(histograms[0]) == 1
        assert len(histograms[0][0]) > 10


def test_ndvi_udf_reduce_bands_udf(connection, tmp_path):
    udf_code = read_data("udfs/raster_collections_ndvi.py")

    cube = (
        connection.load_collection('CGS_SENTINEL2_RADIOMETRY_V102_001')
            .date_range_filter(start_date="2017-10-15", end_date="2017-10-15")
            .bbox_filter(left=761104, right=763281, bottom=6543830, top=6544655, srs="EPSG:3857")
    )
    # cube.download(tmp_path / "cube.tiff", format="GTIFF")
    res = cube.reduce_bands_udf(udf_code)

    out_file = tmp_path / "ndvi-udf.tiff"
    res.download(out_file, format="geotiff")
    assert_geotiff_basics(out_file, expected_band_count=1)
    with rasterio.open(out_file) as ds:
        ndvi = ds.read(1)
        assert 0.35 < ndvi.min(axis=None)
        assert ndvi.max(axis=None) < 0.95


def test_ndvi_band_math(connection, tmp_path, api_version):
    if api_version.at_least("1.0.0"):
        pytest.skip("1.0-style add/subtract/divide not supported yet. see https://github.com/Open-EO/openeo-geotrellis-extensions/pull/4")
    # http://bboxfinder.com/#50.560007,6.8371137,50.5647147,6.8566699
    bbox = {
        "west": 6.8371137, "south": 50.560007,
        "east": 6.8566699, "north": 50.5647147,
        "crs": "EPSG:4326"
    }
    cube = (
        connection.load_collection("CGS_SENTINEL2_RADIOMETRY_V102_001")
            .filter_temporal("2017-10-10", "2017-10-30")
            .filter_bbox(**bbox)
    )
    # cube.download(tmp_path / "cube.tiff", format="GTIFF")

    red = cube.band("4")
    nir = cube.band("8")
    ndvi = (red - nir) / (red + nir)

    out_file = tmp_path / "ndvi.tiff"
    ndvi.download(out_file, format="GTIFF")
    assert_geotiff_basics(out_file, expected_band_count=1)
    with rasterio.open(out_file) as ds:
        x = ds.read(1)
        assert -0.9 < np.nanmin(x, axis=None)
        assert np.nanmax(x, axis=None) < -0.1
        assert np.isnan(x).sum(axis=None) > 10000


def test_sync_cog(connection, tmp_path):
    out_file = tmp_path / "cog.tiff"
    (
        connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal("2017-11-21", "2017-11-21")
            .filter_bbox(west=0, south=50, east=5, north=55, crs='EPSG:4326')
            .download(out_file, format="GTIFF", options={"tiled": True})
    )
    assert_geotiff_basics(out_file)
    assert_cog(out_file)



class Test(TestCase):

    _rest_base = get_openeo_base_url()


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

        job_id = job_url.split('/')[-1]
        status = self.poll_job_status(
            RESTJob(job_id=job_id, connection=openeo.connect(self._rest_base)),
            until=lambda s: s in ['canceled', 'finished', 'error']
        )
        assert status == "finished"

        get_job_results = requests.get(job_url + "/results")
        self.assertEqual(200, get_job_results.status_code)

        job_result_links = get_job_results.json()["links"]
        self.assertEqual(1, len(job_result_links))

        get_job_result = requests.get(job_result_links[0])
        self.assertEqual(200, get_job_result.status_code)

        zonal_statistics = get_job_result.json()

        self.assertEqual(3, len(zonal_statistics))


    def poll_job_status(
            self, job: RESTJob, until: Callable = lambda s: s == "finished", max_polls: int = 100,
            sleep_max: int = 120) -> str:
        """Helper to poll the status of a job until some condition is reached."""
        sleep = 10.0
        start = time.time()
        for p in range(max_polls):
            elapsed = time.time() - start
            try:
                status = job.describe_job()['status']
            except requests.ConnectionError as e:
                print("job {j} status poll {p} ({e:.2f}s) failed: {x}".format(j=job.job_id, p=p, e=elapsed, x=e))
            else:
                print("job {j} status poll {p} ({e:.2f}s): {s}".format(j=job.job_id, p=p, e=elapsed, s=status))
                if until(status):
                    return status
            time.sleep(sleep)
            # Adapt sleep time
            sleep = min(sleep * 1.1, sleep_max)

        raise RuntimeError("reached max polls {m}".format(m=max_polls))

    @pytest.mark.timeout(600)
    def test_batch_cog(self):
        session = openeo.connect(self._rest_base).authenticate_basic(username='dummy', password='dummy123')

        job = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .date_range_filter(start_date="2017-11-01", end_date="2017-11-21") \
            .bbox_filter(left=0, right=5, bottom=50, top=55, srs='EPSG:4326') \
            .send_job('GTiff', tiled=True,job_options={
            "driver-memory":"3G",
            "driver-cores": "2",
            "executor-memory": "1G",
            "executor-memoryOverhead": "1G",
            "executor-cores": "1"
        })

        job.start_job()

        status = self.poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
        assert status == "finished"

        with tempfile.TemporaryDirectory() as tempdir:
            output_file = "%s/%s.geotiff" % (tempdir, job.job_id)

            job.download_result(output_file)
            self._assert_geotiff(output_file)

        this_job = [user_job for user_job in session.list_jobs() if user_job['id'] == job.job_id][0]
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
            .send_job(out_format="GTIFF",job_options={
            "driver-memory":"2G",
            "driver-cores": "2",
            "executor-memory": "1G",
            "executor-memoryOverhead": "1G",
            "executor-cores": "1"
        })

        job.start_job()

        # await job running
        status = self.poll_job_status(
            job, until=lambda s: s in ['running', 'canceled', 'finished', 'error'],
            sleep_max=15, max_polls=300
        )
        assert status == "running"

        # cancel it
        job.stop_job()
        print("stopped job")

        # await job canceled
        status = self.poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
        assert status == "canceled"

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


def test_load_collection_from_disk(connection, tmp_path):
    fapar = connection.load_disk_collection(
        format='GTiff',
        glob_pattern='/data/MTDA/CGS_S2/CGS_S2_FAPAR/2019/04/24/*/*/10M/*_FAPAR_10M_V102.tif',
        options={
            'date_regex': r".*\/S2._(\d{4})(\d{2})(\d{2})T.*"
        }
    )
    date = "2019-04-24"
    fapar = fapar.filter_bbox(**BBOX_NIEUWPOORT).filter_temporal(date, date)

    output_file = tmp_path / "fapar_from_disk.tiff"
    fapar.download(output_file, format="GTiff")
    assert_geotiff_basics(output_file)


def assert_geotiff_basics(output_tiff: str, expected_band_count=1, min_width=64, min_height=64):
    """Basic checks that a file is a readable GeoTIFF file"""
    assert imghdr.what(output_tiff) == 'tiff'
    with rasterio.open(output_tiff) as dataset:
        assert dataset.count == expected_band_count
        assert dataset.width > min_width
        assert dataset.height > min_height


def assert_cog(output_tiff: str):
    # FIXME: check if actually a COG
    pass


def test_mask_polygon(connection, api_version, tmp_path):
    bbox = {"west": 7.0, "south": 51.28, "east": 7.7, "north": 51.8, "crs": "EPSG:4326"}
    date = "2017-11-01"
    collection_id = 'PROBAV_L3_S10_TOC_NDVI_333M'
    cube = connection.load_collection(collection_id).filter_bbox(**bbox).filter_temporal(date, date)
    polygon = Polygon(shell=[
        (7.022705078125007, 51.75432477678571),
        (7.659912109375007, 51.74333844866071),
        (7.659912109375007, 51.29289899553571),
        (7.044677734375007, 51.31487165178571),
        (7.022705078125007, 51.75432477678571)
    ])
    if api_version >= "1.0.0":
        masked = cube.mask_polygon(polygon)
    else:
        masked = cube.mask(polygon=polygon)

    output_tiff = tmp_path / "masked.tiff"
    masked.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)


def test_mask_out_all_data(connection, api_version, tmp_path):
    bbox = {"west": 5, "south": 51, "east": 6, "north": 52, "crs": "EPSG:4326"}
    date = "2017-12-21"
    collection_id = 'PROBAV_L3_S10_TOC_NDVI_333M'
    probav = connection.load_collection(collection_id).filter_temporal(date, date).filter_bbox(**bbox)
    opaque_mask = probav.band("ndvi") != 255  # all ones
    _dump_process_graph(opaque_mask, tmp_path=tmp_path, name="opaque_mask.json")
    if api_version >= "1.0.0":
        probav_masked = probav.mask(mask=opaque_mask)
    else:
        probav_masked = probav.mask(rastermask=opaque_mask)

    probav_path = tmp_path / "probav.tiff"
    probav.download(probav_path, format='GTiff')
    _dump_process_graph(probav_masked, tmp_path=tmp_path, name="probav_masked.json")
    masked_path = tmp_path / "probav_masked.tiff"
    probav_masked.download(masked_path, format='GTiff')

    assert_geotiff_basics(probav_path, expected_band_count=1)
    assert_geotiff_basics(masked_path, expected_band_count=1)
    with rasterio.open(probav_path) as probav_ds, rasterio.open(masked_path) as masked_ds:
        assert probav_ds.width == probav_ds.height
        assert np.all(probav_ds.read(1) != 255)

        assert (probav_ds.width, probav_ds.height) == (masked_ds.width, masked_ds.height)
        assert np.all(np.isnan(masked_ds.read(1)))


def test_fuzzy_mask(connection, tmp_path):
    date = "2019-04-26"
    mask = create_simple_mask(connection, band_math_workaround=True)
    mask = mask.filter_bbox(**BBOX_GENT).filter_temporal(date, date)
    _dump_process_graph(mask, tmp_path)
    output_tiff = tmp_path / "mask.tiff"
    mask.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)


def test_simple_cloud_masking(connection, api_version, tmp_path):
    date = "2019-04-26"
    mask = create_simple_mask(connection, band_math_workaround=True)
    mask = mask.filter_bbox(**BBOX_GENT).filter_temporal(date, date)
    # mask.download(tmp_path / "mask.tiff", format='GTIFF')
    s2_radiometry = (
        connection.load_collection("CGS_SENTINEL2_RADIOMETRY_V102_001", bands=["2", "3", "4"])
            .filter_bbox(**BBOX_GENT).filter_temporal(date, date)
    )
    # s2_radiometry.download(tmp_path / "s2.tiff", format="GTIFF")
    if api_version >= "1.0.0":
        masked = s2_radiometry.mask(mask=mask)
    else:
        masked = s2_radiometry.mask(rastermask=mask)

    _dump_process_graph(masked, tmp_path)
    output_tiff = tmp_path / "masked_result.tiff"
    masked.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=3)


def test_advanced_cloud_masking(connection, api_version, tmp_path):
    if api_version.at_least("1.0.0"):
        pytest.skip("does not work in 1.0.0 yet")
    # Retie
    bbox = {"west": 4.996033, "south": 51.258922, "east": 5.091603, "north": 51.282696, "crs": "EPSG:4326"}
    date = "2018-08-14"
    mask = create_advanced_mask(start=date, end=date, connection=connection, band_math_workaround=True)
    mask = mask.filter_bbox(**bbox)
    # mask.download(tmp_path / "mask.tiff", format='GTIFF')
    s2_radiometry = (
        connection.load_collection("CGS_SENTINEL2_RADIOMETRY_V102_001", bands=["2", "3", "4"])
            .filter_bbox(**bbox).filter_temporal(date, date)
    )
    # s2_radiometry.download(tmp_path / "s2.tiff", format="GTIFF")

    if api_version >= "1.0.0":
        masked = s2_radiometry.mask(mask=mask)
    else:
        masked = s2_radiometry.mask(rastermask=mask)

    _dump_process_graph(masked, tmp_path)
    out_file = tmp_path / "masked_result.tiff"
    masked.download(out_file, format='GTIFF')

    with rasterio.open(out_file) as result_ds:
        with rasterio.open(get_path("reference/cloud_masked.tiff")) as ref_ds:
            assert_array_equal(ref_ds.read(), result_ds.read())
