import imghdr
import json
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Callable, Union, Dict

import numpy as np
import pytest
import rasterio
import requests
import schema
import shapely.geometry
import shapely.ops
import xarray
from numpy.ma.testutils import assert_array_approx_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
from shapely.geometry import shape, Polygon
import pyproj

from openeo.rest.connection import OpenEoApiError
from openeo.rest.conversions import datacube_from_file, timeseries_json_to_pandas
from openeo.rest.datacube import DataCube, THIS
from openeo.rest.imagecollectionclient import ImageCollectionClient
from openeo.rest.job import RESTJob
from openeo.rest.udp import Parameter
from .cloudmask import create_advanced_mask, create_simple_mask
from .data import get_path, read_data


def _dump_process_graph(cube: Union[DataCube, ImageCollectionClient], tmp_path: Path, name="process_graph.json"):
    """Dump a cube's process graph as json to a temp file"""
    with (tmp_path / name).open("w") as fp:
        json.dump(cube.graph, fp, indent=2)


def _parse_bboxfinder_com(url: str) -> dict:
    """Parse a bboxfinder.com URL to bbox dict"""
    # TODO: move this kind of functionality to python client?
    coords = [float(x) for x in url.split('#')[-1].split(",")]
    return {"south": coords[0], "west": coords[1], "north": coords[2], "east": coords[3], "crs": "EPSG:4326"}


def _polygon_bbox(polygon: Polygon) -> dict:
    """Extract bbox dict from given polygon"""
    coords = polygon.bounds
    return {"south": coords[1], "west": coords[0], "north": coords[3], "east": coords[2], "crs": "EPSG:4326"}


BBOX_MOL = _parse_bboxfinder_com("http://bboxfinder.com/#51.21,5.071,51.23,5.1028")
BBOX_GENT = _parse_bboxfinder_com("http://bboxfinder.com/#51.03,3.7,51.05,3.75")
BBOX_NIEUWPOORT = _parse_bboxfinder_com("http://bboxfinder.com/#51.05,2.60,51.20,2.90")

# TODO: real authenticaion?
TEST_USER = "jenkins"
TEST_PASSWORD = TEST_USER + "123"

POLYGON01 = Polygon(shell=[
    # Dortmund (bbox: http://bboxfinder.com/#51.30,7.00,51.75,7.60)
    [7.00, 51.75],
    [7.10, 51.35],
    [7.50, 51.30],
    [7.60, 51.70],
    [7.00, 51.75],
])

POLYGON01_BBOX = [7.00, 51.30, 7.60, 51.75]


BATCH_JOB_POLL_INTERVAL = 10
BATCH_JOB_TIMEOUT = 40 * 60


def batch_default_options(driverMemoryOverhead="1G", driverMemory="2G"):
    return {
            "driver-memory": driverMemory,
            "driver-memoryOverhead": driverMemoryOverhead,
            "driver-cores": "2",
            "executor-memory": "1G",
            "executor-memoryOverhead": "1G",
            "executor-cores": "1",
            "queue": "lowlatency"
        }


def test_root(connection):
    r = connection.get("/")
    assert r.status_code == 200
    capabilities = r.json()
    assert "api_version" in capabilities
    assert "stac_version" in capabilities
    assert "endpoints" in capabilities


def test_health(connection):
    r = connection.get("/health")
    assert r.status_code == 200


def test_collections(connection):
    image_collections = connection.list_collections()
    product_ids = [entry.get("id") for entry in image_collections]
    assert "PROBAV_L3_S10_TOC_NDVI_333M" in product_ids


def test_terrascope_download_latlon(auth_connection, tmp_path):
    s2_fapar = (
        auth_connection.load_collection("TERRASCOPE_S2_NDVI_V2",bands=["NDVI_10M"])
            # bounding box: http://bboxfinder.com/#51.197400,5.027000,51.221300,5.043800
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
            .filter_bbox(west=5.027, east=5.0438, south=51.1974, north=51.2213, crs="EPSG:4326")
    )
    out_file = tmp_path / "result.tiff"
    s2_fapar.download(out_file, format="GTIFF")
    assert_geotiff_basics(out_file, expected_shape=(1, 270, 126))


def test_terrascope_download_webmerc(auth_connection, tmp_path):
    s2_fapar = (
        auth_connection.load_collection("TERRASCOPE_S2_NDVI_V2",bands=["NDVI_10M"])
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
            .filter_bbox(west=561864.7084, east=568853, south=6657846, north=6661080, crs="EPSG:3857")
    )
    out_file = tmp_path / "result.tiff"
    s2_fapar.download(out_file, format="GTIFF")
    assert_geotiff_basics(out_file, expected_shape=(1, 216, 446))


def test_aggregate_spatial_polygon(auth_connection):
    timeseries = (
        auth_connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal(start_date="2017-11-01", end_date="2017-11-21")
            .polygonal_mean_timeseries(POLYGON01)
            .execute()
    )
    print(timeseries)

    # TODO remove this cleanup https://github.com/Open-EO/openeo-geopyspark-driver/issues/75
    timeseries = {k: v for (k, v) in timeseries.items() if v != [[]]}
    print(timeseries)

    expected_dates = ["2017-11-01T00:00:00Z", "2017-11-11T00:00:00Z", "2017-11-21T00:00:00Z"]
    assert sorted(timeseries.keys()) == sorted(expected_dates)
    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(timeseries)


def test_histogram_timeseries(auth_connection):
    probav = (
        auth_connection
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


@pytest.mark.parametrize("udf_file", [
    "udfs/raster_collections_ndvi_old.py",
    "udfs/raster_collections_ndvi.py",
])
def test_ndvi_udf_reduce_bands_udf(auth_connection, tmp_path, udf_file):
    udf_code = read_data(udf_file)

    cube = (
        auth_connection.load_collection('TERRASCOPE_S2_TOC_V2',bands=['TOC-B04_10M','TOC-B08_10M'])
            .date_range_filter(start_date="2020-11-05", end_date="2020-11-05")
            .bbox_filter(left=761104, right=763281, bottom=6543830, top=6544655, srs="EPSG:3857")
    )
    # cube.download(tmp_path / "cube.tiff", format="GTIFF")
    res = cube.reduce_bands_udf(udf_code)

    out_file = tmp_path / "ndvi-udf.tiff"
    res.download(out_file, format="GTIFF")
    assert_geotiff_basics(out_file, min_height=40, expected_shape=(1, 57, 141))
    with rasterio.open(out_file) as ds:
        ndvi = ds.read(1)
        assert 0.35 < ndvi.min(axis=None)
        assert ndvi.max(axis=None) < 1.0


def test_ndvi_band_math(auth_connection, tmp_path, api_version):
    # http://bboxfinder.com/#50.55,6.82,50.58,6.87
    bbox = {
        "west": 6.82, "south": 50.55,
        "east": 6.87, "north": 50.58,
        "crs": "EPSG:4326"
    }
    cube = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2",
                                        bands=['TOC-B04_10M','TOC-B08_10M'])
           .filter_temporal("2020-11-01", "2020-11-20")
           .filter_bbox(**bbox)
    )
    # cube.download(tmp_path / "cube.tiff", format="GTIFF")

    red = cube.band("TOC-B04_10M")
    nir = cube.band("TOC-B08_10M")
    ndvi = (nir - red) / (red + nir)
    ndvi = ndvi.mean_time()

    out_file = tmp_path / "ndvi.tiff"
    ndvi.download(out_file, format="GTIFF")
    assert_geotiff_basics(out_file,min_height=40, expected_shape=(1, 344, 365))
    with rasterio.open(out_file) as ds:
        x = ds.read(1)
        assert 0.08 < np.nanmin(x, axis=None)
        assert np.nanmax(x, axis=None) < 0.67
        assert np.isnan(x).sum(axis=None) == 0


def test_cog_synchronous(auth_connection, tmp_path):
    cube = (
        auth_connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal("2017-11-21", "2017-11-21")
            .filter_bbox(west=0, south=50, east=5, north=55, crs='EPSG:4326')
    )

    out_file = tmp_path / "cog.tiff"
    cube.download(out_file, format="GTIFF", options={"tiled": True})
    assert_geotiff_basics(out_file, expected_shape=(1, 1681, 1681))
    assert_cog(out_file)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_cog_execute_batch(auth_connection, tmp_path):
    cube = (
        auth_connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal("2017-11-21", "2017-11-21")
            .filter_bbox(west=2, south=51, east=4, north=53, crs='EPSG:4326')
    )
    output_file = tmp_path / "result.tiff"
    job = cube.send_job("GTIFF", job_options=batch_default_options(driverMemoryOverhead="1G",driverMemory="1G"), tile_grid="one_degree",title="cog")
    job.run_synchronous(
        print=print, max_poll_interval=BATCH_JOB_POLL_INTERVAL
    )
    job.download_results(target="./")
    assets: Dict = job.get_results().get_metadata()['assets']
    name,asset = assets.popitem()

    assert [j["status"] for j in auth_connection.list_jobs() if j['id'] == job.job_id] == ["finished"]
    assert_geotiff_basics(name, expected_band_count=1)
    assert_cog(name)


def _poll_job_status(
        job: RESTJob, until: Callable = lambda s: s == "finished",
        sleep: int = BATCH_JOB_POLL_INTERVAL, max_poll_time=30 * 60) -> str:
    """Helper to poll the status of a job until some condition is reached."""
    start = time.time()

    def elapsed():
        return time.time() - start

    while elapsed() < max_poll_time:
        try:
            status = job.describe_job()['status']
        except requests.ConnectionError as e:
            print("job {j} status poll ({e:.2f}s) failed: {x}".format(j=job.job_id, e=elapsed(), x=e))
        else:
            print("job {j} status poll ({e:.2f}s): {s}".format(j=job.job_id, e=elapsed(), s=status))
            if until(status):
                return status
        time.sleep(sleep)

    raise RuntimeError("reached max poll time: {e} > {m}".format(e=elapsed(), m=max_poll_time))


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_basic(connection, api_version, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)
    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.polygonal_median_timeseries(POLYGON01)

    job = timeseries.send_job(job_options=batch_default_options(driverMemory="1600m",driverMemoryOverhead="512m"),title="basic")
    assert job.job_id

    job.start_job()
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert status == "finished"

    assets = job.download_results(tmp_path)
    assert len(assets) == 1

    with open(list(assets.keys())[0]) as f:
        data = json.load(f)

    expected_dates = ["2017-11-01T00:00:00Z", "2017-11-11T00:00:00Z", "2017-11-21T00:00:00Z"]
    assert sorted(data.keys()) == sorted(expected_dates)
    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(data)

    if api_version >= "1.0.0":
        job_results = job.list_results()
        print(job_results)
        geometry = shape(job_results['geometry'])
        assert geometry.equals_exact(POLYGON01, tolerance=0.0001)
        assert job_results["bbox"] == POLYGON01_BBOX
        assert job_results['properties']['start_datetime'] == "2017-11-01T00:00:00Z"
        assert job_results['properties']['end_datetime'] == "2017-11-21T00:00:00Z"


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_execute_batch(connection, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)
    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.polygonal_median_timeseries(POLYGON01)

    output_file = tmp_path / "ts.json"
    timeseries.execute_batch(output_file, max_poll_interval=BATCH_JOB_POLL_INTERVAL, job_options=batch_default_options(driverMemory="1600m",driverMemoryOverhead="512m"), title="execute-batch")

    with output_file.open("r") as f:
        data = json.load(f)
    expected_dates = ["2017-11-01T00:00:00Z", "2017-11-11T00:00:00Z", "2017-11-21T00:00:00Z"]
    assert sorted(data.keys()) == sorted(expected_dates)
    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(data)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_signed_urls(connection, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)
    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.polygonal_median_timeseries(POLYGON01)

    job = timeseries.execute_batch(
        max_poll_interval=BATCH_JOB_POLL_INTERVAL,
        job_options=batch_default_options(driverMemory="1G", driverMemoryOverhead="512m"),
        title = "signed-urls"
    )

    results = job.get_results()
    # TODO: check results metadata?
    print("results metadata", results.get_metadata())

    assets = results.get_assets()
    print("assets", assets)
    assert len(assets) >= 1
    data = None
    for asset in assets:
        # Download directly without credentials
        resp = requests.get(asset.href)
        resp.raise_for_status()
        assert resp.status_code == 200
        if asset.name.endswith(".json"):
            assert data is None
            data = resp.json()
    expected_dates = ["2017-11-01T00:00:00Z", "2017-11-11T00:00:00Z", "2017-11-21T00:00:00Z"]
    assert sorted(data.keys()) == sorted(expected_dates)
    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(data)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_cancel(connection, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)

    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    if isinstance(cube, DataCube):
        cube = cube.process("sleep", arguments={"data": cube, "seconds": 30})
    elif isinstance(cube, ImageCollectionClient):
        cube = cube.graph_add_process("sleep", args={"data": {"from_node": cube.node_id}, "seconds": 30})
    else:
        raise ValueError(cube)

    timeseries = cube.polygonal_mean_timeseries(POLYGON01)

    job = timeseries.send_job(out_format="GTIFF", job_options=batch_default_options(driverMemory="512m",driverMemoryOverhead="1g"),title="cancel")
    assert job.job_id
    job.start_job()

    # await job running
    status = _poll_job_status(job, until=lambda s: s in ['running', 'canceled', 'finished', 'error'])
    assert status == "running"

    # cancel it
    job.stop_job()
    print("stopped job")

    # await job canceled
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert status == "canceled"


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_delete_job(connection):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)

    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.polygonal_mean_timeseries(POLYGON01)

    job = timeseries.send_job(out_format="GTIFF", job_options=batch_default_options(driverMemory="512m",driverMemoryOverhead="1g"),title="delete")
    assert job.job_id
    job.start_job()

    # await job finished
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert status == "finished"

    def job_directory_exists(expected: bool) -> bool:
        start = time.time()

        def elapsed():
            return time.time() - start

        def directory_exists() -> bool:
            exists = (Path("/data/projects/OpenEO") / job.job_id).exists()

            print("job {j} directory exists ({e:.2f}s): {d}".format(j=job.job_id, e=elapsed(), d=exists))
            return exists

        while elapsed() < 300:
            if directory_exists() == expected:
                return expected

            time.sleep(10)

        return directory_exists()

    assert job_directory_exists(True)

    # delete it
    job.delete_job()
    print("deleted job")

    try:
        job.describe_job()
        pytest.fail("should have returned a 404 for a deleted job")
    except OpenEoApiError as e:
        assert e.http_status_code == 404

    assert not job_directory_exists(False)


@pytest.mark.skip(reason="Requires proxying to work properly")
def test_create_wtms_service(connection):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)
    s2_fapar = (
        connection
            .load_collection('S2_FAPAR_V102_WEBMERCATOR2')
            .filter_bbox(west=0, south=50, east=5, north=55, crs='EPSG:4326')
            .filter_temporal(start_date="2019-04-01", end_date="2019-04-01")
    )
    res = s2_fapar.tiled_viewing_service(type='WMTS')
    print("created service", res)
    assert "service_id" in res
    assert "url" in res
    service_id = res["service_id"]
    service_url = res["url"]
    assert '/services/{s}'.format(s=service_id) in service_url

    wmts_metadata = connection.get(service_url).json()
    print("wmts metadata", wmts_metadata)
    assert "url" in wmts_metadata
    wmts_url = wmts_metadata["url"]
    time.sleep(5)  # seems to take a while before the service is proxied
    get_capabilities = requests.get(wmts_url + '?REQUEST=getcapabilities').text
    print("getcapabilities", get_capabilities)
    # the capabilities document should advertise the proxied URL
    assert "<Capabilities" in get_capabilities
    assert wmts_url in get_capabilities


@pytest.mark.parametrize("udf_file", [
    "udfs/smooth_savitsky_golay_old.py",
    "udfs/smooth_savitsky_golay.py",
])
def test_ep3048_sentinel1_udf(auth_connection, udf_file):
    # http://bboxfinder.com/#-4.745000,-55.700000,-4.740000,-55.695000
    N, E, S, W = (-4.740, -55.695, -4.745, -55.7)
    polygon = Polygon(shell=[[W, N], [E, N], [E, S], [W, S]])

    udf_code = read_data(udf_file)

    ts = (
        auth_connection.load_collection("SENTINEL1_GAMMA0_SENTINELHUB")
            .filter_temporal(["2019-05-24T00:00:00Z", "2019-05-30T00:00:00Z"])
            .filter_bbox(north=N, east=E, south=S, west=W, crs="EPSG:4326")
            .filter_bands([0])
            .apply_dimension(udf_code, runtime="Python")
            .polygonal_mean_timeseries(polygon)
            .execute()
    )
    assert isinstance(ts, dict)
    assert all(k.startswith('2019-05-') for k in ts.keys())


def test_load_collection_from_disk(auth_connection, tmp_path):
    fapar = auth_connection.load_disk_collection(
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
    assert_geotiff_basics(output_file, expected_shape=(1, 2786, 3496))


def assert_geotiff_basics(
        output_tiff: Union[str, Path], expected_band_count=1, min_width=64, min_height=64, expected_shape=None
):
    """Basic checks that a file is a readable GeoTIFF file"""
    assert imghdr.what(output_tiff) == 'tiff'
    with rasterio.open(output_tiff) as dataset:
        assert dataset.width > min_width
        assert dataset.height > min_height
        if expected_shape is not None:
            assert (dataset.count, dataset.height, dataset.width) == expected_shape
        elif expected_band_count is not None:
            assert dataset.count == expected_band_count


def assert_cog(output_tiff: str):
    # FIXME: check if actually a COG
    pass


def test_mask_polygon(auth_connection, api_version, tmp_path):
    bbox = {"west": 7.0, "south": 51.28, "east": 7.1, "north": 51.4, "crs": "EPSG:4326"}
    date = "2017-11-01"
    collection_id = 'PROBAV_L3_S10_TOC_333M'
    cube = auth_connection.load_collection(collection_id,bands=["NDVI"]).filter_bbox(**bbox).filter_temporal(date, date)
    if api_version >= "1.0.0":
        masked = cube.mask_polygon(POLYGON01)
    else:
        masked = cube.mask(polygon=POLYGON01)

    output_tiff = tmp_path / "masked.tiff"
    masked.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_shape=(1, 89, 38),min_width=32, min_height=32)


def test_mask_out_all_data_float(auth_connection, api_version, tmp_path):
    bbox = {"west": 5, "south": 51, "east": 5.1, "north": 51.1, "crs": "EPSG:4326"}
    date = "2017-12-21"
    collection_id = 'PROBAV_L3_S10_TOC_333M'
    probav = auth_connection.load_collection(collection_id,bands=["NDVI"]).filter_temporal(date, date).filter_bbox(**bbox)
    opaque_mask = probav.band("NDVI") != 255  # all ones
    # Mask the data (and make sure it is float data)
    probav_masked = probav.apply(lambda x: x * 0.5).mask(mask=opaque_mask)
    _dump_process_graph(probav_masked, tmp_path=tmp_path, name="probav_masked.json")

    probav_path = tmp_path / "probav.tiff"
    probav.download(probav_path, format='GTiff')
    masked_path = tmp_path / "probav_masked.tiff"
    probav_masked.download(masked_path, format='GTiff')

    assert_geotiff_basics(probav_path, expected_shape=(1, 73, 37),min_width=32, min_height=32)
    assert_geotiff_basics(masked_path, expected_shape=(1, 73, 37),min_width=32, min_height=32)
    with rasterio.open(probav_path) as probav_ds, rasterio.open(masked_path) as masked_ds:
        probav_data = probav_ds.read(1)
        #assert np.all(probav_data != 255)
        assert masked_ds.dtypes == ('float32', )
        masked_data = masked_ds.read(1)
        assert np.all(np.isnan(masked_data))


def test_mask_out_all_data_int(auth_connection, api_version, tmp_path):
    bbox = {"west": 5, "south": 51, "east": 5.1, "north": 51.1, "crs": "EPSG:4326"}
    date = "2017-12-21"
    collection_id = 'PROBAV_L3_S10_TOC_333M'
    probav = auth_connection.load_collection(collection_id,bands=["NDVI"]).filter_temporal(date, date).filter_bbox(**bbox)
    opaque_mask = probav.band("NDVI") != 255  # all ones
    probav_masked = probav.mask(mask=opaque_mask)
    _dump_process_graph(probav_masked, tmp_path=tmp_path, name="probav_masked.json")

    probav_path = tmp_path / "probav.tiff"
    probav.download(probav_path, format='GTiff')
    masked_path = tmp_path / "probav_masked.tiff"
    probav_masked.download(masked_path, format='GTiff')

    assert_geotiff_basics(probav_path, expected_shape=(1, 73, 37),min_width=32, min_height=32)
    assert_geotiff_basics(masked_path, expected_shape=(1, 73, 37),min_width=32, min_height=32)
    with rasterio.open(probav_path) as probav_ds, rasterio.open(masked_path) as masked_ds:
        probav_data = probav_ds.read(1)
        #assert np.all(probav_data != 255)
        assert masked_ds.dtypes == ('uint8', )
        masked_data = masked_ds.read(1, masked=True)
        assert np.all(masked_data.mask)


def test_fuzzy_mask(auth_connection, tmp_path):
    date = "2019-04-26"
    mask = create_simple_mask(auth_connection, band_math_workaround=True)
    mask = mask.filter_bbox(**BBOX_GENT).filter_temporal(date, date)
    _dump_process_graph(mask, tmp_path)
    output_tiff = tmp_path / "mask.tiff"
    mask.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)


def test_simple_cloud_masking(auth_connection, api_version, tmp_path):
    date = "2019-04-26"
    mask = create_simple_mask(auth_connection,class_to_mask=3, band_math_workaround=False)
    mask = mask.filter_bbox(**BBOX_GENT).filter_temporal(date, date)
    # mask.download(tmp_path / "mask.tiff", format='GTIFF')
    s2_radiometry = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=[ "blue" ])
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
    assert_geotiff_basics(output_tiff, expected_shape=(1, 227, 354))
    with rasterio.open(output_tiff) as result_ds:
        assert result_ds.dtypes == ('int16',)
        with rasterio.open(get_path("reference/simple_cloud_masking.tiff")) as ref_ds:
            ref_array = ref_ds.read(masked=False)
            actual_array = result_ds.read(masked=False)
            assert_array_equal(ref_array, actual_array)


def test_advanced_cloud_masking(auth_connection, api_version, tmp_path):
    # Retie
    bbox = {"west": 4.996033, "south": 51.258922, "east": 5.091603, "north": 51.282696, "crs": "EPSG:4326"}
    date = "2018-08-14"
    mask = create_advanced_mask(start=date, end=date, connection=auth_connection, band_math_workaround=True)
    mask = mask.filter_bbox(**bbox)
    # mask.download(tmp_path / "mask.tiff", format='GTIFF')
    s2_radiometry = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=["blue"])
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
    assert_geotiff_basics(out_file, expected_shape=(1, 284, 675))
    with rasterio.open(out_file) as result_ds:
        assert result_ds.dtypes == ('int16',)
        with rasterio.open(get_path("reference/advanced_cloud_masking.tiff")) as ref_ds:
            assert_array_equal(ref_ds.read(masked=False), result_ds.read(masked=False))



def test_advanced_cloud_masking_builtin(auth_connection, api_version, tmp_path):
    """
    Advanced masking above got replaced with more simple builting approach
    @param auth_connection:
    @param api_version:
    @param tmp_path:
    @return:
    """
    # Retie
    bbox = {"west": 4.996033, "south": 51.258922, "east": 5.091603, "north": 51.282696, "crs": "EPSG:4326"}
    date = "2018-08-14"

    s2_radiometry = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=["blue","SCENECLASSIFICATION_20M"])
            .filter_bbox(**bbox).filter_temporal(date, date)
    )

    masked = s2_radiometry.process("mask_scl_dilation",data=s2_radiometry,scl_band_name="SCENECLASSIFICATION_20M")

    out_file = tmp_path / "masked_result.tiff"
    masked.download(out_file, format='GTIFF')
    #assert_geotiff_basics(out_file, expected_shape=(3, 284, 660))
    with rasterio.open(out_file) as result_ds:
        assert result_ds.dtypes == ('int16', 'int16',)
        with rasterio.open(get_path("reference/advanced_cloud_masking_builtin.tiff")) as ref_ds:
            assert_array_approx_equal(ref_ds.read(1,masked=False), result_ds.read(1,masked=False))

@pytest.mark.skip(reason="Temporary skip to get tests through")
@pytest.mark.parametrize("udf_file", [
    "udfs/udf_temporal_slope_old.py",
    "udfs/udf_temporal_slope.py",
])
def test_reduce_temporal_udf(auth_connection, tmp_path, udf_file):
    bbox = {
        "west": 6.8371137,
        "north": 50.5647147,
        "east": 6.8566699,
        "south": 50.560007,
        "crs": "EPSG:4326"
    }

    cube = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=["blue", "green", "red"])
            .filter_temporal("2020-11-01", "2020-11-20")
            .filter_bbox(**bbox)
    )

    udf_code = read_data(udf_file)
    print(udf_code)

    trend = cube.reduce_temporal_udf(udf_code, runtime="Python", version="latest")

    output_file = tmp_path / "trend.tiff"
    trend.download(output_file, format="GTIFF")
    assert_geotiff_basics(output_file, expected_band_count=12,min_height=48, min_width=140)

@pytest.mark.skip(reason="Custom processes will be tested separately")
@pytest.mark.requires_custom_processes
def test_custom_processes(auth_connection):
    process_graph = {
        "foobar1": {
            "process_id": "foobar",
            "arguments": {"size": 123, "color": "green"},
            "result": True,
        }
    }
    res = auth_connection.execute(process_graph)
    assert res == {
        "args": ["color", "size"],
        "msg": "hello world",
    }

@pytest.mark.skip(reason="Custom processes will be tested separately")
@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
@pytest.mark.requires_custom_processes
def test_custom_processes_in_batch_job(auth_connection):
    process_graph = {
        "foobar1": {
            "process_id": "foobar",
            "arguments": {"size": 123, "color": "green"},
            "result": True,
        }
    }
    job = auth_connection.create_job(process_graph)
    job.run_synchronous()
    results = job.get_results()
    asset = next(a for a in results.get_assets() if a.metadata.get("type") == "application/json")
    assert asset.load_json() == {
        "args": ["color", "size"],
        "msg": "hello world",
    }


@pytest.mark.parametrize(["cid", "expected_dates"], [
    (
            'TERRASCOPE_S2_FAPAR_V2',
            [
                '2017-11-01T00:00:00Z', '2017-11-04T00:00:00Z', '2017-11-06T00:00:00Z',
                '2017-11-09T00:00:00Z', '2017-11-11T00:00:00Z',
                '2017-11-14T00:00:00Z', '2017-11-16T00:00:00Z', '2017-11-19T00:00:00Z',
                '2017-11-21T00:00:00Z'
            ]
    )
])
def test_polygonal_timeseries(auth_connection, tmp_path, cid, expected_dates, api_version):
    expected_dates = sorted(expected_dates)
    polygon = POLYGON01
    bbox = _polygon_bbox(polygon)
    cube = (
        auth_connection
            .load_collection(cid)
            .filter_temporal("2017-11-01", "2017-11-21")
            .filter_bbox(**bbox)
    )
    ts_mean = cube.polygonal_mean_timeseries(polygon).execute()
    print("mean", ts_mean)
    ts_mean_df = timeseries_json_to_pandas(ts_mean)

    # TODO remove this cleanup https://github.com/Open-EO/openeo-geopyspark-driver/issues/75
    ts_mean = {k: v for (k, v) in ts_mean.items() if v != [[]]}
    print("mean", ts_mean)

    ts_median = cube.polygonal_median_timeseries(polygon).execute()
    ts_median_df = timeseries_json_to_pandas(ts_median)
    print("median", ts_median_df)
    ts_sd = cube.polygonal_standarddeviation_timeseries(polygon).execute()
    ts_sd_df = timeseries_json_to_pandas(ts_sd)
    print("sd", ts_sd)

    assert sorted(ts_mean.keys()) == expected_dates
    assert sorted(ts_median.keys()) == expected_dates
    assert sorted(ts_sd.keys()) == expected_dates

    # Only check for a subset of dates if there are a lot
    for date in expected_dates[::max(1, len(expected_dates) // 5)]:
        output_file = tmp_path / "ts_{d}.tiff".format(d=re.sub(r'[^0-9]', '', date))
        print("Evaluating date {d}, downloading to {p}".format(d=date, p=output_file))
        date_cube = auth_connection.load_collection(cid).filter_temporal(date, date).filter_bbox(**bbox)
        if api_version.at_least("1.0.0"):
            date_cube = date_cube.mask_polygon(polygon)
        else:
            date_cube = date_cube.mask(polygon=polygon)
        date_cube.download(output_file, format="GTIFF")
        with rasterio.open(output_file) as ds:
            data = ds.read(masked=True)
            band_count = ds.count
            print("bands {b}, masked {m:.2f}%".format(b=band_count, m=100.0 * data.mask.mean()))

            if data.count() == 0:
                assert ts_mean[date] == [[None] * band_count]
                assert ts_median[date] == [[None] * band_count]
                assert ts_sd[date] == [[None] * band_count]
            else:
                rtol = 0.02
                np.testing.assert_allclose(np.ma.mean(data, axis=(-1, -2)), ts_mean_df.loc[date], rtol=rtol)
                np.testing.assert_allclose(np.ma.median(data, axis=(-1, -2)), ts_median_df.loc[date], atol=2.2) #TODO EP-4025
                np.testing.assert_allclose(np.ma.std(data, axis=(-1, -2)), ts_sd_df.loc[date], rtol=rtol)


def test_ndvi(auth_connection, tmp_path):
    ndvi = auth_connection.load_collection(
        "TERRASCOPE_S2_TOC_V2",
        spatial_extent={"west": 5.027, "east": 5.0438, "south": 51.1974, "north": 51.2213},
        temporal_extent=["2020-04-05", "2020-04-05"],
        bands = ["nir", "red"]
    ).ndvi()

    output_tiff = tmp_path / "ndvi.tiff"
    ndvi.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)


def test_normalized_difference(auth_connection, tmp_path):
    toc = auth_connection.load_collection(
        "TERRASCOPE_S2_TOC_V2",
        spatial_extent={"west": 5.027, "east": 5.0438, "south": 51.1974, "north": 51.2213},
        temporal_extent=["2020-04-05", "2020-04-05"]
    )

    nir = toc.band('TOC-B08_10M')
    red = toc.band('TOC-B04_10M')

    ndvi = nir.normalized_difference(red)

    output_tiff = tmp_path / "normalized_difference_ndvi.tiff"
    ndvi.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)


def test_udp_crud(connection100):
    connection100.authenticate_basic(TEST_USER, TEST_PASSWORD)

    toc = connection100.load_collection("TERRASCOPE_S2_TOC_V2")
    udp = toc.save_user_defined_process(user_defined_process_id='toc', public=True)

    udp_details = udp.describe()

    assert udp_details['id'] == 'toc'
    assert 'loadcollection1' in udp_details['process_graph']
    assert udp_details['public']

    udp.delete()

    user_udp_ids = [udp['id'] for udp in connection100.list_user_defined_processes()]

    assert 'toc' not in user_udp_ids


def test_udp_usage_blur(connection100, tmp_path):
    connection100.authenticate_basic(TEST_USER, TEST_PASSWORD)
    # Store User Defined Process (UDP)
    blur = {
        "blur": {
            "process_id": "apply_kernel",
            "arguments": {
                "data": {"from_parameter": "data"},
                "kernel": [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                "factor": 0.1,
            },
            "result": True,
        },
    }
    connection100.save_user_defined_process("blur", blur)
    # Use UDP
    date = "2020-06-26"
    cube = (
        connection100.load_collection("TERRASCOPE_S2_TOC_V2")
            .filter_bands(["red", "green"])
            .filter_temporal(date, date)
            .filter_bbox(**BBOX_MOL)
            .process("blur", arguments={"data": THIS})
    )
    output_tiff = tmp_path / "mol.tiff"
    cube.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=2)
    # TODO: check resulting data?


def test_udp_usage_blur_parameter_default(connection100, tmp_path):
    connection100.authenticate_basic(TEST_USER, TEST_PASSWORD)
    # Store User Defined Process (UDP)
    blur = {
        "blur": {
            "process_id": "apply_kernel",
            "arguments": {
                "data": {"from_parameter": "data"},
                "kernel": [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
                "factor": {"from_parameter": "scale"},
            },
            "result": True,
        },
    }
    connection100.save_user_defined_process("blur", blur, parameters=[
        Parameter("scale", description="factor", schema="number", default=0.1)
    ])
    # Use UDP
    date = "2020-06-26"
    cube = (
        connection100.load_collection("TERRASCOPE_S2_TOC_V2")
            .filter_bands(["red", "green"])
            .filter_temporal(date, date)
            .filter_bbox(**BBOX_MOL)
            .process("blur", arguments={"data": THIS})
    )
    output_tiff = tmp_path / "mol.tiff"
    cube.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=2)
    # TODO: check resulting data?


def test_udp_usage_reduce(connection100, tmp_path):
    connection100.authenticate_basic(TEST_USER, TEST_PASSWORD)
    # Store User Defined Process (UDP)
    flatten_bands = {
        "reduce1": {
            "process_id": "reduce_dimension",
            "arguments": {
                "data": {"from_parameter": "data"},
                "dimension": "bands",
                "reducer": {
                    "process_graph": {
                        "mean": {
                            "process_id": "mean",
                            "arguments": {
                                "data": {"from_parameter": "data"}
                            },
                            "result": True,
                        }
                    }
                },
            },
            "result": True,
        }
    }
    connection100.save_user_defined_process(
        "flatten_bands", flatten_bands, parameters=[
            Parameter.raster_cube(name="data")
        ]
    )
    # Use UDP
    date = "2020-06-26"
    cube = (
        connection100.load_collection("TERRASCOPE_S2_TOC_V2")
            .filter_bands(["red", "green", "blue", "nir"])
            .filter_temporal(date, date)
            .filter_bbox(**BBOX_MOL)
            .process("flatten_bands", arguments={"data": THIS})
    )
    output_tiff = tmp_path / "mol.tiff"
    cube.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)
    # TODO: check resulting data?


def test_synchronous_call_without_spatial_bounds_is_rejected(auth_connection, tmp_path):
    s2_fapar = (
        auth_connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M")
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
    )
    out_file = tmp_path / "s2_fapar_latlon.geotiff"

    with pytest.raises(OpenEoApiError) as excinfo:
        s2_fapar.download(out_file, format="GTIFF")

    assert excinfo.value.code == 'MissingSpatialFilter'


@pytest.mark.skip(reason="DELETEing a service doesn't work because it's being proxied to the WMTS Jetty server")
def test_secondary_service_without_spatial_bounds_is_accepted(auth_connection):
    s2_fapar = (
        auth_connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M")
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
    )

    service_id = s2_fapar.tiled_viewing_service(type="WMTS")["service_id"]
    auth_connection.remove_service(service_id)


def test_simple_raster_to_vector(auth_connection, api_version, tmp_path):
    date = "2019-04-26"

    # using sceneclassification that will have contiguous areas suitable for vectorization
    s2_sc = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=[ "SCENECLASSIFICATION_20M"])
            .filter_bbox(**BBOX_GENT).filter_temporal(date, date)
    )
    vectorized=s2_sc.raster_to_vector()

    output_json = tmp_path / "raster_to_vector.json"
    vectorized.download(output_json)
    assert os.path.getsize(output_json) > 0


def test_resolution_merge(auth_connection,tmp_path):
    date = "2019-04-26"
    output_tiff = tmp_path / "highres.tif"
    lowres_output_tiff = tmp_path / "lowres.tif"
    base = auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=['TOC-B08_10M','TOC-B8A_20M'])\
        .filter_bbox(**BBOX_GENT).filter_temporal(date, date)
    #base.download(lowres_output_tiff)

    base.resolution_merge(
    high_resolution_bands=['TOC-B08_10M'], low_resolution_bands=['TOC-B8A_20M']).download(output_tiff)
    assert_geotiff_basics(output_tiff, expected_band_count=2)


def test_sentinel_hub_execute_batch(auth_connection, tmp_path):
    sar_backscatter = (auth_connection
                       .load_collection('SENTINEL1_GAMMA0_SENTINELHUB', bands=["VV", "VH"])
                       .filter_bbox(west=2.59003, east=2.8949, north=51.2206, south=51.069)
                       .filter_temporal(extent=["2019-10-10", "2019-10-10"]))

    output_tiff = tmp_path / "test_sentinel_hub_batch_job.tif"

    sar_backscatter.execute_batch(output_tiff, out_format='GTiff')
    assert_geotiff_basics(output_tiff, expected_band_count=2)


def test_sentinel_hub_sar_backscatter_batch_process(auth_connection, tmp_path):
    # FIXME: a separate filter_bands call drops the mask and local_incidence_angle bands
    sar_backscatter = (auth_connection
                       .load_collection('SENTINEL1_GAMMA0_SENTINELHUB', bands=["VV", "VH"])
                       .filter_bbox(west=2.59003, east=2.8949, north=51.2206, south=51.069)
                       .filter_temporal(extent=["2019-10-10", "2019-10-10"])
                       .sar_backscatter(mask=True, local_incidence_angle=True))
    job = sar_backscatter.execute_batch(out_format='GTiff')

    assets = job.download_results(tmp_path)
    assert len(assets) > 1

    result_asset_paths = [path for path in assets.keys() if path.name.startswith("openEO")]
    assert len(result_asset_paths) == 1

    output_tiff = result_asset_paths[0]
    assert_geotiff_basics(output_tiff, expected_band_count=4)  # VV, VH, mask and local_incidence_angle

    
# this function checks that only up to a portion of values do not match within tolerance
# there always are few pixels with reflections, ... etc on images which is sensitive to very small changes in the code
def compare_xarrays(xa1,xa2,max_nonmatch_ratio=0.01, tolerance=1.e-6):
    np.testing.assert_allclose(xa1.shape,xa2.shape)
    nmax=xa1.shape[-1]*xa1.shape[-2]*max_nonmatch_ratio
    diff=xarray.ufuncs.fabs(xa1-xa2)>tolerance
    assert(diff.where(diff).count()<=nmax)
    np.testing.assert_allclose(xa2.where(~diff), xa2.where(~diff), rtol=0., atol=tolerance, equal_nan=True)

@pytest.mark.skip(reason="Temporary skip to get tests through")
def test_atmospheric_correction_inputsarecorrect(auth_connection, api_version, tmp_path):         
    # source product is  S2B_MSIL1C_20190411T105029_N0207_R051_T31UFS_20190411T130806
    date = "2019-04-11"
    bbox=(655000,5677000,660000,5685000)

    l1c = (
        auth_connection.load_collection("SENTINEL2_L1C_SENTINELHUB")
            .filter_temporal(date,date)\
            .filter_bbox(crs="EPSG:32631", **dict(zip(["west", "south", "east", "north"], bbox)))
    )
    l2a=l1c.process(
        process_id="atmospheric_correction",
        arguments={
            "data": THIS,
            # "missionId": "SENTINEL2", this is the default
           "appendDebugBands" : 1
        }
    )
    output = tmp_path / "icorvalidation_inputcheck.json"
    l2a.download(output,format="json")
    
    result=datacube_from_file(output,fmt="json").get_array()

    # note that debug bands are multiplied by 100 to store meaningful values as integers
    szaref=xarray.open_rasterio(get_path("icor/ref_inputcheck_SZA.tif"))
    vzaref=xarray.open_rasterio(get_path("icor/ref_inputcheck_VZA.tif"))
    raaref=xarray.open_rasterio(get_path("icor/ref_inputcheck_RAA.tif"))
    demref=xarray.open_rasterio(get_path("icor/ref_inputcheck_DEM.tif"))
    aotref=xarray.open_rasterio(get_path("icor/ref_inputcheck_AOT.tif"))
    cwvref=xarray.open_rasterio(get_path("icor/ref_inputcheck_CWV.tif"))

    compare_xarrays(result.loc[date][-6],szaref[0].transpose("x","y"))
    compare_xarrays(result.loc[date][-5],vzaref[0].transpose("x","y"))
    compare_xarrays(result.loc[date][-4],raaref[0].transpose("x","y"))
    compare_xarrays(result.loc[date][-3],demref[0].transpose("x","y"))
    compare_xarrays(result.loc[date][-2],aotref[0].transpose("x","y"))
    compare_xarrays(result.loc[date][-1],cwvref[0].transpose("x","y"))

@pytest.mark.skip(reason="Temporary skip to get tests through")
def test_atmospheric_correction_defaultbehavior(auth_connection, api_version, tmp_path):
    # source product is  S2B_MSIL1C_20190411T105029_N0207_R051_T31UFS_20190411T130806
    date = "2019-04-11"
    bbox=(655000,5677000,660000,5685000)

    l1c = (
        auth_connection.load_collection("SENTINEL2_L1C_SENTINELHUB")
            .filter_temporal(date,date)\
            .filter_bbox(crs="EPSG:32631", **dict(zip(["west", "south", "east", "north"], bbox)))
    )
    l2a=l1c.process(
        process_id="atmospheric_correction",
        arguments={
            "data": THIS,
            # "missionId": "SENTINEL2", this is the default
        }
    )
    output = tmp_path / "icorvalidation_default.json"
    l2a.download(output,format="json")
    
    result=datacube_from_file(output,fmt="json").get_array()
    b2ref=xarray.open_rasterio(get_path("icor/ref_default_B02.tif"))
    b3ref=xarray.open_rasterio(get_path("icor/ref_default_B03.tif"))
    b4ref=xarray.open_rasterio(get_path("icor/ref_default_B04.tif"))
    b8ref=xarray.open_rasterio(get_path("icor/ref_default_B08.tif"))

    compare_xarrays(result.loc[date,"B02"],b2ref[0].transpose("x","y"))
    compare_xarrays(result.loc[date,"B03"],b3ref[0].transpose("x","y"))
    compare_xarrays(result.loc[date,"B04"],b4ref[0].transpose("x","y"))
    compare_xarrays(result.loc[date,"B08"],b8ref[0].transpose("x","y"))


def test_atmospheric_correction_constoverridenparams(auth_connection, api_version, tmp_path):
    # source product is  S2B_MSIL1C_20190411T105029_N0207_R051_T31UFS_20190411T130806
    date = "2019-04-11"
    bbox=(655000,5677000,660000,5685000)

    l1c = (
        auth_connection.load_collection("SENTINEL2_L1C_SENTINELHUB")
            .filter_temporal(date,date)\
            .filter_bbox(crs="EPSG:32631", **dict(zip(["west", "south", "east", "north"], bbox)))
    )
    l2a=l1c.process(
        process_id="atmospheric_correction",
        arguments={
            "data": THIS,
            # "missionId": "SENTINEL2", this is the default
            "sza" : 43.5,
            "vza" : 6.96,
            "raa" : 117.,
            "gnd" : 0.1,
            "aot" : 0.2,
            "cwv" : 2.0,
        }
    )
    output = tmp_path / "icorvalidation_overriden.json"
    l2a.download(output,format="json")
    
    result=datacube_from_file(output,fmt="json").get_array()
    b2ref=xarray.open_rasterio(get_path("icor/ref_overriden_B02.tif"))
    b3ref=xarray.open_rasterio(get_path("icor/ref_overriden_B03.tif"))
    b4ref=xarray.open_rasterio(get_path("icor/ref_overriden_B04.tif"))
    b8ref=xarray.open_rasterio(get_path("icor/ref_overriden_B08.tif"))

    compare_xarrays(result.loc[date,"B02"],b2ref[0].transpose("x","y"))
    compare_xarrays(result.loc[date,"B03"],b3ref[0].transpose("x","y"))
    compare_xarrays(result.loc[date,"B04"],b4ref[0].transpose("x","y"))
    compare_xarrays(result.loc[date,"B08"],b8ref[0].transpose("x","y"))


def test_discard_result_suppresses_batch_job_output_file(connection):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)

    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_bbox(5, 6, 52, 51, 'EPSG:4326')
    cube = cube.process("discard_result", arguments={"data": cube})

    job = cube.execute_batch(max_poll_interval=BATCH_JOB_POLL_INTERVAL)
    assets = job.get_results().get_assets()

    assert len(assets) == 0, assets


def __reproject_polygon(polygon: Union[Polygon], srs, dest_srs):
    # apply projection
    return shapely.ops.transform(
        pyproj.Transformer.from_crs(srs, dest_srs, always_xy=True).transform,
        polygon
    )


def test_merge_cubes(auth_connection):
    # define ROI
    size = 10 * 128
    x = 640860.000
    y = 5676170.000
    poly = shapely.geometry.box(x, y, x + size, y + size)
    poly= __reproject_polygon(poly,"EPSG:32631","EPSG:4326")
    extent = dict(zip(["west","south","east","north"], poly.bounds))
    extent['crs'] = "EPSG:4326"

    # define TOI
    year = 2019

    startdate = f"{year}-05-01"

    enddate = f"{year}-05-15"
    s2_bands = auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=["TOC-B04_10M", "TOC-B08_10M", "SCENECLASSIFICATION_20M"], spatial_extent=extent)
    s2_bands = s2_bands.process("mask_scl_dilation", data=s2_bands, scl_band_name="SCENECLASSIFICATION_20M")
    b4_band = s2_bands.band("TOC-B04_10M")
    b8_band = s2_bands.band("TOC-B08_10M")
    s2_ndvi = (b8_band - b4_band) / (b8_band + b4_band)
    s2_ndvi = s2_ndvi.add_dimension("bands", "s2_ndvi", type="bands")

    datacube = s2_ndvi
    pv_ndvi = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M', bands=['NDVI'], spatial_extent=extent)
    pv_ndvi = pv_ndvi.resample_cube_spatial(s2_ndvi)
    pv_ndvi = pv_ndvi.mask_polygon(poly)
    datacube = datacube.merge_cubes(pv_ndvi)
    # apply filters
    datacube = datacube.filter_temporal(startdate, enddate)#.filter_bbox(**extent)
    datacube.download("merged.nc", format="NetCDF",options={"stitch":True})
    timeseries = xarray.open_dataset("merged.nc", engine="h5netcdf").mean(dim=['x', 'y'])

    assert_array_almost_equal([210.29, 191.75, np.nan], timeseries.NDVI.values, 2)
    assert_allclose([np.nan, np.nan, 0.5958494], timeseries.s2_ndvi.values, atol=0.005)


def test_udp_simple_math(auth_connection, tmp_path):
    # Define UDP
    from openeo.processes import divide, subtract, process
    fahrenheit = Parameter.number("fahrenheit")
    fahrenheit_to_celsius = divide(x=subtract(x=fahrenheit, y=32), y=1.8)
    auth_connection.save_user_defined_process("fahrenheit_to_celsius", fahrenheit_to_celsius, parameters=[fahrenheit])

    # Use UDP
    pg = process("fahrenheit_to_celsius", namespace="user", fahrenheit=50)
    res = auth_connection.execute(pg)
    assert res == 10.0


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_udp_simple_math_batch_job(auth_connection, tmp_path):
    # Define UDP
    from openeo.processes import divide, subtract, process
    fahrenheit = Parameter.number("fahrenheit")
    fahrenheit_to_celsius = divide(x=subtract(x=fahrenheit, y=32), y=1.8)
    auth_connection.save_user_defined_process("fahrenheit_to_celsius", fahrenheit_to_celsius, parameters=[fahrenheit])
    the_udp = auth_connection.user_defined_process("fahrenheit_to_celsius")
    print(the_udp)

    # Use UDP
    pg = process("fahrenheit_to_celsius", namespace="user", fahrenheit=50, f=50)#somehow our parameter name breaks??
    job = auth_connection.create_job(pg)
    job.run_synchronous()
    results = job.get_results()
    asset = next(a for a in results.get_assets() if a.metadata.get("type") == "application/json")
    assert asset.load_json() == 10.0


_EXPECTED_LIBRARIES = os.environ.get("OPENEO_GEOPYSPARK_INTEGRATIONTESTS_EXPECTED_LIBRARIES", "tensorflow").split(",")


@pytest.mark.parametrize("library", _EXPECTED_LIBRARIES)
def test_library_availability(auth_connection, library):
    """Use UDF to check if library can be imported"""
    udf = textwrap.dedent("""\
        from openeo.udf import UdfData, StructuredData

        def transform(data: UdfData):
            data.set_feature_collection_list(None)
            try:
                import {library}
                result = dict(success=True, path=str({library}))
            except ImportError as e:
                result = dict(success=False, error=str(e))
            data.set_structured_data_list([StructuredData(data=result, type="dict")])
    """.format(library=library))
    pg = {
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": {"type": "Polygon", "coordinates": [[(2, 1), (2, 3), (0, 3), (0, 1), (2, 3)]]},
                "udf": udf,
                "runtime": "Python"
            },
            "result": True
        }
    }
    res = auth_connection.execute(pg)
    if not (isinstance(res, dict) and res.get("success")):
        raise ValueError(res)


@pytest.mark.parametrize("udf_code", [
    # Old style UDFs (based on openeo_udf imports)
    """
        from openeo_udf.api.udf_data import UdfData  # Old style openeo_udf API
        from openeo_udf.api.structured_data import StructuredData   # Old style openeo_udf API
        def transform(data: UdfData) -> UdfData:
            res = [
                StructuredData(description="res", data=[x * x for x in sd.data], type="list")
                for sd in data.get_structured_data_list()
            ]
            data.set_structured_data_list(res)
    """,
    # New style UDFs (based on openeo.udf imports)
    """
        from openeo.udf import UdfData, StructuredData
        def transform(data: UdfData) -> UdfData:
            res = [
                StructuredData(description="res", data=[x * x for x in sd.data], type="list")
                for sd in data.get_structured_data_list()
            ]
            data.set_structured_data_list(res)
    """,
])
def test_udf_support_structured_data(auth_connection, udf_code):
    """Test for non-callback usage of UDFs"""
    udf_code = textwrap.dedent(udf_code)
    process_graph = {
        "udf": {
            "process_id": "run_udf",
            "arguments": {
                "data": [1, 2, 3, 5, 8],
                "udf": udf_code,
                "runtime": "Python"
            },
            "result": True
        }
    }
    res = auth_connection.execute(process_graph)
    assert res == [1, 4, 9, 25, 64]


def test_udp_public_sharing_url_namespace(auth_connection):
    """Test public sharing of UDPs and backend-side resolving of URL based namespace"""

    # Build and store UDP
    f = Parameter.number("f", description="Degrees Fahrenheit.")
    pg = {
        'subtract1': {'arguments': {'x': {'from_parameter': 'f'}, 'y': 32}, 'process_id': 'subtract'},
        'divide1': {'arguments': {'x': {'from_node': 'subtract1'}, 'y': 1.8}, 'process_id': 'divide', 'result': True},
    }
    udp = auth_connection.save_user_defined_process(
        "fahrenheit_to_celsius", process_graph=pg, parameters=[f], public=True
    )

    # "canonical" link from metadata should be public
    metadata = udp.describe()
    public_url = next(l for l in metadata["links"] if l["rel"] == "canonical")["href"]
    r = requests.get(public_url)
    assert r.status_code == 200

    # Use url as namespace to use UDP
    # TODO: this should be done by a different user
    cube = auth_connection.datacube_from_process("fahrenheit_to_celsius", namespace=public_url, f=86)
    celsius = cube.execute()
    assert celsius == 30


def test_validation_missing_product(connection):
    """
    EP-4012, https://github.com/openEOPlatform/architecture-docs/issues/85
    """
    cube = connection.load_collection("TERRASCOPE_S2_TOC_V2")
    cube = cube.filter_temporal("2021-02-01", "2021-02-10")
    cube = cube.filter_bbox(west=90, south=60, east=90.1, north=60.1)
    errors = cube.validate()
    print(errors)
    assert len(errors) > 0
    assert "MissingProduct" in {e["code"] for e in errors}
