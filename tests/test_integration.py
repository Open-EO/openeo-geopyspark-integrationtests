import imghdr
import json
from pathlib import Path
import re
import time
from typing import Callable, Union

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import rasterio
import requests
import schema
from shapely.geometry import shape, Polygon

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
TEST_USER = "geopyspark-integrationtester"
TEST_PASSWORD = TEST_USER + "123"

POLYGON01 = Polygon(shell=[
    # bounding box (Dortmund): http://bboxfinder.com/#51.29289899553571,7.022705078125007,51.75432477678571,7.659912109375007
    [7.022705078125007, 51.75432477678571],
    [7.659912109375007, 51.74333844866071],
    [7.659912109375007, 51.29289899553571],
    [7.044677734375007, 51.31487165178571],
    [7.022705078125007, 51.75432477678571]
])

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
            "queue": "geoviewer"
        }

def test_health(connection):
    r = connection.get("/health")
    assert r.status_code == 200


def test_collections(connection):
    image_collections = connection.list_collections()
    product_ids = [entry.get("id") for entry in image_collections]
    assert "PROBAV_L3_S10_TOC_NDVI_333M" in product_ids


def test_terrascope_download_latlon(connection, tmp_path):
    s2_fapar = (
        connection.load_collection("TERRASCOPE_S2_NDVI_V2")
            # bounding box: http://bboxfinder.com/#51.197400,5.027000,51.221300,5.043800
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
            .filter_bbox(west=5.027, east=5.0438, south=51.1974, north=51.2213, crs="EPSG:4326")
    )
    out_file = tmp_path / "s2_fapar_latlon.geotiff"
    s2_fapar.download(out_file, format="GTIFF")
    assert out_file.exists()


def test_terrascope_download_webmerc(connection, tmp_path):
    s2_fapar = (
        connection.load_collection("TERRASCOPE_S2_NDVI_V2")
            .filter_temporal(["2018-08-06T00:00:00Z", "2018-08-06T00:00:00Z"])
            .filter_bbox(west=561864.7084, east=568853, south=6657846, north=6661080, crs="EPSG:3857")
    )
    out_file = tmp_path / "/tmp/s2_fapar_webmerc.geotiff"
    s2_fapar.download(out_file, format="geotiff")
    assert out_file.exists()


def test_aggregate_spatial_polygon(connection):
    timeseries = (
        connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal(start_date="2017-11-01", end_date="2017-11-21")
            .polygonal_mean_timeseries(POLYGON01)
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


def test_cog_synchronous(connection, tmp_path):
    cube = (
        connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal("2017-11-21", "2017-11-21")
            .filter_bbox(west=0, south=50, east=5, north=55, crs='EPSG:4326')
    )

    out_file = tmp_path / "cog.tiff"
    cube.download(out_file, format="GTIFF", options={"tiled": True})
    assert_geotiff_basics(out_file)
    assert_cog(out_file)


@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_cog_execute_batch(connection, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)
    cube = (
        connection
            .load_collection('PROBAV_L3_S10_TOC_NDVI_333M')
            .filter_temporal("2017-11-21", "2017-11-21")
            .filter_bbox(west=2, south=51, east=3, north=52, crs='EPSG:4326')
    )
    output_file = tmp_path / "result.tiff"
    job = cube.execute_batch(
        output_file, out_format="GTIFF", max_poll_interval=BATCH_JOB_POLL_INTERVAL,
        tiled=True, job_options=batch_default_options(driverMemoryOverhead="1G",driverMemory="1G"))
    assert [j["status"] for j in connection.list_jobs() if j['id'] == job.job_id] == ["finished"]
    assert_geotiff_basics(output_file)
    assert_cog(output_file)


def _poll_job_status(
        job: RESTJob, until: Callable = lambda s: s == "finished",
        max_polls: int = 180, sleep_max: int = BATCH_JOB_POLL_INTERVAL) -> str:
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


@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_basic(connection, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)
    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.polygonal_mean_timeseries(POLYGON01)

    job = timeseries.send_job(job_options=batch_default_options(driverMemory="1G",driverMemoryOverhead="512m"))
    assert job.job_id

    job.start_job()
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert status == "finished"

    assets = job.download_results(tmp_path)
    assert len(assets) == 1

    with open(list(assets.keys())[0]) as f:
        data = json.load(f)

    expected_dates = ["2017-11-01T00:00:00", "2017-11-11T00:00:00", "2017-11-21T00:00:00"]
    assert sorted(data.keys()) == sorted(expected_dates)
    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(data)


@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_execute_batch(connection, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)
    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.polygonal_mean_timeseries(POLYGON01)

    output_file = tmp_path / "ts.json"
    timeseries.execute_batch(output_file, max_poll_interval=BATCH_JOB_POLL_INTERVAL, job_options=batch_default_options(driverMemory="1G",driverMemoryOverhead="512m"))

    with output_file.open("r") as f:
        data = json.load(f)
    expected_dates = ["2017-11-01T00:00:00", "2017-11-11T00:00:00", "2017-11-21T00:00:00"]
    assert sorted(data.keys()) == sorted(expected_dates)
    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(data)


@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_cancel(connection, tmp_path):
    connection.authenticate_basic(TEST_USER, TEST_PASSWORD)

    cube = connection.load_collection("PROBAV_L3_S10_TOC_NDVI_333M").filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.polygonal_mean_timeseries(POLYGON01)

    job = timeseries.send_job(out_format="GTIFF", job_options=batch_default_options(driverMemory="512m",driverMemoryOverhead="512m"))
    assert job.job_id
    job.start_job()

    # await job running
    status = _poll_job_status(job, until=lambda s: s in ['running', 'canceled', 'finished', 'error'], )
    assert status == "running"

    # cancel it
    job.stop_job()
    print("stopped job")

    # await job canceled
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert status == "canceled"


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


@pytest.mark.skip(reason="SENTINEL1_GAMMA0_SENTINELHUB requires secret #EP-3050")
def test_ep3048_sentinel1_udf(connection):
    # http://bboxfinder.com/#-4.745000,-55.700000,-4.740000,-55.695000
    N, E, S, W = (-4.740, -55.695, -4.745, -55.7)
    polygon = Polygon(shell=[[W, N], [E, N], [E, S], [W, S]])

    udf_code = read_data('udfs/smooth_savitsky_golay.py')

    ts = (
        connection.load_collection("SENTINEL1_GAMMA0_SENTINELHUB")
            .filter_temporal(["2019-05-24T00:00:00Z", "2019-05-30T00:00:00Z"])
            .filter_bbox(north=N, east=E, south=S, west=W, crs="EPSG:4326")
            .filter_bands([0])
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
    if api_version >= "1.0.0":
        masked = cube.mask_polygon(POLYGON01)
    else:
        masked = cube.mask(polygon=POLYGON01)

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


def test_reduce_temporal_udf(connection, tmp_path):
    bbox = {
        "west": 6.8371137,
        "north": 50.5647147,
        "east": 6.8566699,
        "south": 50.560007,
        "crs": "EPSG:4326"
    }

    cube = (
        connection.load_collection("CGS_SENTINEL2_RADIOMETRY_V102_001")
            .filter_temporal("2017-03-10", "2017-03-30")
            .filter_bbox(**bbox)
    )

    udf_code = read_data("udfs/udf_temporal_slope.py")
    print(udf_code)

    trend = cube.reduce_temporal_udf(udf_code, runtime="Python", version="latest")

    output_file = tmp_path / "trend.tiff"
    trend.download(output_file, format="GTIFF")
    assert_geotiff_basics(output_file, expected_band_count=16)


@pytest.mark.requires_custom_processes
def test_custom_processes(connection):
    process_graph = {
        "foobar1": {
            "process_id": "foobar",
            "arguments": {"size": 123, "color": "green"},
            "result": True,
        }
    }
    res = connection.execute(process_graph)
    assert res == {
        "args": ["color", "size"],
        "msg": "hello world",
    }


@pytest.mark.parametrize(["cid", "expected_dates"], [
    (
            'S2_FAPAR_V102_WEBMERCATOR2',
            [
                '2017-11-01T00:00:00', '2017-11-02T00:00:00', '2017-11-04T00:00:00', '2017-11-06T00:00:00',
                '2017-11-07T00:00:00', '2017-11-09T00:00:00', '2017-11-11T00:00:00', '2017-11-12T00:00:00',
                '2017-11-14T00:00:00', '2017-11-16T00:00:00', '2017-11-17T00:00:00', '2017-11-19T00:00:00',
                '2017-11-21T00:00:00'
            ]
    ),
    (
            'PROBAV_L3_S10_TOC_NDVI_333M',
            ["2017-11-01T00:00:00", "2017-11-11T00:00:00", "2017-11-21T00:00:00"]
    ),
])
def test_polygonal_timeseries(connection, tmp_path, cid, expected_dates, api_version):
    expected_dates = sorted(expected_dates)
    polygon = POLYGON01
    bbox = _polygon_bbox(polygon)
    cube = (
        connection
            .load_collection(cid)
            .filter_temporal("2017-11-01", "2017-11-21")
            .filter_bbox(**bbox)
    )
    ts_mean = cube.polygonal_mean_timeseries(polygon).execute()
    print("mean", ts_mean)
    ts_median = cube.polygonal_median_timeseries(polygon).execute()
    print("median", ts_median)
    ts_sd = cube.polygonal_standarddeviation_timeseries(polygon).execute()
    print("sd", ts_sd)

    # TODO: see https://jira.vito.be/browse/EP-3434
    assert sorted(ts_mean.keys()) == expected_dates
    assert sorted(ts_median.keys()) == [d + "Z" for d in expected_dates]
    assert sorted(ts_sd.keys()) == [d + "Z" for d in expected_dates]

    # Only check for a subset of dates if there are a lot
    for date in expected_dates[::max(1, len(expected_dates) // 5)]:
        output_file = tmp_path / "ts_{d}.tiff".format(d=re.sub(r'[^0-9]', '', date))
        print("Evaluating date {d}, downloading to {p}".format(d=date, p=output_file))
        date_cube = connection.load_collection(cid).filter_temporal(date, date).filter_bbox(**bbox)
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
                assert ts_median[date + "Z"] == [[None] * band_count]
                assert ts_sd[date + "Z"] == [[None] * band_count]
            else:
                rtol = 0.02
                np.testing.assert_allclose(np.ma.mean(data, axis=(-1, -2)), ts_mean[date][0], rtol=rtol)
                np.testing.assert_allclose(np.ma.median(data, axis=(-1, -2)), ts_median[date + "Z"][0], rtol=rtol)
                np.testing.assert_allclose(np.ma.std(data, axis=(-1, -2)), ts_sd[date + "Z"][0], rtol=rtol)


def test_ndvi(connection, tmp_path):
    ndvi = connection.load_collection(
        "CGS_SENTINEL2_RADIOMETRY_V102_001",
        spatial_extent={"west": 5.027, "east": 5.0438, "south": 51.1974, "north": 51.2213},
        temporal_extent=["2020-04-05", "2020-04-05"]
    ).ndvi()

    output_tiff = tmp_path / "ndvi.tiff"
    ndvi.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)


def test_normalized_difference(connection, tmp_path):
    toc = connection.load_collection(
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


def test_udp_usage_simple(connection100, tmp_path):
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
            Parameter(name="data", description="A data cube.", schema={"type": "object", "subtype": "raster-cube"})
        ]
    )
    # Use UDP
    date = "2020-06-26"
    cube = (
        connection100.load_collection("CGS_SENTINEL2_RADIOMETRY_V102_001")
            .filter_bands(["red", "green", "blue", "nir"])
            .filter_temporal(date, date)
            .filter_bbox(**BBOX_MOL)
            .process("flatten_bands", arguments={"data": THIS})
    )
    output_tiff = tmp_path / "mol.tiff"
    cube.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=1)
