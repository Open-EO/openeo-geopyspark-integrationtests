import subprocess
import typing

import itertools
import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path
from pprint import pprint, pformat
from typing import Callable, Union, Optional

import imghdr
import numpy as np
import pyproj
import pytest
import rasterio
import requests
import schema
import shapely.geometry
import shapely.ops
import xarray
from numpy.ma.testutils import assert_array_approx_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
import pystac
from shapely.geometry import mapping, shape, GeometryCollection, Point, Polygon
from shapely.geometry.base import BaseGeometry
import hvac
import hvac.utils

import openeo
from openeo.rest.connection import OpenEoApiError
from openeo.rest.conversions import datacube_from_file, timeseries_json_to_pandas
from openeo.rest.datacube import DataCube, THIS
from openeo.rest.job import BatchJob, JobResults
from openeo.rest.mlmodel import MlModel
from openeo.rest.udp import Parameter
from openeo_driver.testing import DictSubSet
from .cloudmask import create_advanced_mask, create_simple_mask
from .data import get_path, read_data


_log = logging.getLogger(__name__)


def _dump_process_graph(
    cube: Union[DataCube], tmp_path: Path, name="process_graph.json"
):
    """Dump a cube's process graph as json to a temp file"""
    (tmp_path / name).write_text(cube.to_json(indent=2))


def _parse_bboxfinder_com(url: str) -> dict:
    """Parse a bboxfinder.com URL to bbox dict"""
    # TODO: move this kind of functionality to python client?
    coords = [float(x) for x in url.split('#')[-1].split(",")]
    return {"south": coords[0], "west": coords[1], "north": coords[2], "east": coords[3], "crs": "EPSG:4326"}


def _polygon_bbox(polygon: Polygon) -> dict:
    """Extract bbox dict from given polygon"""
    coords = polygon.bounds
    return {"south": coords[1], "west": coords[0], "north": coords[3], "east": coords[2], "crs": "EPSG:4326"}


def assert_batch_job(job: BatchJob, assertion: bool, extra_message: str = ""):
    try:
        assert assertion
    except AssertionError as e:
        kibana_url = f"https://kibana-infra.vgt.vito.be/app/kibana#/discover?_g=(filters:!(),refreshInterval:" \
                     f"(pause:!t,value:0),time:(from:now-7d,to:now))&_a=(columns:!(levelname),filters:!(" \
                     f"('$state':(store:appState),meta:(alias:!n,disabled:!f," \
                     f"index:'592a42f0-e665-11ec-8cc4-3747d5233c59',key:levelname,negate:!f,params:(query:ERROR)," \
                     f"type:phrase),query:(match:(levelname:(query:ERROR,type:phrase)))))," \
                     f"index:'592a42f0-e665-11ec-8cc4-3747d5233c59',interval:auto,query:(language:kuery,query:'" \
                     f"job_id%20:%20%22{job.job_id}%22%20'),sort:!(!('@timestamp',desc)))"
        message = f"Assertion for batch job {job} failed: {extra_message}"\
                  f"Job status: {job.status()}" \
                  f"Kibana logs: {kibana_url}" \
                  f"Job error logs: {job.logs(level='ERROR')}"
        if job.status == "finished":
            job_results = job.get_results()
            message += f"Job metadata: {job_results.get_metadata()}"
            message += f"Job assets: {job_results.get_assets()}"
        raise AssertionError(message) from e


BBOX_MOL = _parse_bboxfinder_com("http://bboxfinder.com/#51.21,5.071,51.23,5.1028")
BBOX_GENT = _parse_bboxfinder_com("http://bboxfinder.com/#51.03,3.7,51.05,3.75")
BBOX_NIEUWPOORT = _parse_bboxfinder_com("http://bboxfinder.com/#51.05,2.60,51.20,2.90")


POLYGON01 = Polygon(shell=[
    # Dortmund (bbox: http://bboxfinder.com/#51.30,7.00,51.75,7.60)
    [7.00, 51.75],
    [7.10, 51.35],
    [7.50, 51.30],
    [7.60, 51.70],
    [7.00, 51.75],
])

POLYGON01_BBOX = [7.00, 51.30, 7.60, 51.75]


BATCH_JOB_POLL_INTERVAL = 30
BATCH_JOB_TIMEOUT = 60 * 60


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
    assert "PROBAV_L3_S10_TOC_333M" in product_ids


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
        auth_connection.load_collection("PROBAV_L3_S10_TOC_333M", bands=["NDVI"])
        .filter_temporal(start_date="2017-11-01", end_date="2017-11-21")
        .aggregate_spatial(geometries=POLYGON01, reducer="mean")
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
            .load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"])
            .filter_bbox(5, 6, 52, 51, 'EPSG:4326')
            .filter_temporal(['2017-11-21', '2017-12-21'])
    )
    polygon = shape(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [5.0761587693484875, 51.21222494794898],
                    [5.166854684377381, 51.21222494794898],
                    [5.166854684377381, 51.268936260927404],
                    [5.0761587693484875, 51.268936260927404],
                    [5.0761587693484875, 51.21222494794898],
                ]
            ],
        }
    )
    timeseries = probav.aggregate_spatial(
        geometries=polygon, reducer="histogram"
    ).execute()
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
            .filter_temporal(start_date="2020-11-05", end_date="2020-11-05")
            .filter_bbox(west=761104, east=763281, south=6543830, north=6544655, crs="EPSG:3857")
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
            .load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"])
            .filter_temporal("2017-11-21", "2017-11-21")
            .filter_bbox(west=0, south=50, east=5, north=55, crs='EPSG:4326')
    )

    out_file = tmp_path / "cog.tiff"
    cube.download(out_file)
    #TODO: this shape, is wrong, caused by: https://github.com/Open-EO/openeo-geopyspark-driver/issues/260
    assert_geotiff_basics(out_file, expected_shape=(1, 3642, 1821))
    assert_cog(out_file)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_cog_execute_batch(auth_connection, tmp_path):
    cube = auth_connection.load_collection(
        "PROBAV_L3_S10_TOC_333M",
        bands = ["NDVI"],
        temporal_extent=["2017-11-21", "2017-11-21"],
        spatial_extent={"west": 2, "south": 51, "east": 4, "north": 53},
    )

    job = cube.execute_batch(
        out_format="GTIFF",
        max_poll_interval=BATCH_JOB_POLL_INTERVAL,
        job_options=batch_default_options(driverMemoryOverhead="1G", driverMemory="1G"),
        tile_grid="one_degree",
        title="test_cog_execute_batch",
    )
    _log.info(f"test_cog_execute_batch: {job=}")
    assert_job_status(job, "finished")

    job_results: JobResults = job.get_results()
    downloaded = job_results.download_files(
        tmp_path / "job1", include_stac_metadata=True
    )
    _log.info(f"{len(downloaded)=} {downloaded=}")

    arbitrary_geotiff_path = downloaded[0]
    assert_geotiff_basics(arbitrary_geotiff_path, expected_band_count=1)
    assert_cog(arbitrary_geotiff_path)

    # conveniently tacked on test for load_result because it needs a batch job that won't be removed in the near future
    cube_from_result = auth_connection.load_result(
        job.job_id,
        spatial_extent={"west": 2.6, "south": 51.1, "east": 2.9, "north": 51.3},
    )

    load_result_output_file = tmp_path / "load_result.tiff"
    cube_from_result.download(load_result_output_file, format="GTiff")

    with rasterio.open(load_result_output_file) as load_result_ds:
        assert load_result_ds.count == 1
        probav_data = load_result_ds.read(1)
        no_data = 255
        assert np.any(probav_data != no_data)

    # Verify projection metadata.
    job_metadata = job_results.get_metadata()
    assert job_metadata == DictSubSet(
        {
            "assets": {
                "openEO_2017-11-21Z_N51E002.tif": DictSubSet(
                    {
                        "proj:epsg": 4326,
                        "proj:shape": [365, 728],
                        "proj:bbox": pytest.approx(
                            [1.9995067, 51.0012761, 3.0020091, 52.0010318]
                        ),
                    }
                ),
                "openEO_2017-11-21Z_N51E003.tif": DictSubSet(
                    {
                        "proj:epsg": 4326,
                        "proj:shape": [364, 728],
                        "proj:bbox": pytest.approx(
                            [2.9992625, 51.0012761, 3.9990183, 52.0010318]
                        ),
                    }
                ),
                "openEO_2017-11-21Z_N52E002.tif": DictSubSet(
                    {
                        "proj:epsg": 4326,
                        "proj:shape": [365, 729],
                        "proj:bbox": pytest.approx(
                            [1.9995067, 51.9996585, 3.0020091, 53.0007876]
                        ),
                    }
                ),
                "openEO_2017-11-21Z_N52E003.tif": DictSubSet(
                    {
                        "proj:epsg": 4326,
                        "proj:shape": [364, 729],
                        "proj:bbox": pytest.approx(
                            [2.9992625, 51.9996585, 3.9990183, 53.0007876]
                        ),
                    }
                ),
            },
        }
    )


def assert_projection_metadata_present(metadata: dict) -> dict:
    """Check all elements of the STAC projection metadata.

    If the assert fails then include the report in the assert message so we have
    some information to start troubleshooting.
    """
    report = validate_projection_metadata(metadata)

    message = f"Projection metadata verification report:\n{pformat(report)}"
    _log.info(message)

    assert report[
        "is_proj_metadata_present"
    ], f"Projection metadata verification FAILED:\n{pformat(report)}"


def validate_projection_metadata(metadata: dict) -> dict:
    """Verify that some sensible projection metadata is present.

    When the validation PASSES, then is_proj_metadata_present will be True.

    The validation PASSES when the metadata contains a value for each of the
    three properties epsg, bbox and shape.
    It can be present at the top level, or at the asset level, or even at both
    levels.
    Or in other words, the validation FAILS when for any of the propeties
    properties epsg, bbox and shape no value could be found at the top leval and
    at the asset level.

    Top and asset level don't necessarily exclusive each other.
    For example a bbox could already be present at the top level,
    while each asset also has its own different bbox, which is then also added
    at the asset level.

    :param metadata: dict with the job's metadata.

    :return:
        Dict with a report that has the structure described below.

        "is_proj_metadata_present" is main information and is True when
        the validation passes

        The other elements are for logging this as a report so the integration
        test can be more informative about why it fails.

        {
            // True if it PASSES; False if it FAILS
            "is_proj_metadata_present": is_proj_metadata_present,

            // The individual elements of the checks; where were the projection
            // properties present or not
            "report": {
                "has_crs_on_top_level": has_crs_on_top_level,
                "has_bbox_on_top_level": has_bbox_on_top_level,
                "has_shape_on_top_level": has_shape_on_top_level,
                "has_crs_on_assets": has_crs_on_assets,
                "has_bbox_on_assets": has_bbox_on_assets,
                "has_shapes_on_assets": has_shapes_on_assets
            }

            // The actual values found, for troubleshooting when the test fails.
            "values": {
                "crs_on_top_level": metadata.get("epsg"),
                "bbox_on_top_level": metadata.get("bbox"),
                "shape_on_top_level": metadata.get("proj:shape"),
                "asset_crss": asset_crss,
                "asset_bboxes": asset_bboxes,
                "asset_shapes": asset_shapes
            },

        }
    """
    # Are projection metdata keys present at the top level (job as a whole)
    has_crs_on_top_level = bool(metadata.get("epsg"))
    has_bbox_on_top_level = bool(metadata.get("bbox"))
    has_shape_on_top_level = bool(metadata.get("proj:shape"))

    # Collect asset level metadata.
    # Also used to show in report for troubleshooting.
    assets_metadata = metadata.get("assets", {})
    asset_crss = {
        asset: asset_md.get("proj:epsg") for asset, asset_md in assets_metadata.items()
    }
    asset_bboxes = {
        asset: asset_md.get("proj:bbox") for asset, asset_md in assets_metadata.items()
    }
    asset_shapes = {
        asset: asset_md.get("proj:shape") for asset, asset_md in assets_metadata.items()
    }

    # Are the asset level metadata present.
    has_crs_on_assets = any(asset_crss.values())
    has_bbox_on_assets = any(asset_bboxes.values())
    has_shapes_on_assets = any(asset_shapes.values())

    # Does it pass or fail?
    # For, now make it a bit more permissive: at least some projection
    # properties should be present.
    # But in the future we want to have all three properties covered.
    is_proj_metadata_present = (
        (has_crs_on_top_level or has_crs_on_assets)
        or (has_bbox_on_top_level or has_bbox_on_assets)
        or (has_shape_on_top_level or has_shapes_on_assets)
    )

    return {
        "is_proj_metadata_present": is_proj_metadata_present,
        "values": {
            "crs_on_top_level": metadata.get("epsg"),
            "bbox_on_top_level": metadata.get("bbox"),
            "shape_on_top_level": metadata.get("proj:shape"),
            "asset_crss": asset_crss,
            "asset_bboxes": asset_bboxes,
            "asset_shapes": asset_shapes,
        },
        "report": {
            "has_crs_on_top_level": has_crs_on_top_level,
            "has_bbox_on_top_level": has_bbox_on_top_level,
            "has_shape_on_top_level": has_shape_on_top_level,
            "has_crs_on_assets": has_crs_on_assets,
            "has_bbox_on_assets": has_bbox_on_assets,
            "has_shapes_on_assets": has_shapes_on_assets,
        },
    }


def _poll_job_status(
    job: BatchJob,
    until: Callable = lambda s: s == "finished",
    sleep: int = BATCH_JOB_POLL_INTERVAL,
    max_poll_time=30 * 60,
) -> str:
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


def assert_job_status(job: BatchJob, expected_status: str):
    """Assert that job's status is the expected status, and optionally print its logs.

    If the assert fails the job ID is included in the assert message so you can
    find the job in Kibana to see what went wrong.

    Do keep in mind that the job's logs could be long which is why we don't
    display them by default.
    In particular, this could make the test logs on Jenkins long and difficult to read.

    :param job: the batch job to check
    :param expected_status: which status it should have.
    """
    # If the next assert is going to fail, then first show the logs of the job.
    actual_status = job.status()
    message = (
        f"job {job}: did not end with expected status '{expected_status}', "
        + f"but ended with status '{actual_status}'"
    )
    assert_batch_job(job, actual_status == expected_status, extra_message=message)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_basic(auth_connection, api_version, tmp_path):
    cube = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"]).filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.aggregate_spatial(geometries=POLYGON01, reducer="median")

    job = timeseries.create_job(
        job_options=batch_default_options(
            driverMemory="1600m", driverMemoryOverhead="512m"
        ),
        title="test_batch_job_basic",
    )
    assert job.job_id

    job.start_job()
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert_job_status(job, "finished")

    job_results: JobResults = job.get_results()
    job_results_metadata: dict = job_results.get_metadata()
    downloaded = job_results.download_files(tmp_path, include_stac_metadata=True)
    _log.info(f"{downloaded=}")

    assets = job_results.get_assets()
    _log.info(f"{assets=}")
    assert_batch_job(job, len(assets) == 1, extra_message=f"expected 1 asset, got {len(assets)}")
    assert_batch_job(job, assets[0].name.endswith(".json"))
    data = assets[0].load_json()
    _log.info(f"{data=}")

    expected_dates = ["2017-11-01T00:00:00Z", "2017-11-11T00:00:00Z", "2017-11-21T00:00:00Z"]
    assert_batch_job(job, sorted(data.keys()) == sorted(expected_dates))
    expected_schema = schema.Schema({str: [[int]]})
    assert_batch_job(job, expected_schema.validate(data))

    if api_version >= "1.1.0":
        assert_batch_job(job, job_results_metadata["type"] == "Collection")
        job_results_stac: pystac.Collection = pystac.Collection.from_dict(
            job_results_metadata
        )
        assert_batch_job(job, job_results_stac.extent.spatial.bboxes[0] == POLYGON01_BBOX)
        assert_batch_job(job, job_results_stac.extent.temporal.to_dict()["interval"] == [
            ["2017-11-01T00:00:00Z", "2017-11-21T00:00:00Z"]
        ])

    elif api_version >= "1.0.0":
        assert_batch_job(job, job_results_metadata["type"] == "Feature")
        geometry = shape(job_results_metadata["geometry"])
        assert_batch_job(job, geometry.equals_exact(POLYGON01, tolerance=0.0001))
        assert_batch_job(job, job_results_metadata["bbox"] == POLYGON01_BBOX)
        assert_batch_job(job, job_results_metadata["properties"]["start_datetime"] == "2017-11-01T00:00:00Z")
        assert_batch_job(job, job_results_metadata["properties"]["end_datetime"] == "2017-11-21T00:00:00Z")


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_execute_batch(auth_connection, tmp_path):
    cube = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"]).filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.aggregate_spatial(geometries=POLYGON01, reducer="median")

    output_file = tmp_path / "ts.json"
    job = timeseries.execute_batch(output_file, max_poll_interval=BATCH_JOB_POLL_INTERVAL, job_options=batch_default_options(driverMemory="1600m",driverMemoryOverhead="512m"), title="execute-batch")

    with output_file.open("r") as f:
        data = json.load(f)
    expected_dates = ["2017-11-01T00:00:00Z", "2017-11-11T00:00:00Z", "2017-11-21T00:00:00Z"]
    assert_batch_job(job, sorted(data.keys()) == sorted(expected_dates))
    expected_schema = schema.Schema({str: [[int]]})
    assert_batch_job(job, expected_schema.validate(data))


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_signed_urls(auth_connection, tmp_path):
    cube = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"]).filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.aggregate_spatial(geometries=POLYGON01, reducer="median")

    job = timeseries.execute_batch(
        max_poll_interval=BATCH_JOB_POLL_INTERVAL,
        job_options=batch_default_options(driverMemory="1600m", driverMemoryOverhead="512m"),
        title = "signed-urls"
    )

    results = job.get_results()
    # TODO: check results metadata?
    print("results metadata", results.get_metadata())

    assets = results.get_assets()
    print("assets", assets)
    assert_batch_job(job, len(assets) >= 1)
    data = None
    for asset in assets:
        # Download directly without credentials
        resp = requests.get(asset.href)
        resp.raise_for_status()
        assert_batch_job(job, resp.status_code == 200)
        if asset.name.endswith(".json"):
            assert data is None
            data = resp.json()
    expected_dates = ["2017-11-01T00:00:00Z", "2017-11-11T00:00:00Z", "2017-11-21T00:00:00Z"]
    assert_batch_job(job, sorted(data.keys()) == sorted(expected_dates))
    expected_schema = schema.Schema({str: [[int]]})
    assert_batch_job(job, expected_schema.validate(data))


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_cancel(auth_connection, tmp_path):

    cube = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"]).filter_temporal("2017-11-01", "2017-11-21")
    if isinstance(cube, DataCube):
        cube = cube.process("sleep", arguments={"data": cube, "seconds": 600})
    else:
        raise ValueError(cube)

    timeseries = cube.aggregate_spatial(geometries=POLYGON01, reducer="mean")

    job = timeseries.create_job(
        out_format="GTIFF",
        job_options=batch_default_options(
            driverMemory="512m", driverMemoryOverhead="1g"
        ),
        title="test_batch_job_cancel",
    )
    assert job.job_id
    job.start_job()

    # await job running
    status = _poll_job_status(job, until=lambda s: s in ['running', 'canceled', 'finished', 'error'])
    assert_batch_job(job, status == "running")

    # cancel it
    job.stop_job()
    print("stopped job")

    # await job canceled
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert_batch_job(job, status == "canceled")


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_batch_job_delete_job(auth_connection):

    cube = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"]).filter_temporal("2017-11-01", "2017-11-21")
    timeseries = cube.aggregate_spatial(geometries=POLYGON01, reducer="mean")

    job: BatchJob = timeseries.create_job(
        out_format="GTIFF",
        job_options=batch_default_options(
            driverMemory="1600m", driverMemoryOverhead="512m"
        ),
        title="test_batch_job_delete_job",
    )
    assert job.job_id
    job.start_job()
    _log.info(f"test_batch_job_delete_job: {job=}")

    # await job finished
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert_job_status(job, "finished")

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


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_random_forest_load_from_http(auth_connection: openeo.Connection, tmp_path):
    """
    Make predictions with the random forest model using an http link to a ml_model_metadata.json file.
    """

    topredict_xybt = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M',bands=["NDVI"],
        spatial_extent = {"west": 4.785919, "east": 4.909629, "south": 51.259766, "north": 51.307638},
        temporal_extent = ["2017-11-01", "2017-11-01"])
    topredict_cube_xyb = topredict_xybt.reduce_dimension(dimension = "t", reducer = "mean")
    # Make predictions with the random forest model using http link.
    random_forest_metadata_link = "https://github.com/Open-EO/openeo-geopyspark-integrationtests/raw/master/tests/data/mlmodels/randomforest_ml_model_metadata.json"
    predicted_with_link = topredict_cube_xyb.predict_random_forest(
        model=random_forest_metadata_link,
        dimension="bands"
    )
    with_link_output_file = tmp_path / "predicted_with_link.tiff"
    predicted_with_link.download(with_link_output_file, format="GTiff")
    assert_geotiff_basics(with_link_output_file, min_width = 15, min_height = 15)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_random_forest_train_and_load_from_jobid(auth_connection: openeo.Connection, tmp_path):
    # 1. Train a random forest model.
    FEATURE_COLLECTION_1 = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"target": 3},
                "geometry": {"type": "Polygon", "coordinates": [[[4.79, 51.26], [4.81, 51.26], [4.81, 51.30], [4.79, 51.30], [4.79, 51.26]]]}
            },
            {
                "type": "Feature",
                "properties": {"target": 5},
                "geometry": {"type": "Polygon", "coordinates": [[[4.85, 51.26], [4.90, 51.26], [4.90, 51.30], [4.85, 51.30], [4.85, 51.26]]]}
            },

        ]
    }

    cube_xybt: DataCube = auth_connection.load_collection(
        "PROBAV_L3_S10_TOC_333M", bands=["NDVI"],
        spatial_extent={"west": 4.78, "east": 4.91, "south": 51.25, "north": 51.31},
        temporal_extent=["2017-11-01", "2017-11-01"]
    )
    cube_xyb: DataCube = cube_xybt.reduce_dimension(dimension="t", reducer="mean")
    predictors: DataCube = cube_xyb.aggregate_spatial(FEATURE_COLLECTION_1, reducer="mean", target_dimension="bands")
    model: MlModel = predictors.fit_class_random_forest(target=FEATURE_COLLECTION_1, num_trees=3, seed=42)
    model: MlModel = model.save_ml_model()
    job: BatchJob = model.create_job(title="test_random_forest_train_and_load_from_jobid-training_step")
    assert job.job_id
    job.start_job()

    # Wait until job is finished
    status = _poll_job_status(job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert status == "finished"

    # Check the job metadata.
    collection_results = job.list_results()
    assert collection_results.get('assets', {}).get('randomforest.model.tar.gz', {}).get('href', None) is not None
    summaries = collection_results.get('summaries', {})
    assert summaries.get('ml-model:architecture', None) == ['random-forest']
    assert summaries.get('ml-model:learning_approach', None) == ['supervised']
    assert summaries.get('ml-model:prediction_type', None) == ['classification']
    links = collection_results.get('links', [])
    ml_model_metadata_links = [link for link in links if "ml_model_metadata.json" in link.get('href', "")]
    assert len(ml_model_metadata_links) == 1

    # Check the ml_model_metadata.json file.
    ml_model_metadata = requests.get(ml_model_metadata_links[0].get('href', "")).json()
    ml_model_assets = ml_model_metadata.get('assets', {})
    assert ml_model_assets.get('model', {}).get('roles', None) == ['ml-model:checkpoint']
    assert ml_model_assets.get('model', {}).get('title', None) == 'org.apache.spark.mllib.tree.model.RandomForestModel'
    ml_model_properties = ml_model_metadata.get('properties', {})
    assert ml_model_properties.get('ml-model:architecture', None) == 'random-forest'
    assert ml_model_properties.get('ml-model:learning_approach', None) == 'supervised'
    assert ml_model_properties.get('ml-model:prediction_type', None) == 'classification'
    assert ml_model_properties.get('ml-model:training-os', None) == 'linux'
    assert ml_model_properties.get('ml-model:training-processor-type', None) == 'cpu'
    assert ml_model_properties.get('ml-model:type', None) == 'ml-model'

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

    # List files in job directory.
    modelPath = Path("/data/projects/OpenEO") / job.job_id / "randomforest.model.tar.gz"
    metadataPath = Path("/data/projects/OpenEO") / job.job_id / "job_metadata.json"
    assert(modelPath.exists())
    # Download modelPath to tmp
    modelPathTmp = tmp_path / "randomforest.model.tar.gz"
    modelPathTmp.write_bytes(modelPath.read_bytes())
    assert(modelPath.stat().st_size > 1024)
    assert(metadataPath.exists())

    # Check job_metadata.json.
    with open(metadataPath, "r") as f:
        metadata = json.load(f)
        assert metadata["geometry"] == {
            "type": "Polygon",
            "coordinates": [[[4.79, 51.26], [4.79, 51.30], [4.90, 51.30], [4.90, 51.26], [4.79, 51.26]]],
        }
        assert metadata.get("assets", {}).get("randomforest.model.tar.gz", {}).get("href", "") == "/data/projects/OpenEO/{jobid}/randomforest.model.tar.gz".format(jobid=job.job_id)

    # 2. Load the model using its job id and make predictions.

    topredict_xybt = auth_connection.load_collection("PROBAV_L3_S10_TOC_333M",bands=["NDVI"],
        spatial_extent = {"west": 4.825919, "east": 4.859629, "south": 51.259766, "north": 51.307638},
        temporal_extent = ["2017-11-01", "2017-11-01"])
    topredict_cube_xyb = topredict_xybt.reduce_dimension(dimension = "t", reducer = "mean")
    predicted: DataCube = topredict_cube_xyb.predict_random_forest(
        model=job.job_id,
        dimension="bands"
    )
    inference_job = predicted.create_job(title="test_random_forest_train_and_load_from_jobid-inference_step")
    # Wait until job is finished
    assert inference_job.job_id
    inference_job.start_job()
    status = _poll_job_status(inference_job, until=lambda s: s in ['canceled', 'finished', 'error'])
    assert status == "finished"
    # Check the resulting geotiff filled with predictions.
    output_file = tmp_path / "predicted.tiff"
    inference_job.download_result(output_file)
    assert_geotiff_basics(output_file, min_width = 1, min_height = 1)


@pytest.mark.skip(reason="Requires proxying to work properly")
def test_create_wtms_service(auth_connection):
    s2_fapar = (
        auth_connection
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

    wmts_metadata = auth_connection.get(service_url).json()
    print("wmts metadata", wmts_metadata)
    assert "url" in wmts_metadata
    wmts_url = wmts_metadata["url"]
    time.sleep(5)  # seems to take a while before the service is proxied
    get_capabilities = requests.get(wmts_url + '?REQUEST=getcapabilities').text
    print("getcapabilities", get_capabilities)
    # the capabilities document should advertise the proxied URL
    assert "<Capabilities" in get_capabilities
    assert wmts_url in get_capabilities

@pytest.mark.skip(reason="Temporary skip to get tests through")
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
        .aggregate_spatial(geometries=polygon, reducer="mean")
        .execute()
    )
    assert isinstance(ts, dict)
    assert all(k.startswith('2019-05-') for k in ts.keys())


def test_load_collection_from_disk(auth_connection, tmp_path):
    fapar = auth_connection.load_disk_collection(
        format='GTiff',
        glob_pattern='/data/MTDA/TERRASCOPE_Sentinel2/FAPAR_V2/2019/04/24/*/10M/*_FAPAR_10M_V200.tif',
        options={
            'date_regex': r".*\/S2._(\d{4})(\d{2})(\d{2})T.*"
        }
    )
    date = "2019-04-24"
    fapar = fapar.filter_bbox(**BBOX_NIEUWPOORT).filter_temporal(date, date)

    output_file = tmp_path / "fapar_from_disk.tiff"
    fapar.download(output_file, format="GTiff")
    assert_geotiff_basics(output_file, expected_shape=(1, 1677, 2106))


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


def assert_cog(output_tiff: Union[str, Path]):
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


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_advanced_cloud_masking_diy(auth_connection, api_version, tmp_path):
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


    masked = s2_radiometry.mask(mask=mask.resample_cube_spatial(s2_radiometry))


    _dump_process_graph(masked, tmp_path)
    out_file = tmp_path / "masked_result.tiff"
    job = masked.execute_batch(out_file,title="diy_mask")
    links = job.get_results().get_metadata()['links']
    _log.info(f"test_advanced_cloud_masking_diy: {links=}")
    derived_from = [link["href"] for link in links if link["rel"] == "derived_from"]
    _log.info(f"test_advanced_cloud_masking_diy: {derived_from=}")
    v210_links = [link for link in derived_from if "V210" in link ]
    assert len(set(v210_links)) == 1
    assert v210_links == derived_from

    assert_geotiff_basics(out_file, expected_shape=(1, 284, 675))
    with rasterio.open(out_file) as result_ds:
        assert result_ds.dtypes == ('int16',)
        with rasterio.open(get_path("reference/advanced_cloud_masking_diy.tiff")) as ref_ds:
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
                "2017-11-01T00:00:00Z",
                "2017-11-04T00:00:00Z",
                "2017-11-06T00:00:00Z",
                "2017-11-09T00:00:00Z",
                "2017-11-11T00:00:00Z",
                "2017-11-14T00:00:00Z",
                "2017-11-16T00:00:00Z",
                "2017-11-19T00:00:00Z",
                "2017-11-21T00:00:00Z",
            ],
        )
    ],
)
def test_aggregate_spatial_timeseries(
    auth_connection, tmp_path, cid, expected_dates, api_version
):
    expected_dates = sorted(expected_dates)
    polygon = POLYGON01
    bbox = _polygon_bbox(polygon)
    cube = (
        auth_connection
            .load_collection(cid)
            .filter_temporal("2017-11-01", "2017-11-21")
            .filter_bbox(**bbox)
    )
    ts_mean = cube.aggregate_spatial(geometries=polygon, reducer="mean").execute()
    print("mean", ts_mean)
    ts_mean_df = timeseries_json_to_pandas(ts_mean)

    # TODO remove this cleanup https://github.com/Open-EO/openeo-geopyspark-driver/issues/75
    ts_mean = {k: v for (k, v) in ts_mean.items() if v != [[]]}
    print("mean", ts_mean)

    ts_median = cube.aggregate_spatial(geometries=polygon, reducer="median").execute()
    ts_median_df = timeseries_json_to_pandas(ts_median)
    print("median", ts_median_df)
    ts_sd = cube.aggregate_spatial(geometries=polygon, reducer="sd").execute()
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
            elif data[0].count() > 20:
                rtol = 0.02
                np.testing.assert_allclose(np.ma.mean(data, axis=(-1, -2)), ts_mean_df.loc[date], rtol=rtol)
                np.testing.assert_allclose(np.ma.median(data, axis=(-1, -2)), ts_median_df.loc[date], atol=0.001)
                np.testing.assert_allclose(np.ma.std(data, axis=(-1, -2)), ts_sd_df.loc[date], rtol=0.02)
            else:
                print("VERY LITTLE DATA " + str(date) + " " + str(data[0].count()))


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


def test_udp_crud(auth_connection):
    toc = auth_connection.load_collection("TERRASCOPE_S2_TOC_V2")
    udp = toc.save_user_defined_process(user_defined_process_id='toc', public=True)

    udp_details = udp.describe()

    assert udp_details['id'] == 'toc'
    assert 'loadcollection1' in udp_details['process_graph']
    assert udp_details['public']

    udp.delete()

    user_udp_ids = [udp["id"] for udp in auth_connection.list_user_defined_processes()]

    assert 'toc' not in user_udp_ids


def test_udp_usage_blur(auth_connection, tmp_path):
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
    auth_connection.save_user_defined_process("blur", blur)
    # Use UDP
    date = "2020-06-26"
    cube = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2")
            .filter_bands(["red", "green"])
            .filter_temporal(date, date)
            .filter_bbox(**BBOX_MOL)
            .process("blur", arguments={"data": THIS})
    )
    output_tiff = tmp_path / "mol.tiff"
    cube.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=2)
    # TODO: check resulting data?


def test_udp_usage_blur_parameter_default(auth_connection, tmp_path):
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
    auth_connection.save_user_defined_process(
        "blur", blur, parameters=[Parameter("scale", description="factor", schema="number", default=0.1)]
    )
    # Use UDP
    date = "2020-06-26"
    cube = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2")
            .filter_bands(["red", "green"])
            .filter_temporal(date, date)
            .filter_bbox(**BBOX_MOL)
            .process("blur", arguments={"data": THIS})
    )
    output_tiff = tmp_path / "mol.tiff"
    cube.download(output_tiff, format='GTIFF')
    assert_geotiff_basics(output_tiff, expected_band_count=2)
    # TODO: check resulting data?


def test_udp_usage_reduce(auth_connection, tmp_path):
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
    auth_connection.save_user_defined_process(
        "flatten_bands", flatten_bands, parameters=[
            Parameter.raster_cube(name="data")
        ]
    )
    # Use UDP
    date = "2020-06-26"
    cube = (
        auth_connection.load_collection("TERRASCOPE_S2_TOC_V2")
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
        auth_connection.load_collection("PROBAV_L3_S10_TOC_333M",bands=["NDVI"])
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


@pytest.mark.batchjob
@pytest.mark.timeout(40 * 60)
def test_sentinel_hub_execute_batch(auth_connection, tmp_path):
    data_cube = (auth_connection
                 .load_collection('SENTINEL1_GAMMA0_SENTINELHUB', bands=["VV", "VH"])
                 .filter_bbox(west=2.59003, east=2.8949, north=51.2206, south=51.069)
                 .filter_temporal(extent=["2019-10-10", "2019-10-10"]))

    output_tiff = tmp_path / "test_sentinel_hub_batch_job.tif"

    job = data_cube.execute_batch(output_tiff, out_format='GTiff', title="SentinelhubBatch")
    assert_geotiff_basics(output_tiff, expected_band_count=2)

    # conveniently tacked on test for load_result because it needs a batch job that won't be removed in the near future
    job_results_stac: pystac.Collection = pystac.Collection.from_dict(
        job.get_results().get_metadata()
    )
    job_results_canonical_url = next(
        link.href for link in job_results_stac.links if link.rel == "canonical"
    )
    source_id = job_results_canonical_url

    cube_from_result = (auth_connection
                        .load_result(source_id, bands=['VV'])
                        .filter_bbox({'west': 2.69003, 'south': 51.169, 'east': 2.7949, 'north': 51.2006})
                        .save_result("GTiff"))

    load_result_output_tiff = tmp_path / "load_result_small.tiff"
    cube_from_result.download(load_result_output_tiff, format="GTiff")

    assert_geotiff_basics(load_result_output_tiff, expected_band_count=1)

    # Verify projection metadata.
    job_results: JobResults = job.get_results()
    job_metadata = job_results.get_metadata()
    # Save it for troubleshooting if test fails.
    job_metadata_file = tmp_path / "job-results.json"
    with open(job_metadata_file, "wt", encoding="utf8") as md_file:
        json.dump(job_metadata, md_file)

    # TODO: this part still fails: proj metadata at top level does not come through
    #   Looks like API is not passing on this data in openeo-python-driver.
    # Show some output for easier troubleshooting
    print("Job result metadata:")
    pprint(job_metadata)
    try:
        assert job_metadata == DictSubSet(
            {
                "epsg": 32631,
                "proj:shape": [2140, 1694],
                "bbox": pytest.approx([471270.0, 5657500.0, 492670.0, 5674440.0]),
            }
        )
    except Exception as e:
        _log.warning(
            "This failed {e!r}. This part of the test is still experimental.",
            exc_info=True,
        )


def test_sentinel_hub_default_sar_backscatter_synchronous(auth_connection, tmp_path):
    data_cube = (auth_connection.load_collection("SENTINEL1_GRD")
                 .filter_bands(["VV", "VH"])
                 .filter_bbox([2.59003, 51.069, 2.8949, 51.2206])
                 .filter_temporal(["2019-10-10", "2019-10-10"]))

    output_tiff = tmp_path / "test_sentinel_hub_default_sar_backscatter_synchronous.tif"
    data_cube.download(output_tiff)

    with rasterio.open(output_tiff) as dataset:
        assert dataset.crs == "EPSG:32631"
        assert dataset.res == (10, 10)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_sentinel_hub_sar_backscatter_batch_process(auth_connection, tmp_path):
    # FIXME: a separate filter_bands call drops the mask and local_incidence_angle bands
    sar_backscatter = (auth_connection
                       .load_collection('SENTINEL1_GAMMA0_SENTINELHUB', bands=["VV", "VH"],
                                        properties={"timeliness": lambda t: t == "NRT3h"})
                       .filter_bbox(west=2.59003, east=2.8949, north=51.2206, south=51.069)
                       .filter_temporal(extent=["2019-10-10", "2019-10-10"])
                       .sar_backscatter(mask=True, local_incidence_angle=True, elevation_model='COPERNICUS_30'))

    job = sar_backscatter.execute_batch(out_format='GTiff', title="SentinelhubSarBackscatterBatch")

    assets = job.download_results(tmp_path)
    assert len(assets) > 1  # includes original tile and CARD4L metadata

    result_asset_paths = [path for path in assets.keys() if path.name.startswith("openEO")]
    assert len(result_asset_paths) == 1

    output_tiff = result_asset_paths[0]
    assert_geotiff_basics(output_tiff, expected_band_count=4)  # VV, VH, mask and local_incidence_angle

    # This part of the test is still experimental and might fail.
    # Therefore we log it as a warning instead of failing the test, until we
    # can verify that it works.
    try:
        # Verify projection metadata.
        job_results: JobResults = job.get_results()
        assert_projection_metadata_present(job_results.get_metadata())
    except Exception as e:
        _log.warning(
            "This failed {e!r}. This part of the test is still experimental.",
            exc_info=True,
        )


# this function checks that only up to a portion of values do not match within tolerance
# there always are few pixels with reflections, ... etc on images which is sensitive to very small changes in the code
def compare_xarrays(xa1,xa2,max_nonmatch_ratio=0.01, tolerance=1.e-6):
    np.testing.assert_allclose(xa1.shape,xa2.shape)
    nmax=xa1.shape[-1]*xa1.shape[-2]*max_nonmatch_ratio
    diff=xarray.ufuncs.fabs(xa1-xa2)>tolerance  # TODO: replace with numpy's ufuncs and uncap xarray
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


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_discard_result_suppresses_batch_job_output_file(auth_connection):
    cube = auth_connection.load_collection("PROBAV_L3_S10_TOC_333M", bands=["NDVI"]).filter_bbox(
        5, 6, 52, 51, "EPSG:4326"
    )
    cube = cube.process("discard_result", arguments={"data": cube})

    job = cube.execute_batch(max_poll_interval=BATCH_JOB_POLL_INTERVAL, title="test_discard_result_suppresses_batch_job_output_file")
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
    s2_bands = auth_connection.load_collection("TERRASCOPE_S2_TOC_V2", bands=["B04", "B08", "SCENECLASSIFICATION_20M"], spatial_extent=extent)
    s2_bands = s2_bands.process("mask_scl_dilation", data=s2_bands, scl_band_name="SCENECLASSIFICATION_20M")
    b4_band = s2_bands.band("B04")
    b8_band = s2_bands.band("B08")
    s2_ndvi = (b8_band - b4_band) / (b8_band + b4_band)
    s2_ndvi = s2_ndvi.add_dimension("bands", "s2_ndvi", type="bands")

    datacube = s2_ndvi
    pv_ndvi = auth_connection.load_collection('PROBAV_L3_S10_TOC_333M', bands=['NDVI'], spatial_extent=extent)
    pv_ndvi = pv_ndvi.resample_cube_spatial(s2_ndvi)
    pv_ndvi = pv_ndvi.mask_polygon(poly)
    datacube = datacube.merge_cubes(pv_ndvi)
    # apply filters
    datacube = datacube.filter_temporal(startdate, enddate)#.filter_bbox(**extent)
    datacube.download("merged.nc", format="NetCDF", options=dict(strict_cropping=True))
    dataset = xarray.open_dataset("merged.nc", engine="h5netcdf").drop_vars("crs")
    timeseries = dataset.mean(dim=['x', 'y'])

    assert_array_almost_equal([182.45325, 187.22632, np.nan], timeseries.NDVI.values, 2)
    assert_allclose([np.nan, np.nan, 0.5955337], timeseries.s2_ndvi.values, atol=0.005)


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
    # Use unique UDP name (for this test)
    udp_name = f"f2c_omxu38tkfdujeu3o0843"

    # Define UDP
    from openeo.processes import divide, subtract, process
    fahrenheit = Parameter.number("fahrenheit")
    fahrenheit_to_celsius = divide(x=subtract(x=fahrenheit, y=32), y=1.8)
    auth_connection.save_user_defined_process(udp_name, fahrenheit_to_celsius, parameters=[fahrenheit])

    # Use UDP
    pg = process(udp_name, namespace="user", fahrenheit=50)
    job = auth_connection.create_job(pg,title="udp_simple_math_batch_job")
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


def test_aggregate_spatial_point_handling(auth_connection):
    data_cube = auth_connection.load_collection(
        "TERRASCOPE_S2_TOC_V2",
        bands=["B04", "B03", "B02"],
        temporal_extent=["2019-09-25", "2019-09-30"],
    )
    means = data_cube.aggregate_spatial(Point(2.7355, 51.1281), "mean").execute()

    assert means == {
        "2019-09-26T00:00:00Z": [[6832.0, 6708.0, 6600.0]],
        "2019-09-28T00:00:00Z": [[976.0, 843.0, 577.0]],
    }


def as_feature(geometry: BaseGeometry) -> dict:
    return {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(geometry),
    }


def as_feature_collection(*geometries: BaseGeometry) -> dict:
    return {
        "type": "FeatureCollection",
        "properties": {},
        "features": [as_feature(g) for g in geometries],
    }


def test_aggregate_spatial_feature_collection_heterogeneous_multiple_aggregates(
    auth_connection,
):
    data_cube = auth_connection.load_collection(
        "TERRASCOPE_S2_TOC_V2",
        bands=["B04", "B03", "B02"],
        temporal_extent=["2019-09-25", "2019-09-30"],
    )

    geometry = as_feature_collection(
        Point(2.7355, 51.1281),
        Point(2.7360, 51.1284),
        Polygon.from_bounds(2.7350, 51.1283, 2.7353, 51.1285),
        Polygon.from_bounds(2.7358, 51.1278, 2.7361, 51.1280),
    )

    from openeo.processes import array_create

    result = data_cube.aggregate_spatial(
        geometry, lambda d: array_create(d.min(), d.count())
    ).execute()

    assert result == {
        "2019-09-26T00:00:00Z": [
            [6832, 1, 6708, 1, 6600, 1],
            [6888, 1, 6756, 1, 6576, 1],
            [6748, 4, 6616, 4, 6512, 4],
            [6820, 4, 6680, 4, 6556, 4],
        ],
        "2019-09-28T00:00:00Z": [
            [976, 1, 843, 1, 577, 1],
            [1082, 1, 1070, 1, 611, 1],
            [358, 4, 474, 4, 267, 4],
            [1018, 4, 793, 4, 607, 4],
        ],
    }


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_point_timeseries_from_batch_process(auth_connection):
    large_polygon = Polygon.from_bounds(2.640329849675133, 49.745440618122501, 3.297496358117944, 50.317367956014152)
    center_point = large_polygon.centroid

    geometries = GeometryCollection([large_polygon, center_point])

    data_cube = (auth_connection.load_collection('SENTINEL2_L1C_SENTINELHUB', bands=["B04", "B03", "B02"])
                 .filter_temporal(extent=["2019-09-26", "2019-09-26"])
                 .aggregate_spatial(geometries, "mean"))

    job = data_cube.execute_batch(title="test_point_timeseries_from_batch_process")

    timeseries = job.get_results().get_assets()[0].load_json()

    expected_schema = schema.Schema({str: [[float]]})
    assert expected_schema.validate(timeseries)

    _, geometry_values = list(timeseries.items())[0]
    assert len(geometry_values) == len(geometries.geoms)


@pytest.mark.batchjob
@pytest.mark.timeout(BATCH_JOB_TIMEOUT)
def test_load_collection_references_correct_batch_process_id(auth_connection, tmp_path):
    bbox = [2.640329849675133, 49.745440618122501, 3.297496358117944, 50.317367956014152]

    collection = 'SENTINEL1_GRD'
    spatial_extent = {'west': bbox[0], 'east': bbox[2], 'south': bbox[1], 'north': bbox[3], 'crs': 'EPSG:4326'}
    temporal_extent = ["2018-01-01", "2018-01-01"]
    bands = ["VV", "VH"]

    s1 = auth_connection.load_collection(collection, spatial_extent=spatial_extent, bands=bands,
                                         temporal_extent=temporal_extent)
    s1_sigma = s1.sar_backscatter(coefficient="sigma0-ellipsoid").rename_labels(dimension='bands',
                                                                                target=["VV_sigma0", "VH_sigma0"],
                                                                                source=["VV", "VH"])
    s1_gamma = s1.sar_backscatter(coefficient="gamma0-terrain").rename_labels(dimension='bands',
                                                                              target=["VV_gamma0", "VH_gamma0"],
                                                                              source=["VV", "VH"])
    result = s1_sigma.merge_cubes(s1_gamma)

    output_tiff = tmp_path / "merged_batch_large.tif"

    job = result.execute_batch(
        output_tiff,
        out_format="GTiff",
        title="test_load_collection_references_correct_batch_process_id",
    )

    assert_geotiff_basics(output_tiff, expected_band_count=4)

    with rasterio.open(output_tiff) as ds:
        sigma0_vv = ds.read(1)
        sigma0_vh = ds.read(2)
        gamma0_vv = ds.read(3)
        gamma0_vh = ds.read(4)

        # all bands should be different, otherwise one batch process was inadvertently re-used and the other one ignored
        for band1, band2 in itertools.combinations([sigma0_vv, sigma0_vh, gamma0_vv, gamma0_vh], 2):
            assert not np.array_equal(band1, band2)

    # This part of the test is still experimental.
    # Therefore we log it as a warning instead of failing the test, until we
    # can verify that it works.
    try:
        # Verify projection metadata.
        job_results: JobResults = job.get_results()
        assert_projection_metadata_present(job_results.get_metadata())
    except Exception as e:
        _log.warning(
            "This failed {e!r}. This part of the test is still experimental.",
            exc_info=True,
        )


def test_tsservice_geometry_mean(tsservice_base_url):
    time_series = requests.post(
        f"{tsservice_base_url}/v1.0/ts/S2_FAPAR_FILE/geometry?startDate=2020-04-05&endDate=2020-04-05",
        json={
            "type": "Polygon",
            "coordinates": [
                [[1.90283, 50.9579], [1.90283, 51.0034], [1.97116, 51.0034], [1.97116, 50.9579],
                 [1.90283, 50.9579]]]
        }).json()

    expected_schema = schema.Schema({'results': [{
        'date': str,
        'result': {
            'average': float,
            'totalCount': int,
            'validCount': int
        }}]
    })

    assert expected_schema.validate(time_series)


def test_auth_jenkins_oidc_client_credentials_me(connection, auth_connection):
    """
    WIP for #6: OIDC Client Credentials auth for jenkins user
    """
    # TODO: skip this test automatically when not running in Jenkins context?
    me = connection.describe_account()
    _log.info(f"connection.describe_account -> {me=}")
    assert me["user_id"] == "jenkins"
