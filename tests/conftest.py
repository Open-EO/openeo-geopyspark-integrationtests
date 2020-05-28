import os

import pytest

import openeo
from openeo.capabilities import ComparableVersion


def get_openeo_base_url(version: str = "1.0.0"):
    try:
        endpoint = os.environ["ENDPOINT"].rstrip("/")
    except Exception:
        raise RuntimeError("Environment variable 'ENDPOINT' should be set"
                           " with URL pointing to OpenEO backend to test against"
                           " (e.g. 'http://localhost:8080/' or 'http://openeo-dev.vgt.vito.be/')")
    return "{e}/openeo/{v}".format(e=endpoint.rstrip("/"), v=version)


@pytest.fixture(params=[
    "0.4.2",
    "1.0.0",
])
def api_version(request) -> ComparableVersion:
    return ComparableVersion(request.param)


@pytest.fixture
def api_base_url(api_version):
    return get_openeo_base_url(str(api_version))


@pytest.fixture
def connection(api_base_url) -> openeo.Connection:
    return openeo.connect(api_base_url)
