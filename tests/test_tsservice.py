import json
import os

import pytest
import requests
import schema


def get_tsservice_base_url():
    tsservice_endpoint = os.environ.get("TSSERVICE_ENDPOINT")

    if not tsservice_endpoint:
        raise RuntimeError(
            "Environment variable 'TSSERVICE_ENDPOINT' should be set"
            " with URL pointing to OpenEO/tsservice backend to test against"
            " (e.g. 'http://localhost:8155/')"
        )

    return tsservice_endpoint


@pytest.fixture
def tsservice_base_url():
    return get_tsservice_base_url()


def test_tsservice_geometry_mean(tsservice_base_url):
    request = requests.Request(
        "POST",
        f"{tsservice_base_url}/v1.0/ts/S2_FAPAR_FILE/geometry?startDate=2020-04-05&endDate=2020-04-05",
        json={
            "type": "Polygon",
            "coordinates": [
                [
                    [1.90283, 50.9579],
                    [1.90283, 51.0034],
                    [1.97116, 51.0034],
                    [1.97116, 50.9579],
                    [1.90283, 50.9579],
                ]
            ],
        },
        headers={"referer": "https://viewer.terrascope.be"},
    ).prepare()

    _test_tsservice_geometry_mean(
        request,
        expected_response={
            "results": [
                {
                    "date": "2020-04-05",
                    "result": {
                        "totalCount": 670232,
                        "validCount": 669368,
                        "average": pytest.approx(0.24494559046742598, rel=0.01),
                    },
                }
            ],
        },
    )


def test_tsservice_coherence(tsservice_base_url):
    request = requests.Request(
        "POST",
        f"{tsservice_base_url}/v1.0/ts/TERRASCOPE_S1_SLC_COHERENCE_V1_VV/geometry?startDate=2025-04-25&endDate=2025-04-25&zoom=13",
        json={
            "type": "Polygon",
            "coordinates": [
                [
                    [4.609179178723652, 50.274169923319334],
                    [4.609182083202798, 50.27348832084863],
                    [4.609183826988623, 50.27292430267184],
                    [4.609144423048233, 50.27193865421333],
                    [4.608711743051483, 50.27189418017679],
                    [4.608110258677359, 50.27184317120293],
                    [4.607014348283774, 50.27180029131881],
                    [4.606271561022157, 50.271765418046556],
                    [4.60631931130743, 50.27280997962623],
                    [4.606352788488081, 50.27411178263253],
                    [4.60704612129811, 50.274114569520464],
                    [4.609179178723652, 50.274169923319334],
                ]
            ],
        },
        headers={"referer": "https://viewer.terrascope.be"},
    ).prepare()

    _test_tsservice_geometry_mean(
        request,
        expected_response={
            "results": [
                {
                    "date": "2025-04-25",
                    "result": {
                        "totalCount": 336,
                        "validCount": 336,
                        "average": pytest.approx(0.2991666666666667, rel=0.01),
                    },
                }
            ],
        },
    )


def _test_tsservice_geometry_mean(request: requests.PreparedRequest, expected_response: dict):
    response_text = requests.Session().send(request).text

    try:
        time_series = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON. Error: {e} \nFull response: {response_text}") from e

    expected_schema = schema.Schema(
        {
            "results": [
                {
                    "date": str,
                    "result": {"average": float, "totalCount": int, "validCount": int},
                }
            ]
        }
    )

    assert expected_schema.validate(time_series)
    assert time_series == expected_response
