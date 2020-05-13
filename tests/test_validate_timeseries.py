from unittest import TestCase

import numpy as np
import rasterio
from numpy.testing import assert_allclose
from shapely.geometry import Polygon

import openeo
from .conftest import get_openeo_base_url


class Test(TestCase):

    _rest_base = get_openeo_base_url()

    configs = [
        {
            'layer': 'S2_FAPAR_V102_WEBMERCATOR2',
            'expected_dates': ['2017-11-01T00:00:00',
                                  '2017-11-02T00:00:00',
                                  '2017-11-04T00:00:00',
                                  '2017-11-06T00:00:00',
                                  '2017-11-07T00:00:00',
                                  '2017-11-09T00:00:00',
                                  '2017-11-11T00:00:00',
                                  '2017-11-12T00:00:00',
                                  '2017-11-14T00:00:00',
                                  '2017-11-16T00:00:00',
                                  '2017-11-17T00:00:00',
                                  '2017-11-19T00:00:00',
                                  '2017-11-21T00:00:00']
        },
        {
            'layer':'PROBAV_L3_S10_TOC_NDVI_333M',
            'expected_dates': ["2017-11-01T00:00:00", "2017-11-11T00:00:00", "2017-11-21T00:00:00"]
        }
    ]

    def test_zonal_statistics(self):

        """
        This unit test in openeo-geotrellis-extensions tests a similar case:
        org.openeo.geotrellis.ComputeStatsGeotrellisAdapterTest#compute_median_timeseries_on_accumulo_datacube

        :return:
        """

        for config in Test.configs:
            layer = config['layer']
            with self.subTest(layer,layer=layer):
                self._validate_stats(layer,config['expected_dates'])

    def _validate_stats(self, layer,expected_dates):
        session = openeo.connect(self._rest_base)
        image_collection = session \
            .imagecollection(layer) \
            .filter_temporal("2017-11-01", "2017-11-21")
        polygon = Polygon(shell=[
            (7.022705078125007, 51.75432477678571),
            (7.659912109375007, 51.74333844866071),
            (7.659912109375007, 51.29289899553571),
            (7.044677734375007, 51.31487165178571),
            (7.022705078125007, 51.75432477678571)
        ])
        minx, miny, maxx, maxy = polygon.bounds
        bboxed = image_collection.filter_bbox(west=minx, east=maxx, north=maxy, south=miny, crs="EPSG:4326")
        timeseries = bboxed.polygonal_mean_timeseries(polygon).execute()
        median_timeseries = bboxed.polygonal_median_timeseries(polygon).execute()
        sd_timeseries = bboxed.polygonal_standarddeviation_timeseries(polygon).execute()

        actual_dates = timeseries.keys()
        self.assertEqual(sorted(expected_dates), sorted(actual_dates))

        for date in actual_dates:
            print("Evaluating date: " + date)
            filename = "ts_%s.tif" % date
            session.imagecollection(layer) \
                .filter_temporal(date, date) \
                .filter_bbox(west=minx, east=maxx, north=maxy, south=miny, crs="EPSG:4326") \
                .mask(polygon).download(filename, format="GTIFF")

            with rasterio.open(filename) as f:
                data = f.read(masked=True)
                values_per_band = data[None, ~data.mask]
                if values_per_band.count() == 0:
                    assert [v for v in timeseries[date][0] if v is not None] == []
                    continue
                mean = values_per_band.mean(axis=1)
                print(mean)
                print(timeseries[date])
                assert_allclose(mean, timeseries[date][0], atol=0.6)
                median = np.median(values_per_band, axis=1)

                print(median)
                print(median_timeseries[date + 'Z'][0])
                assert_allclose(median, median_timeseries[date + 'Z'][0], atol=1.0)
                std = np.std(values_per_band, axis=1)
                assert_allclose(std, sd_timeseries[date + 'Z'][0], atol=1.0)

