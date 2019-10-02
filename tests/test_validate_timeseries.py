from pathlib import Path
from unittest import TestCase,skip
from openeo.rest import rest_connection as rest_session
from shapely.geometry import shape, Polygon
import os

import rasterio
import numpy as np
from numpy.testing import assert_allclose


class Test(TestCase):
    _rest_base = "%s/openeo/0.4.0" % os.environ['ENDPOINT']
    #_rest_base =  "http://openeo.vgt.vito.be/openeo/0.4.0"


    def test_zonal_statistics(self):
        session = rest_session.session(userid=None, endpoint=self._rest_base)

        image_collection = session \
            .imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
            .filter_temporal("2017-11-01","2017-11-21")

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

        expected_dates = ["2017-11-01T00:00:00", "2017-11-11T00:00:00", "2017-11-21T00:00:00"]
        actual_dates = timeseries.keys()

        self.assertEqual(sorted(expected_dates), sorted(actual_dates))
        self.assertTrue(type(timeseries["2017-11-01T00:00:00"][0]) == float)


        for date in actual_dates:
            print("Evaluating date: " + date)
            filename = "ts_%s.tif" % date
            session.imagecollection('PROBAV_L3_S10_TOC_NDVI_333M') \
                .filter_temporal(date,date) \
                .filter_bbox(west=minx, east=maxx, north=maxy, south=miny, crs="EPSG:4326") \
                .mask(polygon).download(filename, format="GTIFF")

            with rasterio.open(filename) as f:
                data = f.read(masked=True)
                values_per_band = data[None, ~data.mask]
                mean = values_per_band.mean(axis=1)
                assert_allclose(mean,timeseries[date],atol=0.6)
                median = np.median(values_per_band, axis=1)
                assert_allclose(median, median_timeseries[date + 'Z'][0], atol=1.0)
                print(median)
                print(median_timeseries[date + 'Z'][0])
                std = np.std(values_per_band,axis=1)
                assert_allclose(std, sd_timeseries[date + 'Z'][0], atol=0.3)

