import os
import pandas as pd
import geopandas as gpd
from datetime import datetime
from S2IcebergArea.S2IcebergArea import S2IcebergArea

dir_safe = "/media/henrik/DATA/S2B_MSIL1C_20240918T140039_N0511_R010_T25WER_20240918T174153.SAFE"

TEST_PREP = False
TEST_PROC = True

file_s2 = "/media/henrik/DATA/raster/s2/25WER/S2A_MSIL1C_20170804T141011_N0500_R053_T25WER_20230829T152854_pB5.0.jp2"

aoi = gpd.read_file("/home/henrik/Output/icebergs/validation/{tile_id}/ocean_buffered_300.0_{tile_id}.gpkg".format(tile_id=os.path.basename(os.path.dirname(file_s2))))
aoi = gpd.read_file("/media/henrik/DATA/aoi_tmp_s2.gpkg")

dir_out = "/home/henrik/Output/icebergs/validation/s2_iceberg_detection"


def test(file_s2, aoi):
    s2_iceberg_area = S2IcebergArea()
    if TEST_PREP:
        file_s2 = s2_iceberg_area.preprocess_s2(dir_safe, os.path.dirname(dir_safe))
    if TEST_PROC:
        ice_polygons = s2_iceberg_area.run_model(file_s2=file_s2, aoi=aoi)
    return ice_polygons


if __name__ == "__main__":
    t0 = datetime.now()
    ice_polygons = test(file_s2, aoi)
    ice_polygons.to_file("/media/henrik/DATA/ice_s2_{}.gpkg".format(os.path.basename(file_s2).split(".")[0]))
