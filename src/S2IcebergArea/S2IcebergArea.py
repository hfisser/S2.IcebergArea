import os
import numpy as np
import rasterio as rio
import geopandas as gpd
from skimage.measure import label
from S2IcebergArea.io.IO import IO
from rasterstats import zonal_stats
from rasterio.features import shapes
from shapely.geometry import Polygon
from S2IcebergArea.s2_preprocessing.S2Prep import S2Prep

STATS = "mean std percentile_25 percentile_75 min max"
RANDOM_STATE = 436
FEATURE_NAMES = [
    "b2_mean", 
    "b8_mean",
    "b2_std",
    "b8_std", 
    "b2_p25",
    "b8_p25", 
    "b2_p75",
    "b8_p75", 
    "b2_min",   
    "b8_min",
    "b2_max",
    "b8_max"
    ]

B8_THRESHOLD = 0.12  # reflectance
CLASSIFICATION_PROBABILITY_THRESHOLD = 0.35  # iceberg >= 0.35 > sea ice


class S2IcebergArea:
    def __init__(self) -> None:
        self.io = IO()
        self.meta_s2 = None
        self.file_s2 = None

    def preprocess_s2(self, dir_safe):
        """
        Read relevant bands, calculate cloud probabilities.
        :param: dir_safe str the location of the unzipped Sentinel-2 level 1C .SAFE folder.
        """
        prep = S2Prep()
        self.file_s2 = prep.preprocess_s2(dir_safe)
        

    def run_model(self, file_s2=None, aoi=None, cloud_probability_threshold=5):
        """
        Delineate ice features, distinguish icebergs and sea ice.
        :param: file_s2 str file path of preprocessed Sentinel-2 data.
        :param: aoi geopandas GeoDataFrame delineating the area to be processed.
        :param: cloud_probability_threshold float specifies the cloud probability value above which pixels are flagged as clouds. Default is 5%.
        """
        self.file_s2 = self.file_s2 if file_s2 is None else file_s2
        data_s2, self.meta_s2 = self.io.read_s2(file_s2, indexes=[0, 1, 4], aoi=aoi)  # b8, b2, cloud probabilities
        data_s2 = self._mask_clouds(data_s2, cloud_probability_threshold)
        ice = self.ice(data_s2[0])
        ice_polygons = self.to_polygons(ice)
        for band_idx, band in zip([0, 1], ["b8", "b2"]):
            stats = zonal_stats(ice_polygons, data_s2[band_idx])
            for i, stat in enumerate(stats):
                for key, value in stat.items():
                    ice_polygons.loc[i, f"{band}_{key}"] = value
        features = self._reshape_features(ice_polygons)
        model = self.io.read_model()
        proba = model.predict_proba(features)  # probabilities
        ice_polygons["predicted_ice_feature_int"] = np.ones(len(ice_polygons)) * np.int8(proba[:, 0] < CLASSIFICATION_PROBABILITY_THRESHOLD)
        ice_polygons["predicted_ice_feature_name"] = ["sea_ice" if class_int else "iceberg" for class_int in ice_polygons["predicted_ice_feature_int"]]
        ice_polygons["probability_iceberg"] = proba[:, 0]
        ice_polygons["probability_sea_ice"] = proba[:, 1]
        return ice_polygons

    def to_polygons(self, outliers):
        """
        :param: outliers np.int8 binary outliers (1=outlier, 0=no outlier).
        """
        polygons = gpd.GeoDataFrame()
        label_objects = label(outliers).astype(np.float32)  # float32 for shapes() method
        results = ({"properties": {"raster_val": v}, "geometry": s}
        for _, (s, v) in enumerate(shapes(label_objects, transform=self.meta_s2["transform"])))
        for i, polygon in enumerate(results):
            polygons.loc[i, "geometry"] = Polygon(polygon["geometry"]["coordinates"][0])
            polygons.loc[i, "raster_val"] = polygon["properties"]["raster_val"]
        polygons.geometry = polygons["geometry"]
        polygons = polygons[polygons.area < np.nanmax(polygons.area)]  # eliminate bounding box (always created)
        polygons.crs = self.meta_s2["transform"]
        polygons.index = list(range(len(polygons)))
        polygons = polygons[polygons.type == "Polygon"]  # no MultiPolygons
        polygons = polygons[np.add(np.int8(~polygons.is_empty), np.int8(polygons.is_valid)) == 2]
        polygons.index = list(range(len(polygons)))
        if len(polygons) > 0:
            polygons = self._merge_touching_polygons(polygons)
        polygons["area"] = polygons.area
        return polygons

    @staticmethod
    def _reshape_features(ice_polygons):
        return np.float32([ice_polygons[feature_name] for feature_name in FEATURE_NAMES]).swapaxes(0, 1)

    @staticmethod
    def _mask_clouds(data_s2, cloud_probability_threshold):
        data_s2[:, data_s2[-1] >= cloud_probability_threshold] = np.nan
        return data_s2

    @staticmethod
    def _merge_touching_polygons(gdf):
        geoms = gpd.GeoSeries(gdf.geometry.buffer(0.1).unary_union.buffer(-0.1)).explode(index_parts=False)
        gdf_merged = gpd.GeoDataFrame({"geometry": list(geoms.geometry)})
        gdf_merged.geometry = gdf_merged.geometry
        gdf_merged.crs = gdf.crs
        gdf_merged.index = list(range(len(gdf_merged)))
        return gdf_merged

    @staticmethod
    def ice(band8):
        return np.int8(band8 >= B8_THRESHOLD)
