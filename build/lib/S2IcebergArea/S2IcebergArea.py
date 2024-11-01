import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from skimage import measure
from datetime import datetime
from skimage.measure import label
from S2IcebergArea.io.IO import IO
from scipy.ndimage import convolve
from rasterstats import zonal_stats
from rasterio.features import shapes
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from S2IcebergArea.s2_preprocessing.S2Prep import S2Prep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-3s %(message)s")

STATS = "mean std percentile_25 percentile_75 min max"
RANDOM_STATE = 436
FEATURE_NAMES = [
    "b2_mean", 
    "b3_mean",
    "b4_mean",
    "b8_mean",
    "b2_std",
    "b3_std",
    "b4_std",
    "b8_std",
    "b2_p25",
    "b3_p25",
    "b4_p25",
    "b8_p25", 
    "b2_p75",
    "b3_p25", 
    "b4_p75",    
    "b8_p75",
    "b2_max",
    "b3_max",
    "b4_max",
    "b8_max",
    "b2_min",
    "b3_min",
    "b4_min",
    "b8_min"
    ]

FEATURE_NAMES = [
    "b2_mean",
    "b3_mean",
    "b4_mean",
    "b8_mean",
    "b2_std",
    "b3_std",
    "b4_std",
    "b8_std",
    "b2_p25",
    "b2_p75",
    "b3_p25",
    "b3_p75",
    "b4_p25",
    "b4_p75",
    "b8_p25",
    "b8_p75"
]

B8_THRESHOLD = 0.12  # reflectance
CLASSIFICATION_PROBABILITY_THRESHOLD = 0.35
CLASSIFICATION_PROBABILITY_THRESHOLD_UNCERTAIN = 0.5
PERIMETER_INDEX_UNCERTAIN = 0.11  # bottom 1% perimeter index present in reference icebergs
MELANGE_AND_UNCERTAIN_ICE = "MELANGE_AND_UNCERTAIN_ICE"
MINIMUM_SIZE = 4  # pixels
MAXIMUM_SIZE = 90000  # pixels

MEANS_SEA_ICE = np.float32([0.19579765, 0.2676879, 0.29685718, 0.3382447])


class S2IcebergArea:
    def __init__(self) -> None:
        self.io = IO()
        self.meta_s2 = None
        self.file_s2 = None
        self.data_s2 = None

    def preprocess_s2(self, dir_safe, dir_out):
        """
        Read relevant bands, calculate cloud probabilities.
        :param: dir_safe str the location of the unzipped Sentinel-2 level 1C .SAFE folder.
        :param: dir_out str the location where the preprocessed Sentinel-2 data should be written.
        """
        self.file_s2 = S2Prep().preprocess_s2(dir_safe, dir_out, ["B08", "B04", "B03", "B02"])
        return self.file_s2

    def run_model(self, file_s2=None, aoi=None, cloud_probability_threshold=5):
        """
        Delineate ice features, distinguish icebergs and sea ice.
        :param: file_s2 str file path of preprocessed Sentinel-2 data.
        :param: aoi geopandas GeoDataFrame delineating the area to be processed.
        :param: cloud_probability_threshold float specifies the cloud probability value above which pixels are flagged as clouds. Default is 5%.
        """
        t0 = datetime.now()
        self.file_s2 = self.file_s2 if file_s2 is None else file_s2
        logging.info("Reading Sentinel-2 data")
        self.data_s2, self.meta_s2 = self.io.read_s2(file_s2, indexes=[1, 2, 3, 4, 5], aoi=aoi)
        self.data_s2 = self._mask_clouds(self.data_s2, cloud_probability_threshold)
        ice = self.ice(self.data_s2[0])
        try:
            ice_polygons = self._to_polygons(ice, True)
        except KeyError:
            return
        ice_polygons["perimeter_index"] = self._calc_perimeter_index(ice_polygons)
        logging.info("Classifying sea ice and icebergs")
        ice_polygons = self._predict(ice_polygons, self._get_probability_threshold())
        max_prob = np.max(np.vstack([ice_polygons["probability_iceberg"], ice_polygons["probability_sea_ice"]]), axis=0)
        subset_mask = np.max(np.bool8([
            max_prob < self._get_uncertain_probability_threshold(),
            np.bool8(ice_polygons["perimeter_index"] < PERIMETER_INDEX_UNCERTAIN)
        ]), axis=0) == 1
        if sum(subset_mask) > 0:
            logging.info("Classifying uncertain ice features")
            classified = self._classify_melange(ice_polygons, subset_mask, cloud_probability_threshold)
            ice_polygons = ice_polygons if classified is None else classified
        print("Elapsed:", (datetime.now() - t0).total_seconds() / 60, "min")
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 0, "predicted_ice_feature_name"] = "iceberg"
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 1, "predicted_ice_feature_name"] = "sea_ice"
        return ice_polygons
    
    def _classify_melange(self, ice_polygons, subset_mask, cloud_probability_threshold):
        ca = "classification_algorithm"
        ice_polygons_uncertain = ice_polygons[subset_mask]  # uncertain about those, segment them and classify again
        ice_polygons_uncertain.to_file("/media/henrik/DATA/ice_polys_uncertain.gpkg")
        ice_polygons_certain = ice_polygons[~subset_mask]
        logging.info("Reading S2 data")
        self.data_s2, self.meta_s2 = self.io.read_s2(self.file_s2, indexes=[1, 2, 3, 4, 5], aoi=ice_polygons_uncertain)  # B2, cloud probabilities
        logging.info("Masking clouds")
        self.data_s2 = self._mask_clouds(self.data_s2, cloud_probability_threshold)
        self.data_s2 = self.data_s2[:-1]
        logging.info("Calculating reflectance difference")
        difference_from_reference = np.full_like(self.data_s2, fill_value=np.nan)
        for i in range(self.data_s2.shape[0]):
            difference_from_reference[i] = self.data_s2[i] - MEANS_SEA_ICE[i]
        mean_difference_from_reference = np.mean(difference_from_reference, 0)
        difference_from_reference = None
        difference_convoled = convolve(mean_difference_from_reference, np.ones((5, 5)) / 25)
        ice = np.int8(difference_convoled >= 0.18)
        difference_convoled = None
        try:
            ice_polygons_melange = self._to_polygons(ice, False)
        except ValueError:
            return
        logging.info("Classifying sea ice and icebergs in melange")
        ice_polygons_melange = self._predict(ice_polygons_melange, self._get_uncertain_probability_threshold())
        ice_polygons_melange.to_file("/media/henrik/DATA/ipm.gpkg")
        ice_polygons_melange["perimeter_index"] = self._calc_perimeter_index(ice_polygons_melange)
        ice_polygons_melange[ca] = MELANGE_AND_UNCERTAIN_ICE
        if len(ice_polygons_melange) > 0:
            icebergs = ice_polygons_melange[ice_polygons_melange["predicted_ice_feature_int"] == 0]
            sea_ice = ice_polygons_melange[ice_polygons_melange["predicted_ice_feature_int"] == 1]
            try:
                sea_ice = gpd.overlay(sea_ice, icebergs, how="difference")
            except IndexError:
                only_sea_ice = len(sea_ice) > 0 and len(icebergs) == 0
                ice_polygons_melange = sea_ice if only_sea_ice else icebergs
                ice_polygons_melange["predicted_ice_feature_int"] = int(only_sea_ice)
            else:
                #logging.info("Merging touching polygons")
                #icebergs = self._merge_touching_polygons(icebergs)
                #sea_ice = self._merge_touching_polygons(sea_ice)
                icebergs["predicted_ice_feature_int"] = 0
                sea_ice["predicted_ice_feature_int"] = 1
            ice_polygons_melange[ca] = MELANGE_AND_UNCERTAIN_ICE
            ice_polygons_melange = gpd.GeoDataFrame(pd.concat([icebergs, sea_ice]))
            ice_polygons_melange.geometry = ice_polygons_melange["geometry"]
            ice_polygons_melange.crs = icebergs.crs
        else:
            ice_polygons_melange = gpd.GeoDataFrame()
        ice_polygons_certain[ca] = "OPEN_WATER"        
        ice_polygons_merged = gpd.GeoDataFrame(pd.concat([ice_polygons_certain, ice_polygons_melange]))  # merge initial certain ice and ice melange
        ice_polygons_difference = gpd.overlay(ice_polygons_uncertain, ice_polygons_merged, how="difference")
        ice_polygons_difference[ca] = "RESIDUAL_UNCERTAIN_ICE"
        ice_polygons_difference["predicted_ice_feature_int"] = 1  #np.int8(ice_polygons_difference["probability_iceberg"] < self._get_probability_threshold())
        ice_polygons_merged = gpd.GeoDataFrame(pd.concat([ice_polygons_merged, ice_polygons_difference]))
        ice_polygons_merged.geometry = ice_polygons_merged["geometry"]
        ice_polygons_merged.crs = ice_polygons_certain.crs       
        return ice_polygons_merged

    def _predict(self, ice_polygons, classification_probability_threshold):
        logging.info("Extracting statistics")
        ice_polygons = self._extract_statistics(ice_polygons)
        features = self._reshape_features(ice_polygons)
        model = self.io.read_model()
        logging.info("Predicting")
        proba = model.predict_proba(features)  # probabilities
        ice_polygons["predicted_ice_feature_int"] = np.ones(len(ice_polygons)) * np.int8(proba[:, 0] < classification_probability_threshold)  # 0: iceberg, 1: sea ice
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 0, "predicted_ice_feature_name"] = "iceberg"
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 1, "predicted_ice_feature_name"] = "sea_ice"
        ice_polygons["probability_iceberg"] = proba[:, 0]
        ice_polygons["probability_sea_ice"] = proba[:, 1]
        return ice_polygons

    def _get_uncertain_probability_threshold(self):
        return CLASSIFICATION_PROBABILITY_THRESHOLD_UNCERTAIN

    def _to_polygons(self, ice, eliminate_small_features=True):
        """
        :param: outliers np.int8 binary outliers (1=outlier, 0=no outlier).
        """
        logging.info("Polygonizing detected ice")
        polygons = gpd.GeoDataFrame()
        if eliminate_small_features:
            logging.info("Eliminating out-of-size-range ice features")
            ice[convolve(ice, np.ones((3, 3))) < 3] = 0  # eliminate pixels with less than 2 neighbors
            ice[convolve(ice, np.ones((3, 3))) < 3] = 0  # eliminate pixels with less than 2 neighbors
            ice[convolve(ice, np.ones((3, 3))) < 3] = 0  # eliminate pixels with less than 2 neighbors
            ice[convolve(ice, np.ones((3, 3))) < 3] = 0  # eliminate pixels with less than 2 neighbors
        t0 = datetime.now()
        polygons = []
        for contour in measure.find_contours(ice, level=0.5):
            transformed_coords = [self.meta_s2["transform"] * (x, y) for y, x in contour]  # (y, x) from skimage, so reverse to (x, y)            
            poly = Polygon(transformed_coords)
            if poly.is_valid:
                polygons.append(poly)
        polygons = gpd.GeoDataFrame(geometry=polygons, crs=self.meta_s2["crs"]).dissolve().explode(index_parts=True)
        logging.info("measure.find_contours - Polygonizing took {} minutes".format((datetime.now() - t0).total_seconds() / 60))
        #polygons = gpd.GeoDataFrame(pd.concat(results))
        polygons.geometry = polygons["geometry"]
        polygons = polygons[np.bool8(polygons.area >= (MINIMUM_SIZE * 10 ** 2))]
        polygons.crs = self.meta_s2["crs"]
        polygons.index = list(range(len(polygons)))
        polygons["area"] = polygons.area
        return polygons

    def _do_polygonize(self, labels, value):
        polygons = gpd.GeoDataFrame()
        shapes_results = shapes(labels, mask=labels == value, connectivity=4, transform=self.meta_s2["transform"])
        for i, (s, v) in enumerate(shapes_results):
            if v != 0:
                polygons.loc[i, "geometry"] = Polygon(s["coordinates"][0])
                polygons.loc[i, "raster_val"] = v
        return polygons

    def _extract_statistics(self, ice_polygons):
        band_names = ["b8", "b4", "b3", "b2"]
        t0 = datetime.now()
        stats_all_bands = Parallel(n_jobs=7)(delayed(self._do_extract_statistics)(ice_polygons, self.data_s2[band_idx]) for band_idx, band in zip([0, 1, 2, 3], band_names))
        print("Elapsed extract statistics:", (datetime.now() - t0).total_seconds() / 60)
        self.data_s2 = None
        #t0 = datetime.now()
        #for band_idx, band in zip([0, 1, 2, 3], ["b8", "b4", "b3", "b2"]):
            #stats = zonal_stats(ice_polygons, self.data_s2[band_idx], affine=self.meta_s2["transform"], stats=STATS, nodata=np.nan)
        #print("Normal - Elapsed extract statistics:", (datetime.now() - t0).total_seconds() / 60)
        for band_idx, stats in enumerate(stats_all_bands):
            for i, stat in enumerate(stats):
                for key, value in stat.items():
                    key = key.replace("percentile_", "p")
                    ice_polygons.loc[i, "_".join([band_names[band_idx], key])] = value
        return ice_polygons
    
    def _do_extract_statistics(self, ice_polygons, band_data):
        return zonal_stats(ice_polygons, band_data, affine=self.meta_s2["transform"], stats=STATS, nodata=np.nan)

    @staticmethod
    def _get_probability_threshold():
        return CLASSIFICATION_PROBABILITY_THRESHOLD

    @staticmethod
    def _calc_perimeter_index(polygons):
        return np.sqrt(polygons.area) / polygons.geometry.length

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
