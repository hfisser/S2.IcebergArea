import logging
import diptest
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
from skimage import measure
from datetime import datetime
from rasterio.mask import mask
from S2IcebergArea.io.IO import IO
from scipy.ndimage import convolve
from rasterstats import zonal_stats
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
    "b2_p75",
    "b3_p25",
    "b3_p75",
    "b4_p25",
    "b4_p75",
    "b8_p25",
    "b8_p75"
]
B8_THRESHOLD = 0.12  # reflectance
CLASSIFICATION_PROBABILITY_THRESHOLD = 0.4
CLASSIFICATION_PROBABILITY_THRESHOLD_UNCERTAIN = 0.5
PERIMETER_INDEX_UNCERTAIN = 0.17  # bottom 10% perimeter index present in reference icebergs
MELANGE_AND_UNCERTAIN_ICE = "MELANGE_AND_UNCERTAIN_ICE"
MINIMUM_SIZE = 4  # pixels
MAXIMUM_SIZE = 4000000

MEANS_SEA_ICE = np.float32([0.19579765, 0.2676879, 0.29685718, 0.3382447])  # B8, B4, B3, B2
MEANS_ICEBERGS = np.float32([0.3059331543269159, 0.42718960445418225, 0.45473081093161494, 0.5013769382575902])  # B8, B4, B3, B2
STDS_SEA_ICE = np.float32([0.07266498358413727, 0.090049245607463, 0.08898753339505419, 0.08966630508171987])  # B8, B4, B3, B2
STDS_ICEBERGS = np.float32([0.09416581207557331, 0.10649453614239697, 0.09122658032360802, 0.08765274171544463])

MEAN_PERIMETER_INDEX_ICEBERGS = 0.18872174773715583
STD_PERIMETER_INDEX_ICEBERGS = 0.03314749059550769

THRESHOLD_PERIMETER_INDEX_HIGH = -1
THRESHOLD_PERIMETER_INDEX_LOW = -3


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
        self.data_s2, self.meta_s2 = self.io.read_s2(file_s2, indexes=[1], aoi=aoi)
        self.data_s2 = self._mask_clouds(self.file_s2, aoi, self.data_s2, cloud_probability_threshold)
        ice = self.ice(self.data_s2[0])
        self.data_s2 = None
        if np.max(ice) == 0:
            return
        try:
            ice_polygons = self._to_polygons(ice, True)
        except KeyError:
            return
        ice_polygons["perimeter_index"] = self._calc_perimeter_index(ice_polygons)
        logging.info("Classifying sea ice and icebergs")
        subset_mask = np.bool8(ice_polygons.area < MAXIMUM_SIZE)
        subset = ice_polygons[subset_mask]
        subset.index = list(range(len(subset)))
        large = ice_polygons[~subset_mask]
        ice_polygons = self._predict(subset, CLASSIFICATION_PROBABILITY_THRESHOLD)
        for col in ice_polygons:
            if any([s in col for s in STATS.split(" ")]):
                large[col] = np.nan
        large["predicted_ice_feature_name"] = "sea_ice"
        large["predicted_ice_feature_int"] = 1
        large["probability_iceberg"] = 0
        large["probability_sea_ice"] = 1
        ice_polygons = gpd.GeoDataFrame(pd.concat([ice_polygons, large]))
        ice_polygons.geometry = ice_polygons["geometry"]
        ice_polygons.crs = subset.crs
        ice_polygons.index = list(range(len(ice_polygons)))
        md_perimeter_index = (ice_polygons["perimeter_index"] - MEAN_PERIMETER_INDEX_ICEBERGS) / STD_PERIMETER_INDEX_ICEBERGS
        subset_mask = np.bool8(md_perimeter_index < THRESHOLD_PERIMETER_INDEX_HIGH)
        if sum(subset_mask) > 0:
            logging.info("Classifying uncertain ice features")
            classified = self._classify_melange(ice_polygons, subset_mask, cloud_probability_threshold)
            ice_polygons = ice_polygons if classified is None else classified
        logging.info("Reclassifying")
        ice_polygons = self._reclassify(ice_polygons)
        print("Elapsed:", (datetime.now() - t0).total_seconds() / 60, "min")
        ice_polygons["predicted_ice_feature_int"] = np.int8(ice_polygons["predicted_ice_feature_int"])
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 0, "predicted_ice_feature_name"] = "iceberg"
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 1, "predicted_ice_feature_name"] = "sea_ice"
        return ice_polygons

    def _classify_melange(self, ice_polygons, subset_mask, cloud_probability_threshold):
        ca = "classification_algorithm"
        ice_polygons_uncertain = ice_polygons[subset_mask]  # uncertain about those, segment them and classify again
        ice_polygons_certain = ice_polygons[~subset_mask]
        logging.info("Reading S2 data")
        self.data_s2, self.meta_s2 = self.io.read_s2(self.file_s2, indexes=[4], aoi=ice_polygons_uncertain)  # B2
        logging.info("Masking clouds")
        self.data_s2 = self._mask_clouds(self.file_s2, ice_polygons_uncertain, self.data_s2, cloud_probability_threshold)
        logging.info("Calculating reflectance difference")
        md_icebergs = (convolve(self.data_s2[0], np.ones((7, 7)) / 7 ** 2) - MEANS_ICEBERGS[3]) / STDS_ICEBERGS[3]
        self.data_s2 = None
        ice = np.int8(md_icebergs >= 0)
        md_icebergs = None
        try:
            ice_polygons_melange = self._to_polygons(ice, False)
        except ValueError:
            return
        ice = None
        ice_polygons_melange["perimeter_index"] = self._calc_perimeter_index(ice_polygons_melange)
        md_perimeter_index = (ice_polygons_melange["perimeter_index"] - MEAN_PERIMETER_INDEX_ICEBERGS) / STD_PERIMETER_INDEX_ICEBERGS
        subset_mask = np.max(
            [ice_polygons_melange.area >= MAXIMUM_SIZE, 
            md_perimeter_index < THRESHOLD_PERIMETER_INDEX_LOW], 0
        ) == 1
        ice_polygons_melange_uncertain = ice_polygons_melange[subset_mask]
        ice_polygons_melange = ice_polygons_melange[~subset_mask]
        ice_polygons_melange.index = list(range(len(ice_polygons_melange)))
        logging.info("Classifying sea ice and icebergs in melange")
        ice_polygons_melange = self._predict(ice_polygons_melange, CLASSIFICATION_PROBABILITY_THRESHOLD)
        ice_polygons_melange["perimeter_index"] = self._calc_perimeter_index(ice_polygons_melange)
        md_perimeter_index = (ice_polygons_melange["perimeter_index"] - MEAN_PERIMETER_INDEX_ICEBERGS) / STD_PERIMETER_INDEX_ICEBERGS  # perimeter index
        ice_polygons_melange.loc[md_perimeter_index < THRESHOLD_PERIMETER_INDEX_LOW, "predicted_ice_feature_int"] = 1
        ice_polygons_melange[ca] = MELANGE_AND_UNCERTAIN_ICE
        for col in ice_polygons_melange:
            if any([s in col for s in STATS.split(" ")]):
                ice_polygons_melange_uncertain[col] = np.nan
        ice_polygons_melange = gpd.GeoDataFrame(pd.concat([ice_polygons_melange, ice_polygons_melange_uncertain]))
        ice_polygons_melange.geometry = ice_polygons_melange["geometry"]
        ice_polygons_melange.crs = ice_polygons_melange.crs
        ice_polygons_melange.index = list(range(len(ice_polygons_melange)))
        if len(ice_polygons_melange) > 0:
            icebergs = ice_polygons_melange[ice_polygons_melange["predicted_ice_feature_int"] == 0]
            sea_ice = ice_polygons_melange[ice_polygons_melange["predicted_ice_feature_int"] == 1]
            try:
                sea_ice = gpd.overlay(sea_ice, icebergs, how="difference")
            except IndexError:
                only_sea_ice = len(sea_ice) > 0 and len(icebergs) == 0
                ice_polygons_melange = sea_ice if only_sea_ice else icebergs
                ice_polygons_melange["predicted_ice_feature_int"] = int(only_sea_ice)
                #logging.info("Merging touching polygons")
                #try:
                    #icebergs = self._merge_touching_polygons(icebergs)
                    #icebergs["predicted_ice_feature_int"] = 0
                #except AttributeError:
                    #pass
                #try:
                    #sea_ice = self._merge_touching_polygons(sea_ice)                    
                    #sea_ice["predicted_ice_feature_int"] = 1
                #except AttributeError:
                    #pass
            ice_polygons_melange[ca] = MELANGE_AND_UNCERTAIN_ICE
            ice_polygons_melange = gpd.GeoDataFrame(pd.concat([icebergs, sea_ice]))
            ice_polygons_melange.geometry = ice_polygons_melange["geometry"]
            ice_polygons_melange.crs = icebergs.crs
        else:
            ice_polygons_melange = gpd.GeoDataFrame()
        ice_polygons_certain[ca] = "OPEN_WATER"        
        ice_polygons_merged = gpd.GeoDataFrame(pd.concat([ice_polygons_certain, ice_polygons_melange]))  # merge initial certain ice and ice melange
        #ice_polygons_difference = gpd.overlay(ice_polygons_uncertain, ice_polygons_merged, how="difference")
        ice_polygons_uncertain[ca] = "RESIDUAL_UNCERTAIN_ICE"
        ice_polygons_uncertain["predicted_ice_feature_int"] = 1  #np.int8(ice_polygons_difference["probability_iceberg"] < self._get_probability_threshold())
        ice_polygons_merged = gpd.GeoDataFrame(pd.concat([ice_polygons_merged, ice_polygons_uncertain]))
        ice_polygons_merged.geometry = ice_polygons_merged["geometry"]
        ice_polygons_merged.crs = ice_polygons_certain.crs
        logging.info("Melange completed")
        return ice_polygons_merged

    def _reclassify(self, ice_polygons):
        sea_ice = ice_polygons[ice_polygons["predicted_ice_feature_int"] == 1]        
        try:
            b2 = sea_ice["b2_mean"]
            if len(b2) >= 100:
                dip_statistic, p_value = diptest.diptest(b2[~np.isnan(b2)])
            else:
                raise KeyError
        except KeyError:
            multimodal_sea_ice = False
        else:
            multimodal_sea_ice = p_value < 0.05
        if multimodal_sea_ice:
            ice_polygons["multimodal_sea_ice"] = 1
        else:
            pass
        return ice_polygons
    
    def _predict(self, ice_polygons, classification_probability_threshold):
        logging.info("Extracting statistics")
        ice_polygons = self._extract_statistics(ice_polygons)
        ice_polygons.to_file("/media/henrik/DATA/ice_polygons_melange.gpkg")
        features = self._reshape_features(ice_polygons)
        model = self.io.read_model()
        logging.info("Predicting")
        t0 = datetime.now()
        proba = model.predict_proba(features)  # probabilities
        print("Predicting normal:", (datetime.now() - t0).total_seconds())
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
        t0 = datetime.now()
        polygons = Parallel(n_jobs=6)(delayed(self._do_polygonize)(contour) for contour in measure.find_contours(ice, level=0.5))
        polygons = gpd.GeoDataFrame(geometry=polygons, crs=self.meta_s2["crs"]).dissolve().explode(index_parts=True)
        logging.info("measure.find_contours - Polygonizing took {} minutes".format((datetime.now() - t0).total_seconds() / 60))
        #polygons = gpd.GeoDataFrame(pd.concat(results))
        polygons.geometry = polygons["geometry"]
        polygons = polygons[np.bool8(polygons.area >= (MINIMUM_SIZE * 10 ** 2))]
        polygons.crs = self.meta_s2["crs"]
        polygons.index = list(range(len(polygons)))
        polygons["area"] = polygons.area
        return polygons
    
    def _do_polygonize(self, contour):
        transformed_coords = [self.meta_s2["transform"] * (x, y) for y, x in contour]  # (y, x) from skimage, so reverse to (x, y)            
        poly = Polygon(transformed_coords).buffer(0.01)
        return poly

    def _extract_statistics(self, ice_polygons):
        band_names = ["b8", "b4", "b3", "b2"]
        t0 = datetime.now()
        stats_all_bands = Parallel(n_jobs=6)(delayed(self._do_extract_statistics)(ice_polygons, band_idx + 1) for band_idx in [0, 1, 2, 3])
        print("Elapsed extract statistics:", (datetime.now() - t0).total_seconds() / 60)
        self.data_s2 = None
        for band_idx, stats in enumerate(stats_all_bands):
            for i, stat in enumerate(stats):
                for key, value in stat.items():
                    key = key.replace("percentile_", "p")
                    ice_polygons.loc[i, "_".join([band_names[band_idx], key])] = value
        return ice_polygons
    
    def _do_extract_statistics(self, ice_polygons, band_index):
        band_data, meta = self.io.read_s2(self.file_s2, [band_index], aoi=ice_polygons)
        stats = zonal_stats(ice_polygons, band_data.squeeze(), affine=meta["transform"], stats=STATS, nodata=np.nan)
        #stats = zonal_stats(ice_polygons, self.file_s2, affine=meta["transform"], stats=STATS, nodata=np.nan, band=band_index)
        band_data = None
        return stats

    @staticmethod
    def _mahalanobis_distance(points, mean, std):
        return np.abs(points - mean) / std

    @staticmethod
    def _get_probability_threshold():
        return CLASSIFICATION_PROBABILITY_THRESHOLD

    @staticmethod
    def _calc_perimeter_index(polygons):
        return (2 * np.sqrt(np.pi * polygons.area)) / polygons.exterior.length

    @staticmethod
    def _reshape_features(ice_polygons):
        return np.float32([ice_polygons[feature_name] for feature_name in FEATURE_NAMES]).swapaxes(0, 1)

    @staticmethod
    def _mask_clouds(file_s2, aoi, data_s2, cloud_probability_threshold):
        with rio.open(file_s2) as src:
            cloud_mask, _ = mask(src, list(aoi.to_crs(src.crs).geometry), crop=True, indexes=5)
        data_s2[:, cloud_mask >= cloud_probability_threshold] = np.nan
        data_s2[:, np.sum(~np.isfinite(data_s2), 0) > 0] = np.nan
        cloud_mask = None
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
