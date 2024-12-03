import logging
import diptest
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
from skimage import measure
from scipy.stats import chi2
from datetime import datetime
from rasterio.mask import mask
from S2IcebergArea.io.IO import IO
from scipy.ndimage import convolve
from sklearn.cluster import KMeans
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from joblib import Parallel, delayed
from shapely.geometry import LineString
from S2IcebergArea.s2_preprocessing.S2Prep import S2Prep

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-3s %(message)s")

N_JOBS = 4
STATS = "mean std percentile_25 percentile_75 min max"
RANDOM_STATE = 436
FEATURE_NAMES = [
    "b2_mean",
    "b3_mean",
    "b4_mean",
    "b8_mean",
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
GEOMETRY_PROBABILITY_THRESHOLD = 0.1
PERIMETER_INDEX_UNCERTAIN = 0.17  # bottom 10% perimeter index present in reference icebergs
MELANGE_AND_UNCERTAIN_ICE = "MELANGE_AND_UNCERTAIN_ICE"
MINIMUM_SIZE = 4  # pixels
MAXIMUM_SIZE = 1000 ** 2

MEANS_SEA_ICE = np.float32([0.19579765, 0.2676879, 0.29685718, 0.3382447])  # B8, B4, B3, B2
MEANS_ICEBERGS = np.float32([0.3059331543269159, 0.42718960445418225, 0.45473081093161494, 0.5013769382575902])  # B8, B4, B3, B2
STDS_SEA_ICE = np.float32([0.07266498358413727, 0.090049245607463, 0.08898753339505419, 0.08966630508171987])  # B8, B4, B3, B2
STDS_ICEBERGS = np.float32([0.09416581207557331, 0.10649453614239697, 0.09122658032360802, 0.08765274171544463])

MEAN_PERIMETER_INDEX_ICEBERGS = 0.6689
STD_PERIMETER_INDEX_ICEBERGS = 0.1176
MEAN_LENGTH_ROOT_LENGTH_RATIO_ICEBERGS = 1.5675
STD_LENGTH_ROOT_LENGTH_RATIO_ICEBERGS = 0.1874

THRESHOLD_PERIMETER_INDEX = 2
THRESHOLD_LENGTH_ROOT_LENGTH_RATIO = 2


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
        self.data_s2, self.meta_s2 = self.io.read_s2(file_s2, indexes=[1, 2, 3, 4], aoi=aoi)
        self.data_s2 = self._mask_clouds(self.file_s2, aoi, self.data_s2, cloud_probability_threshold)
        ice = self._ice()

        #meta = self.meta_s2.copy()
        #meta.update(count=1, dtype=ice.dtype, driver="GTiff")
        #with rio.open("/media/henrik/DATA/ice_0.tif", "w", **meta) as dst:
            #dst.write(ice, 1)

        ice = self._flag_bright_ice_objects(ice)

        #meta = self.meta_s2.copy()
        #meta.update(count=1, dtype=ice.dtype, driver="GTiff")
        #with rio.open("/media/henrik/DATA/ice_1.tif", "w", **meta) as dst:
            #dst.write(ice, 1)

        #self.data_s2 = None
        if np.max(ice) == 0:
            return
        try:
            ice_polygons = self._to_polygons(ice, True)
        except KeyError:
            return
        ice = None
        #ice_polygons["perimeter_index"] = self._calc_perimeter_index(ice_polygons)
        #ice_polygons["perimeter_index_md"] = (ice_polygons["perimeter_index"] - MEAN_PERIMETER_INDEX_ICEBERGS) / STD_PERIMETER_INDEX_ICEBERGS
        logging.info("Classifying sea ice and icebergs")
        #ice_polygons = self._predict(ice_polygons, CLASSIFICATION_PROBABILITY_THRESHOLD)
        #ice_polygons.loc[(np.int8(ice_polygons.area >= MAXIMUM_SIZE) + np.int8(ice_polygons["perimeter_index_md"] < THRESHOLD_PERIMETER_INDEX)) > 0, "predicted_ice_feature_int"] = 1
        #ice_polygons["predicted_ice_feature_int"] = np.int8(ice_polygons["predicted_ice_feature_int"])
        #ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 0, "predicted_ice_feature_name"] = "iceberg"
        #ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 1, "predicted_ice_feature_name"] = "sea_ice"
        #logging.info("Elapsed: {} mins.".format((datetime.now() - t0).total_seconds() / 60))
        #return ice_polygons
        ice_polygons = self._calc_geometrical_parameters(ice_polygons)
        subset_mask = np.bool8(ice_polygons["area"] >= MAXIMUM_SIZE)
        subset = ice_polygons[~subset_mask]
        subset.index = list(range(len(subset)))
        large = ice_polygons[subset_mask]
        logging.info("Number of polygons: {0} / Large polygons: {1}".format(len(ice_polygons), len(large)))
        logging.info("Reading Sentinel-2 data")
        #self.data_s2, self.meta_s2 = self.io.read_s2(file_s2, indexes=[1, 2, 3, 4], aoi=aoi)
        #self.data_s2[:, cloud_mask] = np.nan
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
        subset = None
        ice_polygons.index = list(range(len(ice_polygons)))
        #subset_mask = ice_polygons["area"] >= MAXIMUM_SIZE
        #subset_mask = self._flag_uncertain_ice_objects(ice_polygons)
        #if sum(subset_mask) > 0:
            #logging.info("Classifying uncertain ice features")
            #classified = self._classify_melange(ice_polygons, subset_mask, cloud_probability_threshold)
            #ice_polygons = ice_polygons if classified is None else classified
        #logging.info("Reclassifying")
        #ice_polygons = self._reclassify(ice_polygons)
        #ice_polygons = self._reclassify_spatially(ice_polygons)
        print("Elapsed:", (datetime.now() - t0).total_seconds() / 60, "min")
        ice_polygons["predicted_ice_feature_int"] = np.int8(ice_polygons["predicted_ice_feature_int"])
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 0, "predicted_ice_feature_name"] = "iceberg"
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 1, "predicted_ice_feature_name"] = "sea_ice"
        return ice_polygons

    #def _reclassify_spatially(self, ice_polygons):
        #icebergs = ice_polygons[ice_polygons.predicted_ice_feature_int == 0] 
        #sea_ice = ice_polygons[ice_polygons.predicted_ice_feature_int == 1]
        #sea_ice_buffered = sea_ice.buffer(3000)
        #icebergs_buffered = icebergs.buffer(3000)

    def _____reclassify_spatially(self, ice_polygons):
        ice_polygons.index = list(range(len(ice_polygons)))
        subset = ice_polygons[np.max([ice_polygons["probability_iceberg"], ice_polygons["probability_sea_ice"]], 0) < 0.9]
        print(len(subset))
        t0 = datetime.now()
        ice_polygons["SPATIAL_RECLASSIFICATION"] = 0
        classes = np.int8(Parallel(n_jobs=N_JOBS)(delayed(self._do_reclassify)(row, ice_polygons) for i, row in subset.iterrows()))
        print("Reclassifying took:", (datetime.now() - t0).total_seconds() / 60, "min")
        ice_polygons["SPATIAL_RECLASSIFICATION"] = classes != np.float32(ice_polygons["predicted_ice_feature_int"])
        ice_polygons["predicted_ice_feature_int"] = classes
        return ice_polygons
    
    @staticmethod
    def _do_reclassify(row, ice_polygons):
        old_class = row["predicted_ice_feature_int"]
        neighbors = ice_polygons.clip(row["geometry"].buffer(3000))
        if len(neighbors) > 1:
            mean_proba_sea_ice = np.mean(neighbors["probability_sea_ice"])
            new_class = int(mean_proba_sea_ice >= 0.5)
            return new_class
        else:
            return old_class

    def _classify_melange(self, ice_polygons, subset_mask, cloud_probability_threshold):
        ca = "classification_algorithm"
        ice_polygons_uncertain = ice_polygons[subset_mask]  # uncertain about those, segment them and classify again
        ice_polygons_certain = ice_polygons[~subset_mask]
        logging.info("Reading S2 B2")
        self.data_s2, self.meta_s2 = self.io.read_s2(self.file_s2, indexes=[4], aoi=ice_polygons_uncertain)  # B2
        self.data_s2 = self.data_s2.squeeze()
        logging.info("Masking clouds")
        self.data_s2 = self._mask_clouds(self.file_s2, ice_polygons_uncertain, self.data_s2, cloud_probability_threshold)
        logging.info("Calculating reflectance difference")
        md_icebergs = (convolve(self.data_s2, np.ones((9, 9)) / 9 ** 2) - MEANS_ICEBERGS[3]) / STDS_ICEBERGS[3]
        logging.info("Segmenting")
        ice = self._segment_kmeans(md_icebergs)
        self.data_s2 = None
        md_icebergs = None
        try:
            ice_polygons_melange = self._to_polygons(ice, False)
        except ValueError:
            return
        ice = None
        ice_polygons_melange = self._calc_geometrical_parameters(ice_polygons_melange)
        #subset_mask = self._flag_uncertain_ice_objects(ice_polygons_melange)
        #ice_polygons_melange_uncertain = ice_polygons_melange[subset_mask]
        ice_polygons_melange.to_file("/media/henrik/DATA/test.gpkg")
        ice_polygons_melange = ice_polygons_melange[ice_polygons_melange.area < MAXIMUM_SIZE]  # drop still uncertain features
        ice_polygons_melange.index = list(range(len(ice_polygons_melange)))
        if len(ice_polygons_melange) == 0:
            return
        logging.info("Classifying sea ice and icebergs in melange")
        self.data_s2, self.meta_s2 = self.io.read_s2(self.file_s2, indexes=[1, 2, 3, 4], aoi=ice_polygons_melange)
        ice_polygons_melange = self._predict(ice_polygons_melange, CLASSIFICATION_PROBABILITY_THRESHOLD)
        ice_polygons_melange[ca] = MELANGE_AND_UNCERTAIN_ICE
        #for col in ice_polygons_melange:
            #if any([s in col for s in STATS.split(" ")]):
                #ice_polygons_melange_uncertain[col] = np.nan
        #ice_polygons_melange = gpd.GeoDataFrame(pd.concat([ice_polygons_melange, ice_polygons_melange_uncertain]))
        #ice_polygons_melange.geometry = ice_polygons_melange["geometry"]
        ice_polygons_melange.crs = ice_polygons_melange.crs
        ice_polygons_melange.index = list(range(len(ice_polygons_melange)))
        
        #if len(ice_polygons_melange) > 0:
            #icebergs = ice_polygons_melange[ice_polygons_melange["predicted_ice_feature_int"] == 0]
            #sea_ice = ice_polygons_melange[ice_polygons_melange["predicted_ice_feature_int"] == 1]
            #try:
                #sea_ice = gpd.overlay(sea_ice, icebergs, how="difference")
            #except IndexError:
                #only_sea_ice = len(sea_ice) > 0 and len(icebergs) == 0
                #ice_polygons_melange = sea_ice if only_sea_ice else icebergs
                #ice_polygons_melange["predicted_ice_feature_int"] = int(only_sea_ice)
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
            #ice_polygons_melange[ca] = MELANGE_AND_UNCERTAIN_ICE
            #ice_polygons_melange = gpd.GeoDataFrame(pd.concat([icebergs, sea_ice]))
            #ice_polygons_melange.geometry = ice_polygons_melange["geometry"]
            #ice_polygons_melange.crs = icebergs.crs
        #else:
            #ice_polygons_melange = gpd.GeoDataFrame()
        ice_polygons_certain[ca] = "OPEN_WATER"
        ice_polygons_merged = gpd.GeoDataFrame(pd.concat([ice_polygons_certain, ice_polygons_melange]))  # merge initial certain ice and ice melange
        #ice_polygons_difference = gpd.overlay(ice_polygons_uncertain, ice_polygons_merged, how="difference")
        
        #ice_polygons_uncertain[ca] = "RESIDUAL_UNCERTAIN_ICE"
        #ice_polygons_uncertain["predicted_ice_feature_int"] = 1  #np.int8(ice_polygons_difference["probability_iceberg"] < self._get_probability_threshold())
        
        #ice_polygons_merged = gpd.GeoDataFrame(pd.concat([ice_polygons_merged, ice_polygons_uncertain]))
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
        self.data_s2 = None
        ice_polygons = self._drop_nans(ice_polygons)  # rarely nan stats occur when detected ice on scene edge
        features = self._reshape_features(ice_polygons)
        model = self.io.read_model()
        logging.info("Predicting")
        t0 = datetime.now()
        proba = model.predict_proba(features)  # probabilities
        logging.info("Elapsed predicting: {}".format((datetime.now() - t0).total_seconds()))
        #ice_polygons["predicted_ice_feature_int"] = np.ones(len(ice_polygons)) * np.int8(proba[:, 0] < classification_probability_threshold)  # 0: iceberg, 1: sea ice
        condition_rf = np.int8(proba[:, 0] < classification_probability_threshold)        
        gp_condition_0 = np.int8(ice_polygons["perimeter_index_sf"] < 0.1)
        gp_condition_1 = np.int8(ice_polygons["length_root_length_ratio_sf"] < 0.1)
        gp_condition = (gp_condition_0 + gp_condition_1) > 0
        #gp_min_condition = np.int8(geometric_params.min(0) < 0.6)
        ice_polygons["predicted_ice_feature_int"] = np.int8((condition_rf + gp_condition) > 0)  # if 1: sea ice
        #ice_polygons["predicted_ice_feature_int"] = np.int8(np.mean([proba[:, 0], gp_mean], axis=0) < classification_probability_threshold)
        #ice_polygons.loc[np.bool8(proba[:, 0] >= 0.9), "predicted_ice_feature_int"] = 0
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 0, "predicted_ice_feature_name"] = "iceberg"
        ice_polygons.loc[ice_polygons["predicted_ice_feature_int"] == 1, "predicted_ice_feature_name"] = "sea_ice"
        ice_polygons["probability_iceberg"] = proba[:, 0]
        ice_polygons["probability_sea_ice"] = proba[:, 1]
        return ice_polygons

    def _ice(self):
        return np.int8(self.data_s2[0] >= B8_THRESHOLD)

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
        logging.info("Polygonizing..")
        t0 = datetime.now()
        polygons = Parallel(n_jobs=N_JOBS)(delayed(self._do_polygonize)(contour) for contour in measure.find_contours(ice, level=0.5))
        polygons = gpd.GeoDataFrame(geometry=np.array(polygons)[~np.bool8([p.is_empty for p in polygons])], crs=self.meta_s2["crs"]).dissolve().explode(index_parts=True)
        logging.info("measure.find_contours - Polygonizing took {} minutes".format((datetime.now() - t0).total_seconds() / 60))
        polygons.geometry = polygons["geometry"]
        polygons = polygons[np.bool8(polygons.area >= (MINIMUM_SIZE * 10 ** 2))]
        polygons.crs = self.meta_s2["crs"]
        polygons.index = list(range(len(polygons)))
        polygons["area"] = polygons.area
        return polygons

    def _do_polygonize(self, contour):
        transformed_coords = [self.meta_s2["transform"] * (x, y) for y, x in contour]  # (y, x) from skimage, so reverse to (x, y)            
        try:
            poly = Polygon(transformed_coords).buffer(0.01)
        except ValueError:
            return Polygon
        return poly

    def _extract_statistics(self, ice_polygons):
        band_names = ["b8", "b4", "b3", "b2"]
        t0 = datetime.now()
        ice_polygons_copy = ice_polygons.copy()
        try:
            stats_dfs_all_bands = Parallel(n_jobs=N_JOBS)(delayed(self._do_extract_statistics)(ice_polygons_copy, self.data_s2[band_idx], band_names[band_idx], self.meta_s2["transform"]) for band_idx in [0, 1, 2, 3])
        except:
            logging.info("Parallel failed. Trying with one core")
            stats_dfs_all_bands = []
            for band_idx in range(len(band_names)):
                stats_dfs_all_bands.append(self._do_extract_statistics(ice_polygons, self.data_s2[band_idx], band_names[band_idx], self.meta_s2["transform"]))
        logging.info("Elapsed extract statistics: {}".format((datetime.now() - t0).total_seconds() / 60))
        t0 = datetime.now()
        self.data_s2 = None
        for _, df in enumerate(stats_dfs_all_bands):
            for col in df:
                if col not in list(ice_polygons.columns):
                    ice_polygons[col] = df[col]  # stat column
        #for band_idx, stats in enumerate(gdfs_all_bands):
            #for i, stat in enumerate(stats):
                #for key, value in stat.items():
                    #key = key.replace("percentile_", "p")
                    #ice_polygons.loc[i, "_".join([band_names[band_idx], key])] = value
        logging.info("Elasped register statistics: {}".format((datetime.now() - t0).total_seconds()))
        return ice_polygons
    
    def _flag_bright_ice_objects(self, ice):
        labels = measure.label(ice == 1)
        logging.info("Initial number of ice objects: {}".format(len(np.unique(labels))))
        diff_b2 = (self.data_s2[3] - MEANS_ICEBERGS[3]) / STDS_ICEBERGS[3]
        values = np.unique(labels[diff_b2 >= 1].flatten())
        ice = np.int8(np.isin(labels, values[values != 0]))
        labels = measure.label(ice == 1)
        logging.info("Filtered number of ice objects: {}".format(len(np.unique(labels))))
        return ice

    def _calc_geometrical_parameters(self, polygons):
        polygons["length"] = Parallel(n_jobs=N_JOBS)(delayed(self._calculate_length)(polygon) for polygon in polygons.geometry)
        polygons["perimeter_index"] = self._calc_perimeter_index(polygons)
        polygons["perimeter_index_md"] = self._calc_perimeter_index_md(polygons)
        polygons["perimeter_index_sf"] = chi2.sf(polygons["perimeter_index_md"] ** 2, 1)
        polygons["length_root_length_ratio"] = self._calc_length_root_length_ratio(polygons)
        polygons["length_root_length_ratio_md"] = self._calc_length_root_length_ratio_md(polygons)
        polygons["length_root_length_ratio_sf"] = chi2.sf(polygons["length_root_length_ratio_md"] ** 2, 1)
        return polygons

    @staticmethod
    def _do_extract_statistics(ice_polygons, band, band_name, transform):
        #band_data, meta = self.io.read_s2(self.file_s2, [idx], aoi=ice_polygons)
        #band_data = np.float16(band_data.squeeze())
        stats = pd.DataFrame(zonal_stats(ice_polygons, band, affine=transform, stats=STATS, nodata=np.nan))
        for col in stats:
            ice_polygons["_".join([band_name, col.replace("percentile_", "p")])] = stats[col]
        return ice_polygons

    @staticmethod
    def _flag_uncertain_ice_objects(ice_polygons):
        #return ice_polygons.area >= MAXIMUM_SIZE
        return np.max(
            [ice_polygons.area >= MAXIMUM_SIZE,
            ice_polygons["perimeter_index_sf"] < 0.2,
            ice_polygons["length_root_length_ratio_sf"] < 0.2
            ], axis=0
        ) == 1

    @staticmethod
    def _calc_perimeter_index_md(polygons):
        diff = (polygons["perimeter_index"] - MEAN_PERIMETER_INDEX_ICEBERGS) / STD_PERIMETER_INDEX_ICEBERGS  # only negative difference
        diff = np.clip(diff, -1000, 0)  # clip at 0
        return np.abs(diff)

    @staticmethod
    def _calc_length_root_length_ratio_md(polygons):
        return np.abs(polygons["length_root_length_ratio"] - MEAN_LENGTH_ROOT_LENGTH_RATIO_ICEBERGS) / STD_LENGTH_ROOT_LENGTH_RATIO_ICEBERGS        

    @staticmethod
    def _calc_length_root_length_ratio(polygons):
        return np.float32(polygons["length"] / np.sqrt(polygons.area))

    @staticmethod
    def _segment_kmeans(md_icebergs):
        prediction = md_icebergs.copy()
        md_icebergs_flat = md_icebergs.flatten()
        md_icebergs_flat = md_icebergs_flat[~np.isnan(md_icebergs_flat)].reshape(-1, 1)
        kmeans = KMeans(3, n_init=10, random_state=RANDOM_STATE).fit(md_icebergs_flat)
        prediction[~np.isnan(prediction)] = kmeans.predict(md_icebergs_flat)
        values = np.unique(prediction[np.isfinite(prediction)])
        max_cluster = values[np.argmax([np.nanmean(md_icebergs[prediction == value]) for value in values])]
        prediction = np.int8(prediction == max_cluster)
        return prediction

    @staticmethod
    def _calculate_length(polygon):
        polygon = polygon.simplify(20)
        line_lengths = []
        for p in polygon.exterior.coords:
            for p1 in polygon.exterior.coords:
                line_lengths.append(LineString([p, p1]).length)
        return np.nanmax(line_lengths)

    @staticmethod
    def _drop_nans(ice_polygons):
        for col in FEATURE_NAMES:
            ice_polygons = ice_polygons[~np.isnan(ice_polygons[col])]
        ice_polygons.index = list(range(len(ice_polygons)))
        return ice_polygons

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
            cloud_proba, _ = mask(src, list(aoi.to_crs(src.crs).geometry), crop=True, indexes=5)
        cloud_mask = cloud_proba.squeeze() >= cloud_probability_threshold
        if len(data_s2.shape) == 3:
            data_s2[:, cloud_mask] = np.nan
        else:
            data_s2[cloud_mask] = np.nan
        cloud_mask, cloud_proba = None, None
        return data_s2

    @staticmethod
    def _merge_touching_polygons(gdf):
        geoms = gpd.GeoSeries(gdf.geometry.buffer(0.1).unary_union.buffer(-0.1)).explode(index_parts=False)
        gdf_merged = gpd.GeoDataFrame({"geometry": list(geoms.geometry)})
        gdf_merged.geometry = gdf_merged.geometry
        gdf_merged.crs = gdf.crs
        gdf_merged.index = list(range(len(gdf_merged)))
        return gdf_merged
