import os
import pickle
import importlib
import numpy as np
import rasterio as rio
from datetime import datetime
from rasterio.mask import mask
from joblib import Parallel, delayed
from rasterio.enums import Resampling

FILE_MODEL = "IcebergBlues_RFC_model.pickle"
REFLECTANCE_SCALING = 1e-4


class IO:
    def __init__(self) -> None:
        pass

    #@staticmethod
    def _read_s2(self, file_s2, indexes, aoi=None):
        t0 = datetime.now()
        data_list = Parallel(n_jobs=4)(delayed(self._do_read_s2)(file_s2, i, aoi) for i in indexes)
        print("Reading data took:", (datetime.now() - t0).total_seconds() / 60)
        data = np.float32([data_band[0] for data_band in data_list]) * REFLECTANCE_SCALING
        with rio.open(file_s2) as src:
            meta = src.meta
        meta.update(transform=data_list[0][1], height=data.shape[-2], width=data.shape[-1])
        data_list = None
        #with rio.open(file_s2) as src:
            #meta = src.meta
            #data, transform = mask(src, list(aoi.to_crs(meta["crs"]).geometry), crop=True, nodata=0, indexes=indexes)
            #meta.update(transform=transform, height=data.shape[1], width=data.shape[2])
        #data = np.float32(data * REFLECTANCE_SCALING)  # scale to 0-1 range
        data[data == 0] = np.nan  # new nodata value
        data[:, np.sum(~np.isfinite(data), 0) > 0] = np.nan  # new nodata value
        data[data > 2] = np.nan
        return data, meta

    @staticmethod
    def read_s2(file_s2, indexes, aoi=None):
        with rio.open(file_s2) as src:
            meta = src.meta
            if aoi is None:
                data = src.read(indexes)
            else:
                data, transform = mask(src, list(aoi.to_crs(meta["crs"]).geometry), crop=True, nodata=0, indexes=indexes)
                meta.update(transform=transform, height=data.shape[1], width=data.shape[2])
        data = np.float32(data * REFLECTANCE_SCALING)  # scale to 0-1 range
        data[data == 0] = np.nan  # new nodata value
        data[:, np.sum(~np.isfinite(data), 0) > 0] = np.nan  # new nodata value
        data[data > 1] = np.nan
        return data, meta

    @staticmethod
    def _do_read_s2(file_s2, i, aoi):
        with rio.open(file_s2) as src:
            data, transform = mask(src, list(aoi.to_crs(src.meta["crs"]).geometry), crop=True, nodata=0, indexes=i)
        return data, transform

    @staticmethod
    def _data_to_jp2(files_bands, file_out):
        stack = []
        for file_band in files_bands:  # read band data from separate files
            with rio.open(file_band, "r") as src:
                meta = src.meta
                stack.append(np.uint16(src.read(1)))
        meta.update(count=len(stack), driver="JP2OpenJPEG", dtype=np.uint16)
        with rio.open(file_out, "w", **meta) as tgt:  # write masked and cropped band to GTiff
            for i, file in enumerate(files_bands):  # file names only used to set band description
                idx = i + 1
                stack[i][stack[i] > 10000] = 10000
                stack[i][stack[i] > 12000] = 0
                tgt.write(stack[i].astype(meta["dtype"]), idx)
                tgt.set_band_description(idx, os.path.basename(file).split(".")[0])  # band name
        return meta

    @staticmethod
    def read_model():
        with importlib.resources.path("S2IcebergArea", "model") as dir_model:
            with open(os.path.join(dir_model, FILE_MODEL), "rb") as src:
                model = pickle.load(src)
        return model
