import os
import pickle
import importlib
import numpy as np
import rasterio as rio
from rasterio.mask import mask

FILE_MODEL = "IcebergBlues_RFC_model.pickle"
REFLECTANCE_SCALING = 1e-4


class IO:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read_s2(file_s2, indexes, aoi=None):
        with rio.open(file_s2) as src:
            meta = src.meta
            if aoi is None:
                data = src.read(indexes)
            else:
                data, transform = mask(src, list(aoi.geometry), crop=True, nodata=0)
                meta.update(transform=transform, height=data.shape[0], width=data.shape[1])
        data = np.float32(data * REFLECTANCE_SCALING)  # scale to 0-1 range
        data[data == 0] = np.nan  # new nodata value
        return data, meta

    @staticmethod
    def read_model():
        with importlib.resources.path("S2IcebergArea", "model") as dir_model:
            with open(os.path.join(dir_model, FILE_MODEL), "rb") as src:
                model = pickle.load(src)
        return model
