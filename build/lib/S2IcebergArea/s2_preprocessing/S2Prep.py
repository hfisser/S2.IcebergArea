import os
import numpy as np
import rasterio as rio
from glob import glob
from osgeo import gdal
from scipy.ndimage import convolve
from S2IcebergArea.io.IO import IO
from xml.etree import ElementTree as ET
from s2cloudless.cloud_detector import S2PixelCloudDetector

S2_BANDS = np.hstack([["B0{}".format(i) for i in range(1, 9)], "B8A", "B09", ["B{}".format(i) for i in range(10, 13)]])
S2_CLOUD_MASKING_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]
S2_10M_BANDS = ["B08", "B04", "B03", "B02"]  # 10 m res.


class S2Prep:
    def __init__(self) -> None:
        self.io = IO()

    def preprocess_s2(self, dir_safe, dir_out, bands):
        dn_offset = self._get_radiometric_offset(dir_safe)
        files = self._s2_band_files(dir_safe, bands)  # jp2 files of 10m bands
        crs = self._get_crs(list(files.values())[0])
        files_all = self._s2_band_files(dir_safe, None)  # jp2 files of all bands for cloud masking
        file_s2 = os.path.join(dir_out, "{}.jp2".format(os.path.basename(dir_safe.split(".SAFE")[0])))
        cloud_prb, meta = self._calc_clouds(files_all, dir_out, dn_offset)
        file_cloud_prb = os.path.join(dir_out, os.path.basename(file_s2).replace(".jp2", "_s2cloudless_prb.jp2"))
        with rio.open(file_cloud_prb, "w", **meta) as tgt:
            tgt.write(cloud_prb, 1)
        meta = self.io._data_to_jp2(list(files.values()), file_s2)
        file_resampled = self._resample(file_cloud_prb, file_s2, dir_out)  # to 10 m resolution
        with rio.open(file_resampled) as src:  # read cloud probabilities
            cloud_prb_resampled = src.read(1)
        with rio.open(file_s2) as src:  # read reflectances
            descriptions = list(src.descriptions) + ["s2cloudless_prb"]
            meta, stack = src.meta, src.read()
        stack = np.vstack([stack + dn_offset, np.expand_dims(cloud_prb_resampled, 0)])  # stack reflectances and cloud probabilities
        meta.update(count=stack.shape[0], crs=crs)
        with rio.open(file_s2, "w", **meta) as tgt:  # stack
            for j in range(stack.shape[0]):
                tgt.write(stack[j], j + 1)
                tgt.set_band_description(j + 1, descriptions[j])  # band name
        os.remove(file_cloud_prb)  # cloud prb
        os.remove(file_resampled)  # cloud prb resampled
        return file_s2

    def _calc_clouds(self, files_bands, dir_out, dn_offset):
        try:
            for exclude in ["WVP", "TCI", "SCL", "AOT"]:
                del files_bands[exclude]
        except KeyError:
            pass
        file_reference = files_bands[S2_BANDS[0]]  # B1 as reference for resampling (downsampling)
        with rio.open(file_reference, "r") as src:
            stack = [src.read(1) + dn_offset]
            meta = src.meta
        for idx in S2_CLOUD_MASKING_IDS[1:]:
            file_resampled = self._resample(files_bands[S2_BANDS[idx]], file_reference, dir_out)  # 60 m res.
            with rio.open(file_resampled, "r") as src:
                meta = src.meta
                stack.append(src.read(1) + dn_offset)  # all at same resolution
            os.remove(file_resampled)  # remove resampled band
        stack = np.expand_dims(np.float32(stack).swapaxes(0, 2).swapaxes(0, 1), 0)  # n acquisitions, y, x, n bands
        detector = S2PixelCloudDetector(all_bands=False, average_over=3)  # s2cloudless
        cloud_prb = convolve(np.uint16(np.multiply(detector.get_cloud_probability_maps(np.multiply(stack, 1e-4))[0], 100)), detector.conv_filter)  # average cloud probabilities over 3 60m pixels
        meta.update(height=cloud_prb.shape[0], width=cloud_prb.shape[1], dtype=cloud_prb.dtype, driver="JP2OpenJPEG")
        return cloud_prb, meta

    @staticmethod
    def _get_crs(file_img):
        return str(list(ET.parse(glob(os.path.join(os.path.dirname(os.path.dirname(file_img)), "MTD_TL.xml"))[0]).getroot().iter("HORIZONTAL_CS_CODE"))[0].text)

    @staticmethod
    def _get_radiometric_offset(dir_safe):
        return np.int16(list(ET.parse(os.path.join(dir_safe, "MTD_MSIL1C.xml")).getroot().iter("RADIO_ADD_OFFSET"))[0].text)

    @staticmethod
    def _resample(file, file_reference, dir_out, overwrite=True):
        """
        Resamples to different grid, using average resampling in GDAL.
        :param file: str file to resample.
        :param file_reference: str file to match.
        :param dir_out: str directory where to save the resampled file.
        :param overwrite: bool whether to overwrite file, if false, nothing will be processed.
        """
        base_name = os.path.basename(file).split(".")[0]
        file_out = os.path.join(dir_out, f"{base_name}_resampled.tif")
        if os.path.exists(file_out):
            if overwrite:
                os.remove(file_out)
            else:
                return file_out
        match = gdal.Open(file_reference, gdal.gdalconst.GA_ReadOnly)
        match_proj = match.GetProjection()
        match_geot = match.GetGeoTransform()
        match_width = match.RasterXSize
        match_height = match.RasterYSize
        with rio.open(file) as src_ds:
            n_bands = src_ds.count
        dst = gdal.GetDriverByName("GTiff").Create(file_out, match_width, match_height, n_bands, gdal.gdalconst.GDT_Float32)
        dst.SetGeoTransform(match_geot)
        dst.SetProjection(match_proj)
        src = gdal.Open(file, gdal.gdalconst.GA_ReadOnly)
        src_proj = src.GetProjection()
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdal.gdalconst.GRA_Average)  # average resampling
        dst = None
        return file_out

    @staticmethod
    def _s2_band_files(dir_parse, bands):  # to parse L2A .SAFE archive
        while not any(["GRANULE" in d for d in glob(os.path.join(dir_parse, "*"))]):
            dir_parse = glob(os.path.join(dir_parse, "*"))[0]  # navigate to GRANULE
        # files of 10m bands
        files_img = glob(os.path.join(glob(os.path.join(dir_parse, "GRANULE", "*"))[0], "IMG_DATA", "*.jp2"))
        if len(files_img) == 0:
            files_img = glob(os.path.join(glob(os.path.join(dir_parse, "GRANULE", "*"))[0], "IMG_DATA", "*.jp2"))
        files_img_dict = {file.split("_")[-1].split(".")[0]: file for file in files_img}
        return files_img_dict if bands is None else {band: files_img_dict[band] for band in bands}
