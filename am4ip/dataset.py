
import numpy as np
from glob import glob
from PIL import Image
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple

from am4ip.utils import expanded_join, random_crop


class TID2013(Dataset):
    """Class utility to load, pre-process, put in batch, and convert to PyTorch convention images from the TID2013 dataset.
    """
    N_reference = 25
    N_distortion = 24
    N_params = 5

    root_path = "C:\\RESOURCES\\datasets\\tid2013"
    # root_path = "/net/ens/am4ip/datasets/tid2013"

    def __init__(self, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 crop_size: Optional[Tuple[int, int]] = None) -> None:
        """ Class initialization.

        :param transform: A set of transformations to apply on data.
        :param target_transform: A set of transformations to apply on labels.
        """
        # Get all distorted images
        self.dist_image_paths = glob(expanded_join(self.root_path, "distorted_images/*"))
        idx = np.argsort([p.lower() for p in self.dist_image_paths])
        self.dist_image_paths = np.array(self.dist_image_paths)[idx]

        # Get all reference images
        self.ref_image_paths = glob(expanded_join(self.root_path, "reference_images/*.BMP"))
        self.ref_image_paths.sort()
        self.ref_image_paths = np.array(self.ref_image_paths)

        # Get mean opinion score for each distorted image
        self.mos = np.load(expanded_join(self.root_path, "MOS.npy"))

        self.transform = transform
        self.target_transform = target_transform

        self.crop_size = crop_size
        self.do_crop = False if crop_size is None else True

    def __len__(self):
        """Dataset size.
        :return: Size of the dataset.
        """
        return self.N_reference * self.N_distortion * self.N_params

    def __getitem__(self, index: NDArray[int] or int) -> Tuple[Tuple[NDArray, NDArray], NDArray]:
        dist_img = Image.open(self.dist_image_paths[index])

        rid = index // (self.N_distortion * self.N_params)
        ref_img = Image.open(self.ref_image_paths[rid])

        if self.do_crop:
            ref_img, dist_img = random_crop(ref_img, dist_img, crop_size=self.crop_size)
        # Transforms are any Callable. For simplicity, we keep PIL.Image object, so that
        # transforms from PIL library can be applied (e.g., color mode conversion, resize, ...)
        # before conversion to torch.tensors.
        if self.transform is not None:
            ref_img = self.transform(ref_img)
            dist_img = self.transform(dist_img)

        y = self.mos[index]
        if self.target_transform is not None:
            y = self.target_transform(y)

        return (dist_img, ref_img), y
