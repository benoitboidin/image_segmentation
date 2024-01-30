
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Callable

from am4ip.utils import expanded_join


class CropSegmentationDataset(Dataset):
    ROOT_PATH: str = "project_dataset"
    id2cls: dict = {0: "background",
                    1: "crop",
                    2: "weed",
                    3: "partial-crop",
                    4: "partial-weed"}
    cls2id: dict = {"background": 0,
                    "crop": 1,
                    "weed": 2,
                    "partial-crop": 3,
                    "partial-weed": 4}

    def __init__(self, set_type: str = "train", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 merge_small_items: bool = True,
                 remove_small_items: bool = False,
                 augmentations: Optional[Callable] = None): # Augmentation
        """Class to load datasets for the Project.

        Remark: `target_transform` is applied before merging items (this eases data augmentation).

        :param set_type: Define if you load training, validation or testing sets. Should be either "train", "val" or "test".
        :param transform: Callable to be applied on inputs.
        :param target_transform: Callable to be applied on labels.
        :param merge_small_items: Boolean to either merge classes of small or occluded objects.
        :param remove_small_items: Boolean to consider as background class small or occluded objects. If `merge_small_items` is set to `True`, then this parameter is ignored.
        """
        super(CropSegmentationDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.merge_small_items = merge_small_items
        self.remove_small_items = remove_small_items
        self.augmentations = augmentations # Augmentation

        if set_type not in ["train", "val", "test"]:
            raise ValueError("'set_type has an unknown value. "
                             f"Got '{set_type}' but expected something in ['train', 'val', 'test'].")

        self.set_type = set_type
        images = glob(expanded_join(self.ROOT_PATH, set_type, "images/*"))
        images.sort()
        self.images = np.array(images)

        labels = glob(expanded_join(self.ROOT_PATH, set_type, "labels/*"))
        labels.sort()
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        input_img = Image.open(self.images[index], "r")
        target = Image.open(self.labels[index], "r")

        if self.augmentations is not None:
            input_img, target = self.apply_augmentation(input_img, target)

        if self.transform is not None:
            input_img = self.transform(input_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.merge_small_items:
            target[target == self.cls2id["partial-crop"]] = self.cls2id["crop"]
            target[target == self.cls2id["partial-weed"]] = self.cls2id["weed"]
        elif self.remove_small_items:
            target[target == self.cls2id["partial-crop"]] = self.cls2id["background"]
            target[target == self.cls2id["partial-weed"]] = self.cls2id["background"]

        return input_img, target

    def get_class_number(self):
        if self.merge_small_items or self.remove_small_items:
            return 3
        else:
            return 5

    def apply_augmentation(self, input_img, target):
        augmentation_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        input_img = augmentation_pipeline(input_img)
        target = augmentation_pipeline(target)

        return input_img, target