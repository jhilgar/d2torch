import os
from typing import Any, Callable, Dict, Optional, Tuple, List

import torch
from numpy import int64
from torchvision.datasets import VisionDataset, voc
from torchvision.transforms import functional as F
from PIL import Image
from xml.etree.ElementTree import parse as ET_parse

class VOCCustom(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.transforms = transforms
        self.images, self.annotations = [], []

        for file in os.listdir(root):
            name, _ = os.path.splitext(file)
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                if os.path.exists(os.path.join(root, name + ".xml")):
                    self.images.append(os.path.join(root, file))
                    self.annotations.append(os.path.join(root, name + ".xml"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        target = voc.VOCDetection.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        target = self.voc_to_coco(target, index)

        if self.transforms is not None:
            img, target = self.transforms(img), target

        return img, target

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def voc_to_coco(target: dict, index: int64) -> dict:
        coco_dict: Dict[str, Any] = {}
        coco_dict["image_id"] = torch.tensor([index])
        boxes, labels, area, iscrowd = [], [], [], []
        for object in target["annotation"]["object"]:
            boxes.append([int(x) for x in list(object["bndbox"].values())])
            labels.append(object["name"])
            iscrowd.append(int(object["difficult"]))

        d = dict([(y, x + 1) for x, y in enumerate(sorted(set(labels)))])
        coco_dict["labels"] = torch.as_tensor([d[x] for x in labels], dtype = torch.int64)
        coco_dict["boxes"] = torch.as_tensor(boxes, dtype = torch.float32)
        coco_dict["area"] = \
            (coco_dict["boxes"][:, 3] - coco_dict["boxes"][:, 1]) * \
            (coco_dict["boxes"][:, 2] - coco_dict["boxes"][:, 0])
        coco_dict["iscrowd"] = torch.as_tensor(iscrowd, dtype = torch.uint8)

        return coco_dict