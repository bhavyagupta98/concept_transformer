import albumentations as A
import os
import pytorch_lightning as pl
from albumentations import HorizontalFlip, Normalize
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, random_split

from .cub2011parts import CUB2011Parts_dataset


class CUB2011Parts(pl.LightningDataModule):

    def __init__(self, batch_size: int = 32, num_workers: int = 8,
                 data_dir: str = "~/data/cub2011", **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = A.Compose([Resize(224, 224),
                                          HorizontalFlip(p=0.5),
                                          Rotate(limit=(-30,30),p=1.0),
                                          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                          ToTensorV2()],
                                         keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

        self.test_transform = A.Compose([Resize(224, 224),
                                         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                         ToTensorV2()],
                                        keypoint_params = A.KeypointParams(format='xy', remove_invisible=False))

    def _resolve_root(self):
        data_root = os.path.expanduser(self.data_dir)

        # Support both:
        # - data_dir = /workspace/data            -> expects /workspace/data/CUB_200_2011
        # - data_dir = /workspace/data/CUB_200_2011 -> treat parent as the dataset root
        if os.path.basename(data_root) == 'CUB_200_2011':
            parent_root = os.path.dirname(data_root)
            if os.path.exists(os.path.join(parent_root, 'CUB_200_2011')):
                return parent_root

        return data_root

    def _normalize_extracted_layout(self, data_root):
        cub_root = os.path.join(data_root, 'CUB_200_2011')
        root_attributes = os.path.join(data_root, 'attributes.txt')
        cub_attributes = os.path.join(cub_root, 'attributes.txt')

        # When the archive is manually extracted, attributes.txt can end up in the
        # dataset root instead of inside CUB_200_2011/. Move it into the expected place.
        if os.path.exists(root_attributes) and not os.path.exists(cub_attributes):
            os.makedirs(cub_root, exist_ok=True)
            os.replace(root_attributes, cub_attributes)

    def prepare_data(self):
        # Verify that the extracted dataset exists (manual download required).
        data_root = self._resolve_root()
        self._normalize_extracted_layout(data_root)
        cub_root = os.path.join(data_root, 'CUB_200_2011')
        
        # Check if dataset folder exists
        if not os.path.exists(cub_root):
            raise FileNotFoundError(
                f"CUB_200_2011 dataset folder not found at {cub_root}\n\n"
                f"Please download the dataset manually from:\n"
                f"  http://www.vision.caltech.edu/datasets/cub_200_2011/\n\n"
                f"Then extract it to:\n"
                f"  {data_root}/CUB_200_2011/\n\n"
                f"Extraction command (Linux/macOS):\n"
                f"  tar -xzf CUB_200_2011.tgz -C {data_root}/\n"
                f"Or if downloaded as .zip:\n"
                f"  unzip CUB_200_2011.zip -d {data_root}/ && tar -xzf {data_root}/CUB_200_2011.tgz -C {data_root}/"
            )
        
        # Verify required files exist
        required_files = [
            'images.txt',
            'image_class_labels.txt',
            'train_test_split.txt',
            'bounding_boxes.txt',
            os.path.join('parts', 'part_locs.txt'),
            os.path.join('attributes', 'image_attribute_labels.txt'),
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(cub_root, f))]
        if missing_files:
            raise FileNotFoundError(
                f"CUB_200_2011 dataset is incomplete. Missing files:\n  "
                + "\n  ".join(missing_files) + 
                f"\n\nPlease verify extraction to {cub_root}/"
            )

    def setup(self, stage=None):
        data_root = self._resolve_root()
        self._normalize_extracted_layout(data_root)
        trainset = CUB2011Parts_dataset(train=True, root=data_root, transform=self.train_transform)
        testset = CUB2011Parts_dataset(train=False, root=data_root, transform=self.test_transform)

        self.num_classes = trainset.num_classes

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cub_train = trainset
            self.cub_val, _ = random_split(testset, [1000, len(testset) - 1000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cub_test = testset

        # self.dims is returned when you call dm.size()
        self.dims = trainset[0][0].shape

    def train_dataloader(self):
        return DataLoader(self.cub_train, shuffle=True, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.cub_val, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cub_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.cub_test, shuffle=False, batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == '__main__':
    data_dir = '../../../data'
    explanation = CUB2011Parts(data_dir=data_dir)
    explanation.prepare_data()

    explanation.setup()
    train_dl = explanation.train_dataloader()
    val_dl = explanation.val_dataloader()
    test_dl = explanation.test_dataloader()
    print(f"Dataset split (train/val/test): {len(train_dl.dataset)}/{len(val_dl.dataset)}/{len(test_dl.dataset)}")
