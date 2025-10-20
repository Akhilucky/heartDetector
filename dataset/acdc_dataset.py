import os
import torch
import nibabel as nib
import numpy as np
import torchio as tio
from torch.utils.data import Dataset

class ACDCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        for p in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, p)
            if not os.path.isdir(patient_path):
                continue
            for f in os.listdir(patient_path):
                if f.endswith(".nii.gz") and "gt" not in f:
                    img_path = os.path.join(patient_path, f)
                    gt_path = img_path.replace(".nii.gz", "_gt.nii.gz")
                    if os.path.exists(gt_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(gt_path)

        print(f"Found {len(self.image_paths)} ACDC volumes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.int64)

        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        slices = []
        slice_masks = []
        for i in range(img.shape[2]):
            im = img[:, :, i]
            ms = mask[:, :, i]
            if np.max(ms) == 0:
                continue
            im = np.expand_dims(im, axis=0)
            ms = np.expand_dims(ms, axis=0)
            slices.append(im)
            slice_masks.append(ms)

        idx = np.random.randint(0, len(slices))
        img_slice, mask_slice = slices[idx], slice_masks[idx]

        sample = {'image': torch.tensor(img_slice, dtype=torch.float32),
                  'mask': torch.tensor(mask_slice, dtype=torch.long).squeeze(0)}

        if self.transform:
            tio_subject = tio.Subject(
                img=tio.ScalarImage(tensor=sample['image'].unsqueeze(0)),
                mask=tio.LabelMap(tensor=sample['mask'].unsqueeze(0))
            )
            transformed = self.transform(tio_subject)
            sample['image'] = transformed.img.data.squeeze(0)
            sample['mask'] = transformed.mask.data.squeeze(0)

        return sample['image'], sample['mask']
