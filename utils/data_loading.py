import os

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BiplaneDSADataset(Dataset):

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        csv_path,
        root_dir,
        transform=None,
        input_type="sequence",
        scale=None,
    ):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.input_type = input_type
        self.scale = scale

        if self.input_type not in {"sequence", "minip"}:
            raise ValueError(
                f"Unknown input type: {self.input_type}"
            )

        self.view_pairs = self._build_view_pairs()

    def __len__(self):
        return len(self.view_pairs)

    def __getitem__(self, idx):
        ap_row, lat_row = self.view_pairs[idx]

        patient_id = ap_row["Patient ID"]
        raw_score = ap_row["Score"]

        score = torch.tensor(
            int(raw_score - 1),
            dtype=torch.long,
        )

        if self.input_type == "sequence":
            ap, lat = self._load_sequence_pair(
                ap_row,
                lat_row,
            )
        else:
            ap, lat = self._load_minip_pair(
                ap_row,
                lat_row,
            )

        file_names = (
            ap_row["File Name"],
            lat_row["File Name"],
        )

        return (
            ap,
            lat,
            score,
            file_names,
            patient_id,
        )

    def _build_view_pairs(self):
        view_pairs = []

        grouped = self.data.groupby(
            ["Patient ID", "Series", "Frames"]
        )

        for _, group in grouped:
            ap_rows = group[
                group["View Type"] == "Anterior-Posterior"
            ]
            lat_rows = group[
                group["View Type"] == "Lateral"
            ]

            if ap_rows.empty or lat_rows.empty:
                continue

            ap_row = ap_rows.iloc[0]
            lat_row = lat_rows.iloc[0]

            if self.input_type == "minip":
                ap_path = self._construct_png_path(ap_row)
                lat_path = self._construct_png_path(lat_row)
            else:
                ap_path = self._construct_nifti_path(ap_row)
                lat_path = self._construct_nifti_path(lat_row)

            if os.path.exists(ap_path) and os.path.exists(lat_path):
                view_pairs.append(
                    (ap_row, lat_row)
                )

        return view_pairs

    def _construct_nifti_path(self, row):
        file_base = self._get_file_base(row)
        patient_id = str(row["Patient ID"])
        view = str(row["View Type"])

        return os.path.join(
            self.root_dir,
            f"{patient_id}_{view}_{file_base}.nii",
        )

    def _construct_png_path(self, row):
        file_base = self._get_file_base(row)
        patient_id = str(row["Patient ID"])
        view = str(row["View Type"])

        return os.path.join(
            self.root_dir,
            f"{patient_id}_{view}_{file_base}.png",
        )

    @staticmethod
    def _get_file_base(row):
        return str(row["File Name"]).replace(
            ".dcm",
            "",
        )

    def _load_nifti(self, row):
        path = self._construct_nifti_path(row)

        image = nib.load(path).get_fdata()
        image = image.squeeze(axis=2)
        image = np.transpose(
            image,
            (2, 1, 0),
        )

        return image.copy()

    def _load_sequence_pair(
        self,
        ap_row,
        lat_row,
    ):
        ap = self._load_nifti(ap_row)
        lat = self._load_nifti(lat_row)

        sequence_length = min(
            ap.shape[0],
            lat.shape[0],
        )

        if sequence_length <= 2:
            raise ValueError(
                "Sequence is too short after removing initial frames."
            )

        ap = ap[2:sequence_length]
        lat = lat[2:sequence_length]

        ap = np.clip(
            ap,
            0,
            255,
        ).astype(np.uint8)

        lat = np.clip(
            lat,
            0,
            255,
        ).astype(np.uint8)

        ap = self._resize_sequence(ap)
        lat = self._resize_sequence(lat)

        ap = np.transpose(
            ap,
            (1, 2, 0),
        )
        lat = np.transpose(
            lat,
            (1, 2, 0),
        )

        ap, lat = self._apply_paired_transform(
            ap,
            lat,
        )

        ap = np.asarray(ap)
        lat = np.asarray(lat)

        ap = np.transpose(
            ap,
            (2, 0, 1),
        )
        lat = np.transpose(
            lat,
            (2, 0, 1),
        )

        ap = self._sequence_to_tensor(ap)
        lat = self._sequence_to_tensor(lat)

        ap = self._normalize_sequence(ap)
        lat = self._normalize_sequence(lat)

        return (
            ap.float().contiguous(),
            lat.float().contiguous(),
        )

    def _load_minip_pair(
        self,
        ap_row,
        lat_row,
    ):
        ap = self._load_png(ap_row)
        lat = self._load_png(lat_row)

        ap, lat = self._apply_paired_transform(
            ap,
            lat,
        )

        ap = self._minip_to_tensor(ap)
        lat = self._minip_to_tensor(lat)

        ap = self._normalize_image(ap)
        lat = self._normalize_image(lat)

        return (
            ap.float().contiguous(),
            lat.float().contiguous(),
        )

    def _load_png(self, row):
        path = self._construct_png_path(row)

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {path}"
            )

        image = (
            image.astype(np.float32)
            / 255.0
        )

        if self.scale is not None:
            image = self._resize_image(image)

        return image

    def _resize_sequence(self, sequence):
        if self.scale is None:
            return sequence

        num_frames, height, width = sequence.shape

        new_height = int(
            height * self.scale
        )
        new_width = int(
            width * self.scale
        )

        self._validate_size(
            new_height,
            new_width,
        )

        resized = np.empty(
            (
                num_frames,
                new_height,
                new_width,
            ),
            dtype=sequence.dtype,
        )

        for frame_idx in range(num_frames):
            resized[frame_idx] = cv2.resize(
                sequence[frame_idx],
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        return resized

    def _resize_image(self, image):
        height, width = image.shape

        new_height = int(
            height * self.scale
        )
        new_width = int(
            width * self.scale
        )

        self._validate_size(
            new_height,
            new_width,
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def _validate_size(
        self,
        height,
        width,
    ):
        if height <= 0 or width <= 0:
            raise ValueError(
                f"Invalid scale={self.scale}: "
                f"resulting size is {height}x{width}."
            )

    def _apply_paired_transform(
        self,
        ap,
        lat,
    ):
        if self.transform is None:
            return ap, lat

        transformed = self.transform(
            image=ap,
            image2=lat,
        )

        return (
            transformed["image"],
            transformed["image2"],
        )

    @staticmethod
    def _sequence_to_tensor(sequence):
        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        sequence = torch.from_numpy(
            sequence
        )

        sequence = sequence.unsqueeze(1)
        sequence = sequence.repeat(
            1,
            3,
            1,
            1,
        )

        return sequence / 255.0

    @staticmethod
    def _minip_to_tensor(image):
        image = np.asarray(
            image,
            dtype=np.float32,
        )

        if image.ndim == 2:
            image = np.repeat(
                image[:, :, None],
                3,
                axis=2,
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError(
                f"Expected image shape [H, W, 3], "
                f"got {image.shape}."
            )

        return (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
        )

    def _normalize_sequence(self, sequence):
        mean = torch.tensor(
            self.IMAGENET_MEAN,
            dtype=sequence.dtype,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            self.IMAGENET_STD,
            dtype=sequence.dtype,
        ).view(1, 3, 1, 1)

        return (
            sequence - mean
        ) / std

    def _normalize_image(self, image):
        mean = torch.tensor(
            self.IMAGENET_MEAN,
            dtype=image.dtype,
        ).view(3, 1, 1)

        std = torch.tensor(
            self.IMAGENET_STD,
            dtype=image.dtype,
        ).view(3, 1, 1)

        return (
            image - mean
        ) / std