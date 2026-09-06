# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys
import os
import math
import yaml
import glob
import json
import random
from copy import deepcopy

import lmdb
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

sys.path.append('.')

from .albu import IsotropicResize

FFpp_pool = [
    'FaceForensics++',
    'FaceShifter',
    'DeepFakeDetection',
    'FF-DF',
    'FF-F2F',
    'FF-FS',
    'FF-NT',
]


def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


def _basename_no_ext(path: str) -> str:
    path = str(path).replace('\\', '/')
    return os.path.basename(path).split('.')[0]


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, config=None, mode='train'):
        self.config = config
        self.mode = mode
        self.dataset_split = config.get('dataset_split', mode)

        if self.dataset_split == 'validation':
            self.dataset_split = 'val'

        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)

        self.image_list = []
        self.label_list = []

        # Study metadata kept parallel to image_list / label_list.
        #
        # source_label_list preserves the original DeepfakeBench label
        # before detector-specific numeric label mapping, e.g.:
        #
        #   FF-real
        #   FF-DF
        #   FF-F2F
        #   FF-FS
        #   FF-NT
        #
        # The metadata is intentionally not included in data_dict, because
        # detector batches must retain the original DeepfakeBench tensor
        # interface. It is available to study-controlled samplers through
        # the dataset object itself.
        self.source_label_list = []
        self.video_name_list = []

        if mode == 'train':
            dataset_list = config['train_dataset']

            image_list = []
            label_list = []
            name_list = []
            source_label_list = []

            for one_data in dataset_list:
                (
                    tmp_image,
                    tmp_label,
                    tmp_name,
                    tmp_source_label,
                ) = self.collect_img_and_label_for_one_dataset(
                    one_data
                )

                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
                name_list.extend(tmp_name)
                source_label_list.extend(tmp_source_label)

            if self.lmdb:
                if len(dataset_list) > 1:
                    if all_in_pool(dataset_list, FFpp_pool):
                        lmdb_path = os.path.join(
                            config['lmdb_dir'],
                            "FaceForensics++_lmdb",
                        )

                        self.env = lmdb.open(
                            lmdb_path,
                            create=False,
                            subdir=True,
                            readonly=True,
                            lock=False,
                        )
                    else:
                        raise ValueError(
                            'Training with multiple dataset and lmdb '
                            'is not implemented yet.'
                        )
                else:
                    lmdb_name = (
                        dataset_list[0]
                        if dataset_list[0] not in FFpp_pool
                        else 'FaceForensics++'
                    )

                    lmdb_path = os.path.join(
                        config['lmdb_dir'],
                        f"{lmdb_name}_lmdb",
                    )

                    self.env = lmdb.open(
                        lmdb_path,
                        create=False,
                        subdir=True,
                        readonly=True,
                        lock=False,
                    )

        elif mode == 'test':
            one_data = config['test_dataset']

            (
                image_list,
                label_list,
                name_list,
                source_label_list,
            ) = self.collect_img_and_label_for_one_dataset(
                one_data
            )

            if self.lmdb:
                lmdb_name = (
                    one_data
                    if one_data not in FFpp_pool
                    else 'FaceForensics++'
                )

                lmdb_path = os.path.join(
                    config['lmdb_dir'],
                    f"{lmdb_name}_lmdb",
                )

                self.env = lmdb.open(
                    lmdb_path,
                    create=False,
                    subdir=True,
                    readonly=True,
                    lock=False,
                )

        else:
            raise NotImplementedError(
                'Only train and test modes are supported.'
            )

        if not image_list or not label_list:
            raise ValueError(
                f"Collect nothing for {mode} mode!"
            )

        metadata_lengths = {
            'image_list': len(image_list),
            'label_list': len(label_list),
            'video_name_list': len(name_list),
            'source_label_list': len(source_label_list),
        }

        if len(set(metadata_lengths.values())) != 1:
            raise RuntimeError(
                "Dataset sample metadata is not index-aligned: "
                f"{metadata_lengths}"
            )

        self.image_list = image_list
        self.label_list = label_list
        self.video_name_list = name_list
        self.source_label_list = source_label_list

        # Keep detector-facing data unchanged. Study-only string metadata
        # remains available through dataset attributes.
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }

        self.transform = self.init_data_aug_method()

    def _resolve_rgb_path(self, file_path: str) -> str:
        """
        Build a safe Linux/Windows-neutral RGB path from rgb_dir + relative
        frame path.
        """
        rgb_dir = str(
            self.config['rgb_dir']
        ).replace(
            '\\',
            '/',
        ).rstrip('/')

        file_path = str(
            file_path
        ).replace(
            '\\',
            '/',
        )

        if file_path.startswith('./'):
            file_path = file_path[2:]

        if os.path.isabs(file_path):
            return os.path.normpath(file_path)

        return os.path.normpath(
            os.path.join(
                rgb_dir,
                file_path.lstrip('/'),
            )
        )

    def _rgb_path_to_lmdb_key(self, file_path: str) -> str:
        """
        Convert an RGB-style path into the LMDB key format.
        """
        file_path = str(
            file_path
        ).replace(
            '\\',
            '/',
        )

        if file_path.startswith('./'):
            file_path = file_path[2:]

        if file_path.startswith('datasets/'):
            file_path = file_path[
                len('datasets/') :
            ]

        return file_path

    def init_data_aug_method(self):
        trans = A.Compose(
            [
                A.HorizontalFlip(
                    p=self.config[
                        'data_aug'
                    ][
                        'flip_prob'
                    ]
                ),
                A.Rotate(
                    limit=self.config[
                        'data_aug'
                    ][
                        'rotate_limit'
                    ],
                    p=self.config[
                        'data_aug'
                    ][
                        'rotate_prob'
                    ],
                ),
                A.GaussianBlur(
                    blur_limit=self.config[
                        'data_aug'
                    ][
                        'blur_limit'
                    ],
                    p=self.config[
                        'data_aug'
                    ][
                        'blur_prob'
                    ],
                ),
                A.OneOf(
                    [
                        IsotropicResize(
                            max_side=self.config[
                                'resolution'
                            ],
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC,
                        ),
                        IsotropicResize(
                            max_side=self.config[
                                'resolution'
                            ],
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                        IsotropicResize(
                            max_side=self.config[
                                'resolution'
                            ],
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=(
                        0
                        if self.config[
                            'with_landmark'
                        ]
                        else 1
                    ),
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=self.config[
                                'data_aug'
                            ][
                                'brightness_limit'
                            ],
                            contrast_limit=self.config[
                                'data_aug'
                            ][
                                'contrast_limit'
                            ],
                        ),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=0.5,
                ),
                A.ImageCompression(
                    quality_lower=self.config[
                        'data_aug'
                    ][
                        'quality_lower'
                    ],
                    quality_upper=self.config[
                        'data_aug'
                    ][
                        'quality_upper'
                    ],
                    p=0.5,
                ),
            ],
            keypoint_params=(
                A.KeypointParams(
                    format='xy'
                )
                if self.config[
                    'with_landmark'
                ]
                else None
            ),
        )

        return trans

    def rescale_landmarks(
        self,
        landmarks,
        original_size=256,
        new_size=224,
    ):
        scale_factor = (
            new_size
            / original_size
        )

        return (
            landmarks
            * scale_factor
        )

    def collect_img_and_label_for_one_dataset(
        self,
        dataset_name: str,
    ):
        label_list = []
        frame_path_list = []
        video_name_list = []
        source_label_list = []

        if not os.path.exists(
            self.config[
                'dataset_json_folder'
            ]
        ):
            self.config[
                'dataset_json_folder'
            ] = self.config[
                'dataset_json_folder'
            ].replace(
                '/Youtu_Pangu_Security_Public',
                '/Youtu_Pangu_Security/public',
            )

        try:
            with open(
                os.path.join(
                    self.config[
                        'dataset_json_folder'
                    ],
                    dataset_name + '.json',
                ),
                'r',
            ) as file:
                dataset_info = json.load(
                    file
                )

        except Exception as exc:
            print(exc)

            raise ValueError(
                f'dataset {dataset_name} not exist!'
            )

        cp = None

        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'

        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'

        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'

        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'

        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'

        for label_key in dataset_info[
            dataset_name
        ]:
            sub_dataset_info = (
                dataset_info[
                    dataset_name
                ][
                    label_key
                ][
                    self.dataset_split
                ]
            )

            if (
                cp is None
                and dataset_name
                in [
                    'FF-DF',
                    'FF-F2F',
                    'FF-FS',
                    'FF-NT',
                    'FaceForensics++',
                    'DeepFakeDetection',
                    'FaceShifter',
                ]
            ):
                sub_dataset_info = (
                    sub_dataset_info[
                        self.compression
                    ]
                )

            elif (
                cp == 'c40'
                and dataset_name
                in [
                    'FF-DF',
                    'FF-F2F',
                    'FF-FS',
                    'FF-NT',
                    'FaceForensics++',
                    'DeepFakeDetection',
                    'FaceShifter',
                ]
            ):
                sub_dataset_info = (
                    sub_dataset_info[
                        'c40'
                    ]
                )

            for (
                video_name,
                video_info,
            ) in sub_dataset_info.items():
                source_label = (
                    video_info[
                        'label'
                    ]
                )

                unique_video_name = (
                    source_label
                    + '_'
                    + video_name
                )

                if (
                    source_label
                    not in self.config[
                        'label_dict'
                    ]
                ):
                    raise ValueError(
                        'Label {} is not found in the '
                        'configuration file.'.format(
                            source_label
                        )
                    )

                numeric_label = (
                    self.config[
                        'label_dict'
                    ][
                        source_label
                    ]
                )

                frame_paths = (
                    video_info[
                        'frames'
                    ]
                )

                frame_paths = sorted(
                    frame_paths,
                    key=lambda path: int(
                        _basename_no_ext(
                            path
                        )
                    ),
                )

                total_frames = len(
                    frame_paths
                )

                if (
                    self.frame_num
                    < total_frames
                ):
                    if self.video_level:
                        start_frame = (
                            random.randint(
                                0,
                                total_frames
                                - self.frame_num,
                            )
                            if self.mode
                            == 'train'
                            else 0
                        )

                        frame_paths = (
                            frame_paths[
                                start_frame:
                                start_frame
                                + self.frame_num
                            ]
                        )

                    else:
                        step = (
                            total_frames
                            // self.frame_num
                        )

                        frame_paths = [
                            frame_paths[index]
                            for index in range(
                                0,
                                total_frames,
                                step,
                            )
                        ][
                            :self.frame_num
                        ]

                    total_frames = len(
                        frame_paths
                    )

                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError(
                            'clip_size must be specified '
                            'when video_level is True.'
                        )

                    if (
                        total_frames
                        >= self.clip_size
                    ):
                        selected_clips = []

                        num_clips = (
                            total_frames
                            // self.clip_size
                        )

                        if num_clips > 1:
                            clip_step = (
                                (
                                    total_frames
                                    - self.clip_size
                                )
                                // (
                                    num_clips
                                    - 1
                                )
                            )

                            for clip_index in range(
                                num_clips
                            ):
                                if (
                                    self.mode
                                    == 'train'
                                ):
                                    start_frame = (
                                        random.randrange(
                                            clip_index
                                            * clip_step,
                                            min(
                                                (
                                                    clip_index
                                                    + 1
                                                )
                                                * clip_step,
                                                total_frames
                                                - self.clip_size
                                                + 1,
                                            ),
                                        )
                                    )
                                else:
                                    start_frame = (
                                        clip_index
                                        * clip_step
                                    )

                                continuous_frames = (
                                    frame_paths[
                                        start_frame:
                                        start_frame
                                        + self.clip_size
                                    ]
                                )

                                assert (
                                    len(
                                        continuous_frames
                                    )
                                    == self.clip_size
                                ), (
                                    'clip_size is not equal '
                                    'to the length of '
                                    'frame_path_list'
                                )

                                selected_clips.append(
                                    continuous_frames
                                )

                        else:
                            if (
                                self.mode
                                == 'train'
                            ):
                                start_frame = (
                                    random.randrange(
                                        0,
                                        total_frames
                                        - self.clip_size
                                        + 1,
                                    )
                                )
                            else:
                                start_frame = 0

                            continuous_frames = (
                                frame_paths[
                                    start_frame:
                                    start_frame
                                    + self.clip_size
                                ]
                            )

                            assert (
                                len(
                                    continuous_frames
                                )
                                == self.clip_size
                            ), (
                                'clip_size is not equal '
                                'to the length of '
                                'frame_path_list'
                            )

                            selected_clips.append(
                                continuous_frames
                            )

                        sample_count = len(
                            selected_clips
                        )

                        label_list.extend(
                            [
                                numeric_label
                            ]
                            * sample_count
                        )

                        frame_path_list.extend(
                            selected_clips
                        )

                        video_name_list.extend(
                            [
                                unique_video_name
                            ]
                            * sample_count
                        )

                        source_label_list.extend(
                            [
                                source_label
                            ]
                            * sample_count
                        )

                    else:
                        print(
                            "Skipping video {} because it has "
                            "less than clip_size ({}) frames "
                            "({}).".format(
                                unique_video_name,
                                self.clip_size,
                                total_frames,
                            )
                        )

                else:
                    label_list.extend(
                        [
                            numeric_label
                        ]
                        * total_frames
                    )

                    frame_path_list.extend(
                        frame_paths
                    )

                    video_name_list.extend(
                        [
                            unique_video_name
                        ]
                        * total_frames
                    )

                    source_label_list.extend(
                        [
                            source_label
                        ]
                        * total_frames
                    )

        shuffled = list(
            zip(
                label_list,
                frame_path_list,
                video_name_list,
                source_label_list,
            )
        )

        random.shuffle(
            shuffled
        )

        if not shuffled:
            raise ValueError(
                "No samples collected for dataset={}, "
                "split={}, compression={}".format(
                    dataset_name,
                    self.dataset_split,
                    self.compression,
                )
            )

        (
            label_list,
            frame_path_list,
            video_name_list,
            source_label_list,
        ) = zip(
            *shuffled
        )

        return (
            frame_path_list,
            label_list,
            video_name_list,
            source_label_list,
        )

    def load_rgb(self, file_path):
        size = self.config[
            'resolution'
        ]

        if not self.lmdb:
            file_path = (
                self._resolve_rgb_path(
                    file_path
                )
            )

            assert os.path.exists(
                file_path
            ), (
                f"{file_path} does not exist"
            )

            img = cv2.imread(
                file_path
            )

            if img is None:
                raise ValueError(
                    f'Loaded image is None: '
                    f'{file_path}'
                )

        else:
            with self.env.begin(
                write=False
            ) as txn:
                lmdb_key = (
                    self._rgb_path_to_lmdb_key(
                        file_path
                    )
                )

                image_bin = txn.get(
                    lmdb_key.encode()
                )

                if image_bin is None:
                    raise ValueError(
                        f'LMDB key not found: '
                        f'{lmdb_key}'
                    )

                image_buf = np.frombuffer(
                    image_bin,
                    dtype=np.uint8,
                )

                img = cv2.imdecode(
                    image_buf,
                    cv2.IMREAD_COLOR,
                )

                if img is None:
                    raise ValueError(
                        f'Failed to decode LMDB '
                        f'image: {lmdb_key}'
                    )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        img = cv2.resize(
            img,
            (
                size,
                size,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        return Image.fromarray(
            np.array(
                img,
                dtype=np.uint8,
            )
        )

    def load_mask(self, file_path):
        size = self.config[
            'resolution'
        ]

        if file_path is None:
            return np.zeros(
                (
                    size,
                    size,
                    1,
                )
            )

        if not self.lmdb:
            file_path = (
                self._resolve_rgb_path(
                    file_path
                )
            )

            if os.path.exists(
                file_path
            ):
                mask = cv2.imread(
                    file_path,
                    0,
                )

                if mask is None:
                    mask = np.zeros(
                        (
                            size,
                            size,
                        )
                    )

            else:
                return np.zeros(
                    (
                        size,
                        size,
                        1,
                    )
                )

        else:
            with self.env.begin(
                write=False
            ) as txn:
                lmdb_key = (
                    self._rgb_path_to_lmdb_key(
                        file_path
                    )
                )

                image_bin = txn.get(
                    lmdb_key.encode()
                )

                if image_bin is None:
                    mask = np.zeros(
                        (
                            size,
                            size,
                            3,
                        )
                    )

                else:
                    image_buf = (
                        np.frombuffer(
                            image_bin,
                            dtype=np.uint8,
                        )
                    )

                    mask = cv2.imdecode(
                        image_buf,
                        cv2.IMREAD_COLOR,
                    )

                    if mask is None:
                        mask = np.zeros(
                            (
                                size,
                                size,
                                3,
                            )
                        )

        mask = (
            cv2.resize(
                mask,
                (
                    size,
                    size,
                ),
            )
            / 255
        )

        mask = np.expand_dims(
            mask,
            axis=2,
        )

        return np.float32(
            mask
        )

    def load_landmark(self, file_path):
        if file_path is None:
            return np.zeros(
                (
                    81,
                    2,
                )
            )

        if not self.lmdb:
            file_path = (
                self._resolve_rgb_path(
                    file_path
                )
            )

            if os.path.exists(
                file_path
            ):
                landmark = np.load(
                    file_path
                )
            else:
                return np.zeros(
                    (
                        81,
                        2,
                    )
                )

        else:
            with self.env.begin(
                write=False
            ) as txn:
                lmdb_key = (
                    self._rgb_path_to_lmdb_key(
                        file_path
                    )
                )

                binary = txn.get(
                    lmdb_key.encode()
                )

                if binary is None:
                    return np.zeros(
                        (
                            81,
                            2,
                        )
                    )

                landmark = np.frombuffer(
                    binary,
                    dtype=np.uint32,
                ).reshape(
                    (
                        81,
                        2,
                    )
                )

                landmark = (
                    self.rescale_landmarks(
                        np.float32(
                            landmark
                        ),
                        original_size=256,
                        new_size=self.config[
                            'resolution'
                        ],
                    )
                )

        return landmark

    def to_tensor(self, img):
        return T.ToTensor()(
            img
        )

    def normalize(self, img):
        mean = self.config[
            'mean'
        ]

        std = self.config[
            'std'
        ]

        normalize = T.Normalize(
            mean=mean,
            std=std,
        )

        return normalize(
            img
        )

    def data_aug(
        self,
        img,
        landmark=None,
        mask=None,
        augmentation_seed=None,
    ):
        if augmentation_seed is not None:
            random.seed(
                augmentation_seed
            )

            np.random.seed(
                augmentation_seed
            )

        kwargs = {
            'image': img
        }

        if landmark is not None:
            kwargs[
                'keypoints'
            ] = landmark

            kwargs[
                'keypoint_params'
            ] = A.KeypointParams(
                format='xy'
            )

        if mask is not None:
            mask = mask.squeeze(
                2
            )

            if mask.max() > 0:
                kwargs[
                    'mask'
                ] = mask

        transformed = (
            self.transform(
                **kwargs
            )
        )

        augmented_img = (
            transformed[
                'image'
            ]
        )

        augmented_landmark = (
            transformed.get(
                'keypoints'
            )
        )

        augmented_mask = (
            transformed.get(
                'mask',
                mask,
            )
        )

        if (
            augmented_landmark
            is not None
        ):
            augmented_landmark = (
                np.array(
                    augmented_landmark
                )
            )

        if (
            augmentation_seed
            is not None
        ):
            random.seed()
            np.random.seed()

        return (
            augmented_img,
            augmented_landmark,
            augmented_mask,
        )

    def __getitem__(
        self,
        index,
        no_norm=False,
    ):
        image_paths = (
            self.data_dict[
                'image'
            ][
                index
            ]
        )

        label = (
            self.data_dict[
                'label'
            ][
                index
            ]
        )

        if not isinstance(
            image_paths,
            list,
        ):
            image_paths = [
                image_paths
            ]

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            if (
                self.video_level
                and image_path
                == image_paths[0]
            ):
                augmentation_seed = (
                    random.randint(
                        0,
                        2**32 - 1,
                    )
                )

            mask_path = (
                image_path.replace(
                    'frames',
                    'masks',
                )
            )

            landmark_path = (
                image_path.replace(
                    'frames',
                    'landmarks',
                ).replace(
                    '.png',
                    '.npy',
                )
            )

            try:
                image = self.load_rgb(
                    image_path
                )

            except Exception as exc:
                raise RuntimeError(
                    "Error loading image at "
                    "index {}: {} | {}".format(
                        index,
                        image_path,
                        exc,
                    )
                )

            image = np.array(
                image
            )

            if self.config[
                'with_mask'
            ]:
                mask = self.load_mask(
                    mask_path
                )
            else:
                mask = None

            if self.config[
                'with_landmark'
            ]:
                landmarks = (
                    self.load_landmark(
                        landmark_path
                    )
                )
            else:
                landmarks = None

            if (
                self.mode == 'train'
                and self.config[
                    'use_data_augmentation'
                ]
            ):
                (
                    image_trans,
                    landmarks_trans,
                    mask_trans,
                ) = self.data_aug(
                    image,
                    landmarks,
                    mask,
                    augmentation_seed,
                )

            else:
                image_trans = deepcopy(
                    image
                )

                landmarks_trans = deepcopy(
                    landmarks
                )

                mask_trans = deepcopy(
                    mask
                )

            if not no_norm:
                image_trans = (
                    self.normalize(
                        self.to_tensor(
                            image_trans
                        )
                    )
                )

                if (
                    self.config[
                        'with_landmark'
                    ]
                    and landmarks
                    is not None
                ):
                    landmarks_trans = (
                        torch.from_numpy(
                            landmarks
                        )
                    )

                if (
                    self.config[
                        'with_mask'
                    ]
                    and mask_trans
                    is not None
                ):
                    mask_trans = (
                        torch.from_numpy(
                            mask_trans
                        )
                    )

            image_tensors.append(
                image_trans
            )

            landmark_tensors.append(
                landmarks_trans
            )

            mask_tensors.append(
                mask_trans
            )

        if self.video_level:
            image_tensors = (
                torch.stack(
                    image_tensors,
                    dim=0,
                )
            )

            if not any(
                landmark is None
                or (
                    isinstance(
                        landmark,
                        list,
                    )
                    and None
                    in landmark
                )
                for landmark
                in landmark_tensors
            ):
                landmark_tensors = (
                    torch.stack(
                        landmark_tensors,
                        dim=0,
                    )
                )

            if not any(
                item is None
                or (
                    isinstance(
                        item,
                        list,
                    )
                    and None
                    in item
                )
                for item
                in mask_tensors
            ):
                mask_tensors = (
                    torch.stack(
                        mask_tensors,
                        dim=0,
                    )
                )

        else:
            image_tensors = (
                image_tensors[0]
            )

            if not any(
                landmark is None
                or (
                    isinstance(
                        landmark,
                        list,
                    )
                    and None
                    in landmark
                )
                for landmark
                in landmark_tensors
            ):
                landmark_tensors = (
                    landmark_tensors[
                        0
                    ]
                )

            if not any(
                item is None
                or (
                    isinstance(
                        item,
                        list,
                    )
                    and None
                    in item
                )
                for item
                in mask_tensors
            ):
                mask_tensors = (
                    mask_tensors[0]
                )

        return (
            image_tensors,
            label,
            landmark_tensors,
            mask_tensors,
        )

    @staticmethod
    def collate_fn(batch):
        (
            images,
            labels,
            landmarks,
            masks,
        ) = zip(
            *batch
        )

        images = torch.stack(
            images,
            dim=0,
        )

        labels = torch.LongTensor(
            labels
        )

        if not any(
            landmark is None
            or (
                isinstance(
                    landmark,
                    list,
                )
                and None
                in landmark
            )
            for landmark
            in landmarks
        ):
            landmarks = torch.stack(
                landmarks,
                dim=0,
            )

        else:
            landmarks = None

        if not any(
            item is None
            or (
                isinstance(
                    item,
                    list,
                )
                and None
                in item
            )
            for item
            in masks
        ):
            masks = torch.stack(
                masks,
                dim=0,
            )

        else:
            masks = None

        return {
            'image': images,
            'label': labels,
            'landmark': landmarks,
            'mask': masks,
        }

    def __len__(self):
        if not (
            len(self.image_list)
            == len(self.label_list)
            == len(self.video_name_list)
            == len(self.source_label_list)
        ):
            raise RuntimeError(
                "Dataset sample metadata lost index alignment"
            )

        return len(
            self.image_list
        )


if __name__ == "__main__":
    with open(
        '/data/home/zhiyuany/DeepfakeBench/'
        'training/config/detector/video_baseline.yaml',
        'r',
    ) as file:
        config = yaml.safe_load(
            file
        )

    train_set = (
        DeepfakeAbstractBaseDataset(
            config=config,
            mode='train',
        )
    )

    train_data_loader = (
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config[
                'train_batchSize'
            ],
            shuffle=True,
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    )

    from tqdm import tqdm

    for iteration, batch in enumerate(
        tqdm(
            train_data_loader
        )
    ):
        ...