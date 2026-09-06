'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods
where it is necessary to ensure an equal number of positive and negative
samples.
'''

import random
from copy import deepcopy

import numpy as np
import torch

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset


class pairDataset(DeepfakeAbstractBaseDataset):
    """
    Pair dataset used by UCF training.

    Each dataset item contains:
        * one fake input selected by the DataLoader sampler;
        * one real input sampled with replacement from the valid real pool.

    Study-controlled behavior
    -------------------------
    The outer study sampler controls which fake item is exposed and how often.
    This class retains the original UCF real-pair selection semantics.

    Data augmentation is applied only when both:
        * mode == 'train'
        * config['use_data_augmentation'] is True

    This keeps the UCF pair construction intact while respecting the common
    study augmentation policy.
    """

    def __init__(
        self,
        config=None,
        mode='train',
    ):
        super().__init__(
            config,
            mode,
        )

        # Preserve UCF-specific labels:
        #
        # real:
        #   specific label = 0
        #
        # fake:
        #   specific labels distinguish manipulation methods.
        #
        # The final tuple item is the common binary label used by UCF's
        # common-forgery classification branch.
        self.fake_imglist = [
            (
                image,
                label,
                1,
            )
            for image, label
            in zip(
                self.image_list,
                self.label_list,
            )
            if label != 0
        ]

        self.real_imglist = [
            (
                image,
                label,
                0,
            )
            for image, label
            in zip(
                self.image_list,
                self.label_list,
            )
            if label == 0
        ]

        if not self.fake_imglist:
            raise ValueError(
                "pairDataset requires at least one fake input"
            )

        if not self.real_imglist:
            raise ValueError(
                "pairDataset requires at least one real input"
            )

    def __getitem__(
        self,
        index,
        norm=True,
    ):
        # Fake item is controlled by the study DataLoader sampler.
        (
            fake_image_path,
            fake_spe_label,
            fake_label,
        ) = self.fake_imglist[
            index
        ]

        # Preserve original UCF pair semantics:
        # select a valid real item with replacement.
        real_index = random.randint(
            0,
            len(
                self.real_imglist
            ) - 1,
        )

        (
            real_image_path,
            real_spe_label,
            real_label,
        ) = self.real_imglist[
            real_index
        ]

        fake_mask_path = (
            fake_image_path.replace(
                'frames',
                'masks',
            )
        )

        fake_landmark_path = (
            fake_image_path.replace(
                'frames',
                'landmarks',
            ).replace(
                '.png',
                '.npy',
            )
        )

        real_mask_path = (
            real_image_path.replace(
                'frames',
                'masks',
            )
        )

        real_landmark_path = (
            real_image_path.replace(
                'frames',
                'landmarks',
            ).replace(
                '.png',
                '.npy',
            )
        )

        fake_image = np.array(
            self.load_rgb(
                fake_image_path
            )
        )

        real_image = np.array(
            self.load_rgb(
                real_image_path
            )
        )

        if self.config[
            'with_mask'
        ]:
            fake_mask = (
                self.load_mask(
                    fake_mask_path
                )
            )

            real_mask = (
                self.load_mask(
                    real_mask_path
                )
            )

        else:
            fake_mask = None
            real_mask = None

        if self.config[
            'with_landmark'
        ]:
            fake_landmarks = (
                self.load_landmark(
                    fake_landmark_path
                )
            )

            real_landmarks = (
                self.load_landmark(
                    real_landmark_path
                )
            )

        else:
            fake_landmarks = None
            real_landmarks = None

        # Respect the same study-level augmentation switch used by the
        # ordinary DeepfakeAbstractBaseDataset.
        if (
            self.mode == 'train'
            and self.config[
                'use_data_augmentation'
            ]
        ):
            (
                fake_image_trans,
                fake_landmarks_trans,
                fake_mask_trans,
            ) = self.data_aug(
                fake_image,
                fake_landmarks,
                fake_mask,
            )

            (
                real_image_trans,
                real_landmarks_trans,
                real_mask_trans,
            ) = self.data_aug(
                real_image,
                real_landmarks,
                real_mask,
            )

        else:
            fake_image_trans = deepcopy(
                fake_image
            )

            fake_landmarks_trans = deepcopy(
                fake_landmarks
            )

            fake_mask_trans = deepcopy(
                fake_mask
            )

            real_image_trans = deepcopy(
                real_image
            )

            real_landmarks_trans = deepcopy(
                real_landmarks
            )

            real_mask_trans = deepcopy(
                real_mask
            )

        if not norm:
            return {
                "fake": (
                    fake_image_trans,
                    fake_label,
                ),
                "real": (
                    real_image_trans,
                    real_label,
                ),
            }

        fake_image_trans = (
            self.normalize(
                self.to_tensor(
                    fake_image_trans
                )
            )
        )

        real_image_trans = (
            self.normalize(
                self.to_tensor(
                    real_image_trans
                )
            )
        )

        if self.config[
            'with_landmark'
        ]:
            if (
                fake_landmarks_trans
                is not None
            ):
                fake_landmarks_trans = (
                    torch.from_numpy(
                        fake_landmarks_trans
                    )
                )

            if (
                real_landmarks_trans
                is not None
            ):
                real_landmarks_trans = (
                    torch.from_numpy(
                        real_landmarks_trans
                    )
                )

        if self.config[
            'with_mask'
        ]:
            if (
                fake_mask_trans
                is not None
            ):
                fake_mask_trans = (
                    torch.from_numpy(
                        fake_mask_trans
                    )
                )

            if (
                real_mask_trans
                is not None
            ):
                real_mask_trans = (
                    torch.from_numpy(
                        real_mask_trans
                    )
                )

        return {
            "fake": (
                fake_image_trans,
                fake_label,
                fake_spe_label,
                fake_landmarks_trans,
                fake_mask_trans,
            ),
            "real": (
                real_image_trans,
                real_label,
                real_spe_label,
                real_landmarks_trans,
                real_mask_trans,
            ),
        }

    def __len__(
        self,
    ):
        return len(
            self.fake_imglist
        )

    @staticmethod
    def collate_fn(
        batch,
    ):
        """
        Combine UCF fake/real pair-items into one effective image batch.

        For loader batch size N:
            N real images
            N fake images

        Effective detector batch:
            2N images

        Real images are placed first, preserving the original UCF batching
        convention.
        """
        (
            fake_images,
            fake_labels,
            fake_spe_labels,
            fake_landmarks,
            fake_masks,
        ) = zip(
            *[
                data[
                    "fake"
                ]
                for data
                in batch
            ]
        )

        (
            real_images,
            real_labels,
            real_spe_labels,
            real_landmarks,
            real_masks,
        ) = zip(
            *[
                data[
                    "real"
                ]
                for data
                in batch
            ]
        )

        fake_images = torch.stack(
            fake_images,
            dim=0,
        )

        fake_labels = (
            torch.LongTensor(
                fake_labels
            )
        )

        fake_spe_labels = (
            torch.LongTensor(
                fake_spe_labels
            )
        )

        real_images = torch.stack(
            real_images,
            dim=0,
        )

        real_labels = (
            torch.LongTensor(
                real_labels
            )
        )

        real_spe_labels = (
            torch.LongTensor(
                real_spe_labels
            )
        )

        if (
            fake_landmarks[0]
            is not None
        ):
            fake_landmarks = (
                torch.stack(
                    fake_landmarks,
                    dim=0,
                )
            )

        else:
            fake_landmarks = None

        if (
            real_landmarks[0]
            is not None
        ):
            real_landmarks = (
                torch.stack(
                    real_landmarks,
                    dim=0,
                )
            )

        else:
            real_landmarks = None

        if (
            fake_masks[0]
            is not None
        ):
            fake_masks = (
                torch.stack(
                    fake_masks,
                    dim=0,
                )
            )

        else:
            fake_masks = None

        if (
            real_masks[0]
            is not None
        ):
            real_masks = (
                torch.stack(
                    real_masks,
                    dim=0,
                )
            )

        else:
            real_masks = None

        images = torch.cat(
            [
                real_images,
                fake_images,
            ],
            dim=0,
        )

        labels = torch.cat(
            [
                real_labels,
                fake_labels,
            ],
            dim=0,
        )

        spe_labels = torch.cat(
            [
                real_spe_labels,
                fake_spe_labels,
            ],
            dim=0,
        )

        if (
            fake_landmarks is not None
            and real_landmarks is not None
        ):
            landmarks = torch.cat(
                [
                    real_landmarks,
                    fake_landmarks,
                ],
                dim=0,
            )

        else:
            landmarks = None

        if (
            fake_masks is not None
            and real_masks is not None
        ):
            masks = torch.cat(
                [
                    real_masks,
                    fake_masks,
                ],
                dim=0,
            )

        else:
            masks = None

        return {
            'image': images,
            'label': labels,
            'label_spe': spe_labels,
            'landmark': landmarks,
            'mask': masks,
        }