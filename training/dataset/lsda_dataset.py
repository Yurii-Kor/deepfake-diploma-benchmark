import sys
sys.path.append('.')

import os
import json
import yaml
import random

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T

import skimage.draw
import albumentations as alb
from albumentations import GaussianBlur
from torch.utils.data.sampler import Sampler

from .abstract_dataset import DeepfakeAbstractBaseDataset


private_path_prefix = '/home/zhaokangran/cvpr24/training'

fake_dict = {
    'real': 0,
    'Deepfakes': 1,
    'Face2Face': 2,
    'FaceSwap': 3,
    'NeuralTextures': 4,
}


def _safe_is_3060():
    try:
        if not torch.cuda.is_available():
            return False
        return "3060" in torch.cuda.get_device_name(0)
    except Exception:
        return False


def _norm_path(path: str) -> str:
    return os.path.normpath(str(path).replace('\\', '/'))


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(img, (int(W / r), int(H / r)), interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)
        return img_ds


augmentation_methods = alb.Compose([
    alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
    GaussianBlur(blur_limit=[3, 7], p=0.5)
], p=1.0)

augmentation_methods2 = alb.Compose([
    alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    alb.HorizontalFlip(p=0.5),
    RandomDownScale(p=0.5),
    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
], additional_targets={f'image1': 'image', f'image2': 'image', f'image3': 'image', f'image4': 'image'}, p=1.0)

normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transforms1 = T.Compose([
    T.ToTensor(),
    normalize
])


def load_rgb(file_path, size=256):
    file_path = _norm_path(file_path)
    assert os.path.exists(file_path), f"{file_path} does not exist"
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f'Img is None: {file_path}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(np.array(img, dtype=np.uint8))


def load_mask(file_path, size=256):
    file_path = _norm_path(file_path)
    mask = cv2.imread(file_path, 0)
    if mask is None:
        mask = np.zeros((size, size))

    mask = cv2.resize(mask, (size, size)) / 255
    mask = np.expand_dims(mask, axis=2)
    return np.float32(mask)


def add_gaussian_noise(ins, mean=0, stddev=0.1):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return torch.clamp(ins + noise, -1, 1)


class CustomSampler(Sampler):
    def __init__(self, num_groups=2 * 360, n_frame_per_vid=32, videos_per_group=5, batch_size=10):
        self.num_groups = num_groups
        self.n_frame_per_vid = n_frame_per_vid
        self.videos_per_group = videos_per_group
        self.batch_size = batch_size
        assert self.batch_size % self.videos_per_group == 0, "Batch size should be a multiple of videos_per_group."
        self.groups_per_batch = self.batch_size // self.videos_per_group

    def __iter__(self):
        group_indices = list(range(self.num_groups))
        random.shuffle(group_indices)

        for i in range(0, len(group_indices), self.groups_per_batch):
            selected_groups = group_indices[i:i + self.groups_per_batch]

            for group in selected_groups:
                frame_idx = random.randint(0, self.n_frame_per_vid - 1)
                for video_offset in range(self.videos_per_group):
                    yield group * self.videos_per_group * self.n_frame_per_vid + video_offset * self.n_frame_per_vid + frame_idx

    def __len__(self):
        return self.num_groups * self.videos_per_group


class LSDADataset(DeepfakeAbstractBaseDataset):
    # Safe CUDA check: does not crash on CPU-only / hidden-GPU runs.
    on_3060 = _safe_is_3060()

    transfer_dict = {
        'youtube': 'FF-real',
        'Deepfakes': 'FF-DF',
        'Face2Face': 'FF-F2F',
        'FaceSwap': 'FF-FS',
        'NeuralTextures': 'FF-NT'
    }

    # Safe defaults for Linux/WSL import.
    # These matter only if LSDA is actually instantiated.
    data_root = './datasets/rgb/FaceForensics++'
    data_list = {
        'test': './datasets/rgb/FaceForensics++/test.json',
        'train': './datasets/rgb/FaceForensics++/train.json',
        'eval': './datasets/rgb/FaceForensics++/val.json'
    }

    def __init__(self, config=None, mode='train', with_dataset=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']):
        super().__init__(config, mode)
        self.mode = mode
        self.res = config['resolution']
        self.fake_dict = fake_dict

        self.normalize = T.Normalize(mean=config['mean'], std=config['std'])
        self.transforms1 = T.Compose([
            T.ToTensor(),
            self.normalize
        ])

        self.img_lines = []
        self.config = config

        with open(self.config['dataset_json_folder'] + '/FaceForensics++.json', 'r') as fd:
            self.img_json = json.load(fd)

        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            img_lines = []

            for pair in data:
                r1, r2 = pair
                step = 1

                for i in range(0, config['frame_num'][mode], step):
                    img_lines.append((f'youtube/{r1}', i, 0, mode))

                for fake_d in with_dataset:
                    for i in range(0, config['frame_num'][mode], step):
                        img_lines.append((f'{fake_d}/{r1}_{r2}', i, self.fake_dict[fake_d], mode))

                for i in range(0, config['frame_num'][mode], step):
                    img_lines.append((f'youtube/{r2}', i, 0, mode))

                for fake_d in with_dataset:
                    for i in range(0, config['frame_num'][mode], step):
                        img_lines.append((f'{fake_d}/{r2}_{r1}', i, self.fake_dict[fake_d], mode))

        assert len(img_lines) == 2 * len(data) * (1 + len(with_dataset)) * config['frame_num'][mode], \
            "to match our custom sampler, the length should be 2*len(data)*(1+len(with_dataset))*frames[mode]"

        self.img_lines.extend(img_lines)

    def get_ids_from_path(self, path):
        parts = path.split('/')
        try:
            if 'youtube' in path:
                return [int(parts[-1])]
            else:
                return list(map(int, parts[-1].split('_')))
        except Exception:
            raise ValueError(f"wrong path: {path}")

    def load_image(self, name, idx):
        instance_type, video_name = name.split('/')

        all_frames = self.img_json[self.data_root.split(os.path.sep)[-1]][self.transfer_dict[instance_type]]['train']['c23'][video_name]['frames']
        img_path = all_frames[idx]
        img_path = _norm_path(img_path)

        img = load_rgb(img_path, size=self.res)
        return img

    def __getitem__(self, index):
        name, idx, label, mode = self.img_lines[index]
        label = int(label)

        try:
            img = self.load_image(name, idx)
        except Exception as e:
            if idx == 0:
                new_index = index + 1
            elif idx == 31:
                new_index = index - 1
            else:
                new_index = index + random.choice([-1, 1])

            if new_index < 0 or new_index >= len(self.img_lines):
                raise RuntimeError(f'Failed to load image {name} at index {idx}: {e}')

            print(f'Error loading image {name} at index {idx}. Try another one at index {new_index}')
            return self.__getitem__(new_index)

        if self.mode == 'train':
            img = np.asarray(img)
            img = augmentation_methods2(image=img)['image']
            img = Image.fromarray(np.array(img, dtype=np.uint8))
            img = self.transforms1(img)
        else:
            raise ValueError("Not implemented yet")

        return (img, label)

    def __len__(self):
        return len(self.img_lines)

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        bs, c, h, w = images.shape
        videos_per_group = 5
        num_groups = bs // videos_per_group

        images_grouped = images.view(num_groups, videos_per_group, c, h, w)
        labels_grouped = labels.view(num_groups, videos_per_group)

        valid_indices = []
        for i, group in enumerate(labels_grouped):
            if set(group.numpy().tolist()) == {0, 1, 2, 3, 4}:
                valid_indices.append(i)

        images_grouped = images_grouped[valid_indices]
        labels_grouped = labels_grouped[valid_indices]

        if not valid_indices:
            raise ValueError("No valid groups found in this batch.")

        return {
            'image': images_grouped,
            'label': labels_grouped,
            'mask': None,
            'landmark': None
        }


if __name__ == '__main__':
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/lsda.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_set = LSDADataset(config=config, mode='train')
    custom_sampler = CustomSampler(
        num_groups=2 * 360,
        n_frame_per_vid=config['frame_num']['train'],
        batch_size=config['train_batchSize'],
        videos_per_group=5
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config['train_batchSize'],
        num_workers=0,
        sampler=custom_sampler,
        collate_fn=train_set.collate_fn,
    )

    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break