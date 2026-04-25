# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-29
# description: Data pre-processing script for deepfake dataset.

"""
Patched rearrange.py for a flat smoke-ready FaceForensics++ layout.

For FaceForensics++, this patched version expects:
- FaceForensics++/original/frames/<video_id>/*.png
- FaceForensics++/Deepfakes/frames/<video_id>/*.png
- FaceForensics++/Face2Face/frames/<video_id>/*.png
- FaceForensics++/FaceSwap/frames/<video_id>/*.png
- FaceForensics++/NeuralTextures/frames/<video_id>/*.png

It does NOT require:
- original_sequences/youtube/<comp>/frames
- original_sequences/actors/<comp>/frames
- manipulated_sequences/<label>/<comp>/frames
- train.json / val.json / test.json
- masks

For the smoke run, all discovered videos are placed under test/raw.
"""

import os
import glob
import cv2
import json
import yaml
import pandas as pd
from pathlib import Path

BASE_RGB_ROOT = None


def sorted_frame_paths(video_dir: str):
    frame_paths = [os.path.join(video_dir, frame.name) for frame in os.scandir(video_dir)]
    frame_paths = sorted(frame_paths)

    if BASE_RGB_ROOT is not None:
        frame_paths = [os.path.relpath(fp, BASE_RGB_ROOT) for fp in frame_paths]

    return frame_paths


def generate_dataset_file(dataset_name, dataset_root_path, output_file_path, compression_level='raw', perturbation='end_to_end'):
    """
    Generate a JSON file containing dataset information.
    """

    dataset_dict = {}
    os.makedirs(output_file_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Patched flat-layout FaceForensics++
    # ------------------------------------------------------------------
    if dataset_name == 'FaceForensics++':
        dataset_path = os.path.join(dataset_root_path, 'FaceForensics++')

        folder_to_label = {
            'original': 'FF-real',
            'Deepfakes': 'FF-DF',
            'Face2Face': 'FF-F2F',
            'FaceSwap': 'FF-FS',
            'NeuralTextures': 'FF-NT',
        }

        dataset_dict['FaceForensics++'] = {}

        # initialize the labels we want
        for label in ['FF-real', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']:
            dataset_dict['FaceForensics++'][label] = {
                'train': {'raw': {}},
                'val': {'raw': {}},
                'test': {'raw': {}},
            }

        for folder_name, target_label in folder_to_label.items():
            frames_root = os.path.join(dataset_path, folder_name, 'frames')
            if not os.path.isdir(frames_root):
                print(f"Skipping missing frames directory: {frames_root}")
                continue

            for video_path in os.scandir(frames_root):
                if video_path.is_dir():
                    video_name = video_path.name
                    frame_paths = sorted_frame_paths(video_path.path)
                    if len(frame_paths) == 0:
                        continue

                    dataset_dict['FaceForensics++'][target_label]['test']['raw'][video_name] = {
                        'label': target_label,
                        'frames': frame_paths
                    }

        # generate per-label FF++ jsons like original script
        for label, value in dataset_dict['FaceForensics++'].items():
            if label != 'FF-real':
                with open(os.path.join(output_file_path, f'{label}.json'), 'w') as f:
                    data = {
                        label: {
                            'FF-real': dataset_dict['FaceForensics++']['FF-real'],
                            label: value,
                        }
                    }
                    json.dump(data, f)
                    print(f"Finish writing {label}.json")

    # ------------------------------------------------------------------
    # Celeb-DF-v1
    # ------------------------------------------------------------------
    elif dataset_name == 'Celeb-DF-v1':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['Celeb-real', 'YouTube-real']:
                label = 'CelebDFv1_real'
            else:
                label = 'CelebDFv1_fake'
            assert label in ['CelebDFv1_real', 'CelebDFv1_fake'], f'Invalid label: {label}'
            dataset_dict[dataset_name][label] = {}
            dataset_dict[dataset_name][label]['train'] = {}
            dataset_dict[dataset_name][label]['val'] = {}
            dataset_dict[dataset_name][label]['test'] = {}
            for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                if video_path.is_dir():
                    video_name = video_path.name
                    frame_paths = sorted_frame_paths(video_path.path)
                    dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}

        with open(os.path.join(dataset_root_path, dataset_name, 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'CelebDFv1_real'
            elif 'synthesis' in line:
                label = 'CelebDFv1_fake'
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")

            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join(dataset_root_path, dataset_name, line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            frame_paths = [os.path.relpath(fp, BASE_RGB_ROOT) for fp in sorted(frame_paths)]
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}

    # ------------------------------------------------------------------
    # Celeb-DF-v2
    # ------------------------------------------------------------------
    elif dataset_name == 'Celeb-DF-v2':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {}
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['Celeb-real', 'YouTube-real']:
                label = 'CelebDFv2_real'
            else:
                label = 'CelebDFv2_fake'
            assert label in ['CelebDFv2_real', 'CelebDFv2_fake'], f'Invalid label: {label}'
            dataset_dict[dataset_name][label] = {}
            dataset_dict[dataset_name][label]['train'] = {}
            dataset_dict[dataset_name][label]['val'] = {}
            dataset_dict[dataset_name][label]['test'] = {}
            for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                if video_path.is_dir():
                    video_name = video_path.name
                    frame_paths = sorted_frame_paths(video_path.path)
                    dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}

        with open(os.path.join(dataset_root_path, dataset_name, 'List_of_testing_videos.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'real' in line:
                label = 'CelebDFv2_real'
            elif 'synthesis' in line:
                label = 'CelebDFv2_fake'
            else:
                raise ValueError(f"wrong in processing vidname {dataset_name}: {line}")

            vidname = line.split('\n')[0].split('/')[-1].split('.mp4')[0]
            frame_paths = glob.glob(
                os.path.join(dataset_root_path, dataset_name, line.split(' ')[1].split('/')[0], 'frames', vidname, '*png'))
            frame_paths = [os.path.relpath(fp, BASE_RGB_ROOT) for fp in sorted(frame_paths)]
            dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
            dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}

    # ------------------------------------------------------------------
    # DFDCP
    # ------------------------------------------------------------------
    elif dataset_name == 'DFDCP':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {
            'DFDCP_Real': {'train': {}, 'test': {}, 'val': {}},
            'DFDCP_FakeA': {'train': {}, 'test': {}, 'val': {}},
            'DFDCP_FakeB': {'train': {}, 'test': {}, 'val': {}}
        }

        with open(os.path.join(dataset_path, 'dataset.json'), 'r') as f:
            dataset_info = json.load(f)

        for dataset in dataset_info.keys():
            index = dataset.split('/')[0]
            vidname = dataset.split('/')[-1].split(".")[0]
            if Path(os.path.join(dataset_path, index, 'frames', vidname)).exists():
                frame_paths = glob.glob(os.path.join(dataset_path, index, 'frames', vidname, '*png'))
                if len(frame_paths) == 0:
                    continue
                label = dataset_info[dataset]['label']
                if label == 'real':
                    label = 'DFDCP_Real'
                elif label == 'fake' and index == 'method_A':
                    label = 'DFDCP_FakeA'
                elif label == 'fake' and index == 'method_B':
                    label = 'DFDCP_FakeB'
                else:
                    raise ValueError(f"wrong in processing vidname {dataset_name}: {dataset}")
                set_attr = dataset_info[dataset]['set']
                frame_paths = [os.path.relpath(fp, BASE_RGB_ROOT) for fp in sorted(frame_paths)]
                dataset_dict[dataset_name][label][set_attr][vidname] = {'label': label, 'frames': frame_paths}

        for label in ['DFDCP_Real', 'DFDCP_FakeA', 'DFDCP_FakeB']:
            dataset_dict[dataset_name][label]['val'] = dataset_dict[dataset_name][label]['test']

    # ------------------------------------------------------------------
    # DFDC
    # ------------------------------------------------------------------
    elif dataset_name == 'DFDC':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {
            'DFDC_Real': {'train': {}, 'test': {}, 'val': {}},
            'DFDC_Fake': {'train': {}, 'test': {}, 'val': {}}
        }
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            if folder.name in ['test']:
                df = pd.read_csv(os.path.join(dataset_path, folder.name, 'labels.csv'))
                labels = ['DFDC_Real', 'DFDC_Fake']
                for _, row in df.iterrows():
                    vidname = row['filename'].split('.mp4')[0]
                    label = labels[row['label']]
                    frame_paths = glob.glob(os.path.join(dataset_path, folder.name, 'frames', vidname, '*png'))
                    if len(frame_paths) == 0:
                        continue
                    frame_paths = [os.path.relpath(fp, BASE_RGB_ROOT) for fp in sorted(frame_paths)]
                    dataset_dict[dataset_name][label]['test'][vidname] = {'label': label, 'frames': frame_paths}
                    dataset_dict[dataset_name][label]['val'][vidname] = {'label': label, 'frames': frame_paths}

            elif folder.name in ['train']:
                num_file = 0
                for dfdc_train_part in os.scandir(os.path.join(dataset_path, folder.name)):
                    if not os.path.isdir(dfdc_train_part):
                        continue
                    num_file += 1
                    print(f'processing {num_file}th file in 50 files.')
                    with open(os.path.join(dfdc_train_part, 'metadata.json'), 'r') as f:
                        metadata = json.load(f)
                    for video_path in os.scandir(os.path.join(dfdc_train_part, 'frames')):
                        if video_path.is_dir():
                            video_name = video_path.name
                            label = metadata[video_name + ".mp4"]["label"]
                            if label == 'REAL':
                                label = 'DFDC_Real'
                            else:
                                label = 'DFDC_Fake'
                            frame_paths = sorted_frame_paths(video_path.path)
                            dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                            dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

    # ------------------------------------------------------------------
    # DeeperForensics-1.0
    # ------------------------------------------------------------------
    elif dataset_name == 'DeeperForensics-1.0':
        with open(os.path.join(dataset_root_path, dataset_name, 'lists/splits/train.txt'), 'r') as f:
            train_txt = [line.strip().split('.')[0] for line in f.readlines()]
        with open(os.path.join(dataset_root_path, dataset_name, 'lists/splits/test.txt'), 'r') as f:
            test_txt = [line.strip().split('.')[0] for line in f.readlines()]
        with open(os.path.join(dataset_root_path, dataset_name, 'lists/splits/val.txt'), 'r') as f:
            val_txt = [line.strip().split('.')[0] for line in f.readlines()]

        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {
            'DF_real': {'train': {}, 'test': {}, 'val': {}},
            'DF_fake': {'train': {}, 'test': {}, 'val': {}}
        }

        if not Path(os.path.join(dataset_path, 'manipulated_videos', perturbation)).exists():
            raise ValueError(f"wrong in processing perturbation {perturbation} in manipulated_videos")

        print(f"processing perturbation {perturbation} in manipulated_videos")
        for video_path in os.scandir(os.path.join(dataset_path, 'manipulated_videos', perturbation, 'frames')):
            if video_path.is_dir():
                video_name = video_path.name
                if video_name in train_txt:
                    set_attr = 'train'
                elif video_name in test_txt:
                    set_attr = 'test'
                elif video_name in val_txt:
                    set_attr = 'val'
                else:
                    raise ValueError(f"wrong in processing vidname {dataset_name}: {video_name}")
                label = 'DF_fake'
                frame_paths = sorted_frame_paths(video_path.path)
                valid_frame_paths = [fp for fp in frame_paths if cv2.imread(os.path.join(BASE_RGB_ROOT, fp)) is not None]
                dataset_dict[dataset_name][label][set_attr][video_name] = {'label': label, 'frames': valid_frame_paths}

        for actor_path in os.scandir(os.path.join(dataset_path, 'source_videos')):
            print("actor", actor_path.name)
            if not os.path.isdir(actor_path):
                continue
            label = 'DF_real'
            video_paths = [os.path.join(actor_path.path, 'frames', video.name) for video in os.scandir(os.path.join(actor_path.path, 'frames'))]
            for video_path in video_paths:
                video_name = video_path.split('/')[-1]
                frame_paths = sorted_frame_paths(video_path)
                valid_frame_paths = [fp for fp in frame_paths if cv2.imread(os.path.join(BASE_RGB_ROOT, fp)) is not None]
                dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': valid_frame_paths}
                dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': valid_frame_paths}
                dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': valid_frame_paths}

    # ------------------------------------------------------------------
    # UADFV
    # ------------------------------------------------------------------
    elif dataset_name == 'UADFV':
        dataset_path = os.path.join(dataset_root_path, dataset_name)
        dataset_dict[dataset_name] = {
            'UADFV_Real': {'train': {}, 'test': {}, 'val': {}},
            'UADFV_Fake': {'train': {}, 'test': {}, 'val': {}}
        }
        for folder in os.scandir(dataset_path):
            if not os.path.isdir(folder):
                continue
            elif folder.name == 'fake':
                for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                    if video_path.is_dir():
                        video_name = video_path.name
                        label = 'UADFV_Fake'
                        frame_paths = sorted_frame_paths(video_path.path)
                        dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}
            elif folder.name == 'real':
                for video_path in os.scandir(os.path.join(dataset_path, folder.name, 'frames')):
                    if video_path.is_dir():
                        video_name = video_path.name
                        label = 'UADFV_Real'
                        frame_paths = sorted_frame_paths(video_path.path)
                        dataset_dict[dataset_name][label]['train'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['test'][video_name] = {'label': label, 'frames': frame_paths}
                        dataset_dict[dataset_name][label]['val'][video_name] = {'label': label, 'frames': frame_paths}

    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    # write the main dataset json
    output_json_path = os.path.join(output_file_path, dataset_name + '.json')
    with open(output_json_path, 'w') as f:
        json.dump(dataset_dict, f)
    print(f"{dataset_name}.json generated successfully.")


if __name__ == '__main__':
    yaml_path = './config.yaml'
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)
        raise

    dataset_name = config['rearrange']['dataset_name']['default']
    dataset_root_path = config['rearrange']['dataset_root_path']['default']
    output_file_path = config['rearrange']['output_file_path']['default']
    comp = config['rearrange']['comp']['default']
    perturbation = config['rearrange']['perturbation']['default']

    BASE_RGB_ROOT = dataset_root_path

    generate_dataset_file(dataset_name, dataset_root_path, output_file_path, comp, perturbation)