"""
eval pretrained model.
"""
import os
import csv
import numpy as np
import random
import yaml
from datetime import datetime
from tqdm import tqdm

from metrics.utils import get_test_metrics
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

import argparse

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument(
    '--detector_path',
    type=str,
    default='/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
    help='path to detector YAML file'
)
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument(
    '--weights_path',
    type=str,
    default='/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth'
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda'] and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False
        )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def _extract_prob_tensor(predictions):
    """
    Normalize model outputs so test.py can work with different detector schemas.
    Supports keys like: prob / cls / pred / logits
    """
    if isinstance(predictions, torch.Tensor):
        prob_tensor = predictions
    elif isinstance(predictions, dict):
        if 'prob' in predictions:
            prob_tensor = predictions['prob']
        elif 'cls' in predictions:
            prob_tensor = predictions['cls']
        elif 'pred' in predictions:
            prob_tensor = predictions['pred']
        elif 'logits' in predictions:
            prob_tensor = predictions['logits']
        else:
            raise KeyError(f"Model output keys do not include prob/cls/pred/logits: {list(predictions.keys())}")
    else:
        raise TypeError(f"Unsupported prediction output type: {type(predictions)}")

    if not torch.is_tensor(prob_tensor):
        prob_tensor = torch.tensor(prob_tensor)

    # Convert logits / 2-class outputs to positive-class probability-like score
    if prob_tensor.ndim > 1:
        if prob_tensor.shape[-1] == 1:
            prob_tensor = prob_tensor.squeeze(-1)
        else:
            prob_tensor = torch.softmax(prob_tensor, dim=1)[:, 1]

    return prob_tensor


def _extract_feat_tensor(predictions):
    """
    Optional feature extractor. If absent, return None.
    """
    if isinstance(predictions, dict):
        for key in ['feat', 'feature', 'features', 'embedding', 'emb']:
            if key in predictions:
                feat_tensor = predictions[key]
                if torch.is_tensor(feat_tensor):
                    return feat_tensor
                return torch.tensor(feat_tensor)
    return None


def _dataset_id_from_subset(subset_name: str) -> str:
    ffpp_subsets = {'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT'}
    return 'FF++' if subset_name in ffpp_subsets else subset_name


def _source_video_id_from_img_name(img_name):
    """
    Mirror benchmark video grouping key:
    parent directory name of the frame path.
    Works for both string paths and list-based video entries.
    """
    if isinstance(img_name, list):
        img_name = img_name[0]

    s = str(img_name).replace('\\', '/')
    parts = s.split('/')
    if len(parts) >= 2:
        return parts[-2]
    return os.path.splitext(os.path.basename(s))[0]


def _stringify_img_name(img_name):
    """
    Store original image reference in CSV.
    For video-level entries (list of paths), join them safely.
    """
    if isinstance(img_name, list):
        return ' || '.join(str(x) for x in img_name)
    return str(img_name)


def _make_run_id(config, weights_path):
    model_id = config.get('model_name', 'unknown_model')
    condition_code = config.get('condition_code', 'C0')
    weights_stem = os.path.splitext(os.path.basename(weights_path or 'unknown'))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{model_id}_{condition_code}_{weights_stem}_{ts}'


def _write_sample_scores_csv(csv_path, run_id, config, subset_name, img_names, predictions, labels):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_id',
            'model_id',
            'family_id',
            'checkpoint_id',
            'dataset_id',
            'subset_id',
            'split',
            'condition_code',
            'image_path',
            'video_id',
            'source_video_id',
            'score',
            'label',
            'score_available',
            'face_found',
            'failure_stage',
            'failure_reason',
        ])

        for img_name, score, label in zip(img_names, predictions, labels):
            source_video_id = _source_video_id_from_img_name(img_name)

            writer.writerow([
                run_id,
                config.get('model_name', 'unknown_model'),
                config.get('family_id', 'unknown_family'),
                os.path.basename(config.get('weights_path', 'unknown')),
                _dataset_id_from_subset(subset_name),
                subset_name,
                config.get('split_tag', 'test'),
                config.get('condition_code', 'C0'),
                _stringify_img_name(img_name),
                source_video_id,
                source_video_id,
                float(score),
                int(label),
                True,
                True,
                '',
                '',
            ])


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []

    for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        data, label, mask, landmark = (
            data_dict['image'],
            data_dict['label'],
            data_dict['mask'],
            data_dict['landmark']
        )

        label = torch.where(data_dict['label'] != 0, 1, 0)

        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        predictions = inference(model, data_dict)

        prob_tensor = _extract_prob_tensor(predictions)
        feat_tensor = _extract_feat_tensor(predictions)

        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(prob_tensor.cpu().detach().numpy())

        if feat_tensor is not None:
            feature_lists += list(feat_tensor.cpu().detach().numpy())

    feat_array = np.array(feature_lists) if len(feature_lists) > 0 else None
    return np.array(prediction_lists), np.array(label_lists), feat_array


def test_epoch(model, test_data_loaders, config, run_id):
    model.eval()
    metrics_all_datasets = {}

    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict

        predictions_nps, label_nps, feat_nps = test_one_dataset(model, test_data_loaders[key])

        study_output_root = os.path.expanduser(
            config.get('study_output_root', '~/deepfake_lab/study_outputs')
        )
        raw_scores_dir = os.path.join(study_output_root, 'raw_scores', config.get('split_tag', 'test'))
        raw_scores_path = os.path.join(
            raw_scores_dir,
            f'{run_id}_{key}_frame.csv'
        )

        _write_sample_scores_csv(
            csv_path=raw_scores_path,
            run_id=run_id,
            config=config,
            subset_name=key,
            img_names=data_dict['image'],
            predictions=predictions_nps,
            labels=label_nps,
        )

        metric_one_dataset = get_test_metrics(
            y_pred=predictions_nps,
            y_true=label_nps,
            img_names=data_dict['image']
        )
        metrics_all_datasets[key] = metric_one_dataset

        tqdm.write(f"dataset: {key}")
        tqdm.write(f"raw_scores_csv: {raw_scores_path}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    with open(args.detector_path, 'r') as f:
        detector_config = yaml.safe_load(f)

    with open('./training/config/test_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    config = dict(base_config)
    config.update(detector_config)

    weights_path = None

    if args.test_dataset:
        config['test_dataset'] = args.test_dataset

    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path

    init_seed(config)

    if config['cudnn'] and device.type == 'cuda':
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    run_id = _make_run_id(config, weights_path)
    best_metric = test_epoch(model, test_data_loaders, config, run_id)
    print('===> Test Done!')
    print(best_metric)


if __name__ == '__main__':
    main()