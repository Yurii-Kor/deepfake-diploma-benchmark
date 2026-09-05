"""
Study-controlled training reproducibility helpers.

The upstream DeepfakeBench training entry point seeds Python ``random`` and
PyTorch, but does not explicitly seed NumPy or DataLoader workers and enables
cuDNN benchmarking. The study training protocol uses the helpers in this
module to make stochastic training inputs reproducible as far as supported by
the underlying PyTorch/CUDA operations.

The split RNG used in Section 5.2 and the training RNG used here are
conceptually independent even when they use the same numerical seed (1024).
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn


UINT32_MODULUS = 2 ** 32


def validate_seed(seed: int) -> int:
    """
    Validate and return an integer study seed.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    return seed


def seed_training_process(
    seed: int,
    use_cuda: bool = True,
) -> None:
    """
    Seed RNGs used by the main training process.

    Seeds:
    * Python ``random``
    * NumPy
    * PyTorch CPU
    * all visible CUDA devices when CUDA is requested and available
    """
    seed = validate_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cudnn(
    deterministic: bool = True,
) -> None:
    """
    Configure cuDNN for the study training protocol.

    When deterministic=True:
    * algorithm benchmarking is disabled;
    * deterministic cuDNN implementations are requested.

    This does not claim that every possible CUDA operation in every detector
    is bitwise deterministic. Unsupported or nondeterministic operations must
    still be treated explicitly if encountered during real training.
    """
    if not isinstance(deterministic, bool):
        raise TypeError(
            "deterministic must be a boolean"
        )

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.deterministic = False


def make_data_loader_generator(
    seed: int,
) -> torch.Generator:
    """
    Return a deterministically seeded PyTorch generator for DataLoader use.
    """
    seed = validate_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_data_loader_worker(
    worker_id: int,
) -> None:
    """
    Seed Python and NumPy RNGs inside a DataLoader worker.

    PyTorch assigns each worker a deterministic ``torch.initial_seed()``
    derived from the DataLoader generator. The lower 32 bits are used for
    NumPy and Python RNGs.

    ``worker_id`` is accepted because this function is passed directly as
    ``worker_init_fn``; the seed itself is obtained from PyTorch.
    """
    if not isinstance(worker_id, int):
        raise TypeError(
            "worker_id must be an integer"
        )

    if worker_id < 0:
        raise ValueError(
            "worker_id must be non-negative"
        )

    worker_seed = (
        torch.initial_seed()
        % UINT32_MODULUS
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_training_reproducibility(
    seed: int,
    use_cuda: bool = True,
    deterministic_cudnn: bool = True,
) -> torch.Generator:
    """
    Configure study-level training randomness and return a DataLoader generator.
    """
    seed_training_process(
        seed=seed,
        use_cuda=use_cuda,
    )

    configure_cudnn(
        deterministic=deterministic_cudnn,
    )

    return make_data_loader_generator(
        seed=seed,
    )