"""
Executable validation of study-controlled training reproducibility.

Run from the repository root:

    python -m study.training.reproducibility_validation
"""

import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .reproducibility import (
    configure_cudnn,
    make_data_loader_generator,
    seed_data_loader_worker,
    seed_training_process,
    validate_seed,
)


class WorkerRandomDataset(Dataset):
    """
    Tiny synthetic dataset exposing RNG outputs produced inside workers.
    """

    def __init__(
        self,
        size: int = 16,
    ) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int,
    ):
        return {
            "index": index,
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": float(torch.rand(1).item()),
        }


def _expect_failure(
    description,
    operation,
) -> None:
    try:
        operation()
    except (TypeError, ValueError):
        return

    raise AssertionError(
        "Expected failure was not raised: {}".format(
            description
        )
    )


def _main_process_signature(
    seed: int,
) -> Tuple[float, float, float]:
    seed_training_process(
        seed=seed,
        use_cuda=False,
    )

    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )


def _collect_loader_signature(
    seed: int,
) -> List[Tuple[int, float, float, float]]:
    dataset = WorkerRandomDataset(
        size=16,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_data_loader_worker,
        generator=make_data_loader_generator(seed),
    )

    signature = []

    for batch in loader:
        batch_size = len(
            batch["index"]
        )

        for position in range(
            batch_size
        ):
            signature.append(
                (
                    int(
                        batch["index"][position].item()
                    ),
                    float(
                        batch["python"][position].item()
                    ),
                    float(
                        batch["numpy"][position].item()
                    ),
                    float(
                        batch["torch"][position].item()
                    ),
                )
            )

    return signature


def _validate_main_process_rng() -> None:
    signature_a = _main_process_signature(
        seed=1024
    )

    signature_b = _main_process_signature(
        seed=1024
    )

    if signature_a != signature_b:
        raise AssertionError(
            "main-process RNG streams were not reproducible"
        )

    signature_c = _main_process_signature(
        seed=1025
    )

    if signature_a == signature_c:
        raise AssertionError(
            "different seeds unexpectedly produced identical RNG output"
        )

    print("MAIN-PROCESS RNG")
    print(
        "  Python random:              reproducible"
    )
    print(
        "  NumPy:                     reproducible"
    )
    print(
        "  PyTorch CPU:               reproducible"
    )
    print(
        "  different seed:            different stream"
    )


def _validate_data_loader_workers() -> None:
    signature_a = _collect_loader_signature(
        seed=1024
    )

    signature_b = _collect_loader_signature(
        seed=1024
    )

    if signature_a != signature_b:
        raise AssertionError(
            "DataLoader worker streams were not reproducible"
        )

    signature_c = _collect_loader_signature(
        seed=1025
    )

    if signature_a == signature_c:
        raise AssertionError(
            "different DataLoader seeds unexpectedly produced "
            "identical worker output"
        )

    indices = [
        item[0]
        for item in signature_a
    ]

    if sorted(indices) != list(range(16)):
        raise AssertionError(
            "DataLoader did not expose every synthetic item exactly once"
        )

    print()
    print("DATALOADER WORKERS")
    print(
        "  shuffle order:             reproducible"
    )
    print(
        "  Python worker RNG:         reproducible"
    )
    print(
        "  NumPy worker RNG:          reproducible"
    )
    print(
        "  PyTorch worker RNG:        reproducible"
    )
    print(
        "  different seed:            different stream"
    )


def _validate_cudnn_policy() -> None:
    configure_cudnn(
        deterministic=True
    )

    if torch.backends.cudnn.benchmark:
        raise AssertionError(
            "cuDNN benchmark must be disabled"
        )

    if not torch.backends.cudnn.deterministic:
        raise AssertionError(
            "cuDNN deterministic flag must be enabled"
        )

    print()
    print("CUDNN POLICY")
    print(
        "  benchmark:                 disabled"
    )
    print(
        "  deterministic:             enabled"
    )


def _validate_negative_cases() -> None:
    _expect_failure(
        "negative seed",
        lambda: validate_seed(-1),
    )

    _expect_failure(
        "non-integer seed",
        lambda: validate_seed("1024"),
    )

    _expect_failure(
        "invalid deterministic flag",
        lambda: configure_cudnn(
            deterministic=1
        ),
    )

    _expect_failure(
        "negative worker id",
        lambda: seed_data_loader_worker(-1),
    )

    print()
    print("NEGATIVE / INVARIANT TESTS")
    print(
        "  invalid configurations:    rejected"
    )


def main() -> None:
    _validate_main_process_rng()
    _validate_data_loader_workers()
    _validate_cudnn_policy()
    _validate_negative_cases()

    print()
    print(
        "REPRODUCIBILITY VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()