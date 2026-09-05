"""
Study-controlled training-budget semantics.

``nEpochs`` denotes the total completed-epoch budget.

Examples
--------
start_epoch = 0
nEpochs = 10
-> epochs 0..9
-> exactly 10 completed epochs

start_epoch = 5
nEpochs = 10
-> epochs 5..9
-> five remaining epochs
"""


def completed_epoch_indices(
    start_epoch: int,
    n_epochs: int,
):
    if not isinstance(
        start_epoch,
        int,
    ):
        raise TypeError(
            "start_epoch must be an integer"
        )

    if not isinstance(
        n_epochs,
        int,
    ):
        raise TypeError(
            "n_epochs must be an integer"
        )

    if start_epoch < 0:
        raise ValueError(
            "start_epoch must be non-negative"
        )

    if n_epochs < 0:
        raise ValueError(
            "n_epochs must be non-negative"
        )

    if start_epoch > n_epochs:
        raise ValueError(
            "start_epoch cannot exceed n_epochs"
        )

    return range(
        start_epoch,
        n_epochs,
    )


def training_budget_summary(
    steps_per_epoch: int,
    effective_images_per_step: int,
    start_epoch: int,
    n_epochs: int,
):
    if not isinstance(
        steps_per_epoch,
        int,
    ) or steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be a positive integer"
        )

    if not isinstance(
        effective_images_per_step,
        int,
    ) or effective_images_per_step <= 0:
        raise ValueError(
            "effective_images_per_step must be a positive integer"
        )

    epochs = completed_epoch_indices(
        start_epoch=start_epoch,
        n_epochs=n_epochs,
    )

    completed_epochs = len(
        epochs
    )

    optimizer_steps = (
        completed_epochs
        * steps_per_epoch
    )

    image_exposures = (
        optimizer_steps
        * effective_images_per_step
    )

    return {
        "start_epoch": start_epoch,
        "n_epochs": n_epochs,
        "completed_epochs_this_run": completed_epochs,
        "steps_per_epoch": steps_per_epoch,
        "effective_images_per_step": effective_images_per_step,
        "optimizer_steps_this_run": optimizer_steps,
        "image_exposures_this_run": image_exposures,
    }