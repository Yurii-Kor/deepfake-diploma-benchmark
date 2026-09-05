from .budget import (
    completed_epoch_indices,
    training_budget_summary,
)


def _expect_failure(
    operation,
) -> None:
    try:
        operation()
    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "expected failure was not raised"
    )


def main() -> None:
    epochs = list(
        completed_epoch_indices(
            start_epoch=0,
            n_epochs=10,
        )
    )

    if epochs != list(
        range(10)
    ):
        raise AssertionError(
            "10-epoch full-run semantics are incorrect"
        )

    resume_epochs = list(
        completed_epoch_indices(
            start_epoch=5,
            n_epochs=10,
        )
    )

    if resume_epochs != [
        5,
        6,
        7,
        8,
        9,
    ]:
        raise AssertionError(
            "resume semantics are incorrect"
        )

    summary = training_budget_summary(
        steps_per_epoch=4608,
        effective_images_per_step=32,
        start_epoch=0,
        n_epochs=10,
    )

    if (
        summary[
            "optimizer_steps_this_run"
        ]
        != 46080
    ):
        raise AssertionError(
            "unexpected optimizer-step budget"
        )

    if (
        summary[
            "image_exposures_this_run"
        ]
        != 1474560
    ):
        raise AssertionError(
            "unexpected image-exposure budget"
        )

    _expect_failure(
        lambda: completed_epoch_indices(
            start_epoch=-1,
            n_epochs=10,
        )
    )

    _expect_failure(
        lambda: completed_epoch_indices(
            start_epoch=11,
            n_epochs=10,
        )
    )

    print("TRAINING BUDGET")
    print(
        "  epoch indices:              0..9"
    )
    print(
        "  completed epochs:           10"
    )
    print(
        "  optimizer steps/epoch:      4608"
    )
    print(
        "  optimizer steps total:      46080"
    )
    print(
        "  effective images/step:      32"
    )
    print(
        "  image exposures total:      1474560"
    )
    print(
        "  resume 5 -> 10:             epochs 5..9"
    )

    print()
    print(
        "TRAINING BUDGET VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()