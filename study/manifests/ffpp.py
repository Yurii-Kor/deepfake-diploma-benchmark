from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from study.manifests.models import (
    DATASET_FFPP,
    FFPP_MANIPULATIONS,
    LABEL_FAKE,
    LABEL_REAL,
    ROLE_DEVELOPMENT,
    ROLE_FIT,
    ROLE_TEST,
    ROLE_VALIDATION,
    SOURCE_SPLIT_TEST,
    SOURCE_SPLIT_TRAIN,
    SOURCE_SPLIT_VAL,
    SourcePair,
    StudyVideoRecord,
    canonical_source_group_id,
)


FFPP_EXPECTED_TRAIN_PAIRS = 360
FFPP_EXPECTED_VAL_PAIRS = 70
FFPP_EXPECTED_TEST_PAIRS = 70

FFPP_FIT_PAIRS = 288
FFPP_DEVELOPMENT_PAIRS = 72


FFPP_MANIPULATION_DIRS = {
    "DeepFakes": "Deepfakes",
    "Face2Face": "Face2Face",
    "FaceSwap": "FaceSwap",
    "NeuralTextures": "NeuralTextures",
}


def _validate_source_id(
    source_id: object,
    path: Path,
    pair_index: int,
) -> str:
    if not isinstance(source_id, str):
        raise ValueError(
            "FF++ source ID must be a string in {} at pair {}: {!r}"
            .format(
                path,
                pair_index,
                source_id,
            )
        )

    source_id = source_id.strip()

    if len(source_id) != 3 or not source_id.isdigit():
        raise ValueError(
            "Invalid FF++ source ID in {} at pair {}: {!r}"
            .format(
                path,
                pair_index,
                source_id,
            )
        )

    return source_id


def _canonical_pair(
    pair: SourcePair,
) -> SourcePair:
    first, second = pair

    ordered = sorted(
        (
            first,
            second,
        )
    )

    return ordered[0], ordered[1]


def validate_pair_list(
    pairs: Sequence[SourcePair],
    split_name: str,
) -> None:
    """
    Validate source-pair integrity within one official FF++ split.

    Every source ID is expected to occur exactly once inside a split.
    Pair direction is ignored when checking duplicate pair identities.
    """
    seen_source_ids = set()
    seen_pairs = set()

    for pair_index, pair in enumerate(
        pairs,
        start=1,
    ):
        if len(pair) != 2:
            raise ValueError(
                "FF++ pair must contain exactly two source IDs "
                "in split '{}', pair {}: {!r}"
                .format(
                    split_name,
                    pair_index,
                    pair,
                )
            )

        first, second = pair

        if first == second:
            raise ValueError(
                "FF++ source pair contains the same source ID twice "
                "in split '{}', pair {}: {}"
                .format(
                    split_name,
                    pair_index,
                    first,
                )
            )

        canonical_pair = _canonical_pair(
            pair
        )

        if canonical_pair in seen_pairs:
            raise ValueError(
                "Duplicate FF++ source pair in split '{}': {}"
                .format(
                    split_name,
                    canonical_pair,
                )
            )

        seen_pairs.add(
            canonical_pair
        )

        for source_id in pair:
            if source_id in seen_source_ids:
                raise ValueError(
                    "Duplicate FF++ source ID within split '{}': {}"
                    .format(
                        split_name,
                        source_id,
                    )
                )

            seen_source_ids.add(
                source_id
            )


def load_official_pairs(
    path: Path,
) -> List[SourcePair]:
    """
    Load one official FF++ source-pair split.

    The original three-digit string identifiers are preserved exactly.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "FF++ split must contain a list: {}"
            .format(path)
        )

    pairs = []

    for pair_index, raw_pair in enumerate(
        data,
        start=1,
    ):
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
        ):
            raise ValueError(
                "Invalid FF++ pair in {} at position {}: {!r}"
                .format(
                    path,
                    pair_index,
                    raw_pair,
                )
            )

        first = _validate_source_id(
            source_id=raw_pair[0],
            path=path,
            pair_index=pair_index,
        )

        second = _validate_source_id(
            source_id=raw_pair[1],
            path=path,
            pair_index=pair_index,
        )

        pairs.append(
            (
                first,
                second,
            )
        )

    validate_pair_list(
        pairs=pairs,
        split_name=path.stem,
    )

    return pairs


def split_training_pairs(
    train_pairs: Sequence[SourcePair],
    split_seed: int,
) -> Tuple[List[SourcePair], List[SourcePair]]:
    """
    Split the 360 official FF++ training source pairs into FIT and DEV.

    The input pairs are first converted to a deterministic canonical order.
    A local pseudorandom generator then performs one seeded shuffle.

    The resulting assignment is:
        288 pairs -> FIT
         72 pairs -> DEV

    Both returned lists are sorted after assignment so that serialized
    artifacts remain stable and easy to inspect.
    """
    if len(train_pairs) != FFPP_EXPECTED_TRAIN_PAIRS:
        raise ValueError(
            "Expected {} official FF++ training source pairs, found {}."
            .format(
                FFPP_EXPECTED_TRAIN_PAIRS,
                len(train_pairs),
            )
        )

    validate_pair_list(
        pairs=train_pairs,
        split_name=SOURCE_SPLIT_TRAIN,
    )

    canonical_pairs = sorted(
        _canonical_pair(pair)
        for pair in train_pairs
    )

    generator = random.Random(
        split_seed
    )

    generator.shuffle(
        canonical_pairs
    )

    fit_pairs = sorted(
        canonical_pairs[
            :FFPP_FIT_PAIRS
        ]
    )

    development_pairs = sorted(
        canonical_pairs[
            FFPP_FIT_PAIRS:
        ]
    )

    if len(fit_pairs) != FFPP_FIT_PAIRS:
        raise RuntimeError(
            "Unexpected FF++ FIT pair count: {}."
            .format(
                len(fit_pairs)
            )
        )

    if (
        len(development_pairs)
        != FFPP_DEVELOPMENT_PAIRS
    ):
        raise RuntimeError(
            "Unexpected FF++ development pair count: {}."
            .format(
                len(development_pairs)
            )
        )

    return fit_pairs, development_pairs


def _build_real_records(
    pairs: Sequence[SourcePair],
    source_split: str,
    role: str,
) -> List[StudyVideoRecord]:
    records = []

    for pair in pairs:
        first, second = pair

        source_group_id = (
            canonical_source_group_id(
                first,
                second,
            )
        )

        for source_id in (
            first,
            second,
        ):
            relative_source_path = str(
                Path("original")
                / "{}.mp4".format(
                    source_id
                )
            )

            records.append(
                StudyVideoRecord(
                    dataset=DATASET_FFPP,
                    source_split=source_split,
                    role=role,
                    base_video_id=source_id,
                    source_group_id=source_group_id,
                    source_video_id=source_id,
                    target_video_id=None,
                    study_label=LABEL_REAL,
                    manipulation=None,
                    relative_source_path=(
                        relative_source_path
                    ),
                )
            )

    return records


def _build_fake_records(
    pairs: Sequence[SourcePair],
    source_split: str,
    role: str,
) -> List[StudyVideoRecord]:
    records = []

    for pair in pairs:
        first, second = pair

        source_group_id = (
            canonical_source_group_id(
                first,
                second,
            )
        )

        for manipulation in FFPP_MANIPULATIONS:
            manipulation_dir = (
                FFPP_MANIPULATION_DIRS[
                    manipulation
                ]
            )

            for source_id, target_id in (
                (
                    first,
                    second,
                ),
                (
                    second,
                    first,
                ),
            ):
                base_video_id = (
                    "{}_{}".format(
                        source_id,
                        target_id,
                    )
                )

                relative_source_path = str(
                    Path(manipulation_dir)
                    / "{}.mp4".format(
                        base_video_id
                    )
                )

                records.append(
                    StudyVideoRecord(
                        dataset=DATASET_FFPP,
                        source_split=source_split,
                        role=role,
                        base_video_id=base_video_id,
                        source_group_id=source_group_id,
                        source_video_id=source_id,
                        target_video_id=target_id,
                        study_label=LABEL_FAKE,
                        manipulation=manipulation,
                        relative_source_path=(
                            relative_source_path
                        ),
                    )
                )

    return records


def build_records_for_pairs(
    pairs: Sequence[SourcePair],
    source_split: str,
    role: str,
) -> List[StudyVideoRecord]:
    """
    Build the complete nominal FF++ record set for a collection of pairs.

    Each pair contributes:
        2 pristine real videos
        2 DeepFakes videos
        2 Face2Face videos
        2 FaceSwap videos
        2 NeuralTextures videos

    Therefore one complete source pair contributes 10 study records.

    Media availability is deliberately not checked here. The canonical
    manifest represents the nominal experimental assignment first; missing
    source files are handled separately by manifest validation.
    """
    validate_pair_list(
        pairs=pairs,
        split_name=role,
    )

    records = []

    records.extend(
        _build_real_records(
            pairs=pairs,
            source_split=source_split,
            role=role,
        )
    )

    records.extend(
        _build_fake_records(
            pairs=pairs,
            source_split=source_split,
            role=role,
        )
    )

    records.sort(
        key=lambda record: (
            record.source_group_id,
            record.study_label,
            record.manipulation or "",
            record.base_video_id,
        )
    )

    return records


def build_ffpp_study_records(
    train_pairs: Sequence[SourcePair],
    val_pairs: Sequence[SourcePair],
    test_pairs: Sequence[SourcePair],
    split_seed: int,
) -> Tuple[
    List[StudyVideoRecord],
    List[SourcePair],
    List[SourcePair],
]:
    """
    Construct the complete nominal FF++ study manifest.

    Official train:
        288 source pairs -> FIT
         72 source pairs -> development

    Official validation:
        retained unchanged as validation

    Official test:
        retained unchanged as held-out test

    The returned FIT and development pair assignments are exposed
    separately so that they can be serialized as reproducibility artifacts.
    """
    if len(val_pairs) != FFPP_EXPECTED_VAL_PAIRS:
        raise ValueError(
            "Expected {} official FF++ validation source pairs, found {}."
            .format(
                FFPP_EXPECTED_VAL_PAIRS,
                len(val_pairs),
            )
        )

    if len(test_pairs) != FFPP_EXPECTED_TEST_PAIRS:
        raise ValueError(
            "Expected {} official FF++ test source pairs, found {}."
            .format(
                FFPP_EXPECTED_TEST_PAIRS,
                len(test_pairs),
            )
        )

    validate_pair_list(
        pairs=val_pairs,
        split_name=SOURCE_SPLIT_VAL,
    )

    validate_pair_list(
        pairs=test_pairs,
        split_name=SOURCE_SPLIT_TEST,
    )

    fit_pairs, development_pairs = (
        split_training_pairs(
            train_pairs=train_pairs,
            split_seed=split_seed,
        )
    )

    records = []

    records.extend(
        build_records_for_pairs(
            pairs=fit_pairs,
            source_split=SOURCE_SPLIT_TRAIN,
            role=ROLE_FIT,
        )
    )

    records.extend(
        build_records_for_pairs(
            pairs=development_pairs,
            source_split=SOURCE_SPLIT_TRAIN,
            role=ROLE_DEVELOPMENT,
        )
    )

    records.extend(
        build_records_for_pairs(
            pairs=val_pairs,
            source_split=SOURCE_SPLIT_VAL,
            role=ROLE_VALIDATION,
        )
    )

    records.extend(
        build_records_for_pairs(
            pairs=test_pairs,
            source_split=SOURCE_SPLIT_TEST,
            role=ROLE_TEST,
        )
    )

    records.sort(
        key=lambda record: (
            record.role,
            record.source_group_id,
            record.study_label,
            record.manipulation or "",
            record.base_video_id,
        )
    )

    return (
        records,
        fit_pairs,
        development_pairs,
    )