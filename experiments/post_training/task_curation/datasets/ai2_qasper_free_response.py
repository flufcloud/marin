# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""QASPER free-response questions graded against references by an LLM judge."""

from collections.abc import Iterator
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import IntendedUse, RawRow
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation.datasets import ai2_qasper
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

RUBRIC = """
Check that the question is understandable from paper.json and that each private reference answers it using information
supported by the paper. Flag references that are wrong, contradictory or materially incomplete. A semantic judge may
accept paraphrases, but it cannot repair a bad reference. Do not require outside knowledge.
"""


def read_questions(path: StoragePath, _context: ConversionContext) -> Iterator[dict[str, Any]]:
    for row in ai2_qasper.read_questions(path):
        annotations: list[ai2_qasper.QasperAnnotation] = row["answers"]["answer"]
        if not annotations or any(
            answer["unanswerable"]
            or answer["yes_no"] is not None
            or answer["extractive_spans"]
            or not answer["free_form_answer"]
            for answer in annotations
        ):
            continue
        evidence = (item for answer in annotations for item in answer["evidence"])
        if ai2_qasper.uses_missing_evidence(evidence):
            continue
        references = tuple(dict.fromkeys(answer["free_form_answer"] for answer in annotations))
        yield {**row, "references": references}


def convert(row: RawRow, context: ConversionContext) -> TaskSpec:
    data = row.data
    package = verifyit_package(
        JudgeSpec(
            references=tuple(data["references"]),
            question=data["question"],
            output=ai2_qasper.ANSWER_PATH,
            answer_json_field="answer",
        ),
        environment=required_grader_environment(context),
    )
    return ai2_qasper.build_task(row, package, ("qa", "paper", "judge", "reference", "qasper"))


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                id="QASPER:ai2_qasper_free_response",
                title="QASPER Free Response",
                origin="QASPER",
                family="paper-question-answering",
                tags=("agentic", "single-turn", "license:cc-by-4.0"),
                count=361,
                notes="Free-form training questions with text evidence and reference-based judge grading.",
            ),
            config=CurationRecipe(
                name="ai2_qasper_free_response",
                source=HfSource(
                    ai2_qasper.REPOSITORY,
                    ai2_qasper.REVISION,
                    (ai2_qasper.TRAIN_FILE,),
                    SourceFormat.PARQUET,
                    read=read_questions,
                ),
                convert=convert,
                version="1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                grader=GRADER_PACKAGES,
            ),
        )
    ]
