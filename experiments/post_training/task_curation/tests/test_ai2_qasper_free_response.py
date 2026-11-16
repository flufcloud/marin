# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from taskcompendium.grading import grade_remote_judge_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, ConversationTrace, GradingAttempt, TextMessage

from experiments.post_training.task_curation.datasets import ai2_qasper, ai2_qasper_free_response
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.tests.conversion import converted_task

RECIPE = cast(CurationRecipe, ai2_qasper_free_response.sources()[0].config)
ROW = {
    "paper": {"id": "paper-1"},
    "question": "How does the method propagate labels?",
    "question_id": "question-1",
    "answers": {},
    "references": ("It uses discourse relations.",),
}


def test_grades_json_answer_file_against_free_response_reference():
    task = converted_task(RECIPE, ROW)
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done.")))
    attempt = GradingAttempt(
        conversation,
        files={ai2_qasper.ANSWER_PATH: b'{"answer":"It uses discourse relations."}'},
    )

    result = grade_remote_judge_answer(task, attempt, None)

    assert task.answer_type == AnswerType.FILE
    assert result.status == Outcome.GRADED
    assert result.reward == 1.0
