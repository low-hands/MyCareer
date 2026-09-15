"""An attached resume version reaches the model as controlled context, not text.

The request names a version by id. The runtime verifies the id under the
authenticated user, hands the model metadata plus a bounded excerpt, and
stores only the reference (with a display snapshot) on the user message. The
file, the full text, and the raw id never appear in the transcript or the
prompt.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.input_resources import (
    InputResourceNotFoundError,
    InputResourceRejectedError,
    excerpt_budgets,
)
from career_agent.agent.main_agent_contracts import (
    ATTACHED_RESUME_EXCERPT_CHARS,
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.harness.streaming import TurnFailedEvent, TurnInputResource
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import ResumeStore

RESUME_TEXT = "张三\n后端工程师\nPRIVATE-LINE 在检索系统做过端到端优化。"


class RecordingDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def _runtime(tmp_path, decisions, store):
    context_store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(context_store)
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(resume_store=store),
    )
    return runtime, context_store


def _seed(tmp_path, content: bytes = RESUME_TEXT.encode("utf-8")):
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role = store.create_target_role(user_id="u1", title="后端工程师", priority=1)
    resume, version = store.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="主简历",
        content=content,
        document_format="text",
    )
    return store, resume, version


def test_an_attached_version_is_verified_bounded_and_referenced_not_copied(tmp_path):
    store, resume, version = _seed(tmp_path)
    decisions = RecordingDecisionMaker(AgentDecision(action="final", message="这份简历偏后端。"))
    runtime, context_store = _runtime(tmp_path, decisions, store)

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我分析这份简历",
        input_resources=(TurnInputResource(kind="resume_version", id=version.id),),
    )

    context = decisions.contexts[0]
    [attached] = context.attached_resumes
    assert attached.resume_name == "主简历"
    assert attached.target_role == "后端工程师"
    assert attached.version_number == 1 and attached.is_latest_version
    assert "PRIVATE-LINE" in attached.excerpt and not attached.excerpt_truncated
    assert context.task.active_resume_version_id == version.id

    projection = project_decision_messages(context)
    prompt = json.dumps(
        [
            message["content"]
            for message in projection.messages(system_prompt="policy", spotlight_nonce="n")
        ],
        ensure_ascii=False,
    )
    assert "PRIVATE-LINE" in prompt
    assert version.id not in prompt and resume.id not in prompt
    assert "[runtime resources: resume_" in projection.current_user_message

    [user_message, _assistant] = context_store.list_messages("u1", "c1", limit=10)
    assert user_message.content == "帮我分析这份简历"
    [reference] = user_message.resource_refs
    assert reference.kind == "resume_version" and reference.resource_id == version.id
    assert reference.title == "主简历 v1"
    assert "PRIVATE-LINE" not in user_message.model_dump_json()
    with sqlite3.connect(context_store.path) as connection:
        dump = "\n".join(connection.iterdump())
    assert "PRIVATE-LINE" not in dump


def test_the_reference_pins_the_version_used_even_after_a_newer_import(tmp_path):
    store, resume, first = _seed(tmp_path)
    decisions = RecordingDecisionMaker(AgentDecision(action="final", message="好。"))
    runtime, context_store = _runtime(tmp_path, decisions, store)
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="看看这版",
        input_resources=(TurnInputResource(kind="resume_version", id=first.id),),
    )

    _, second = store.import_document(
        user_id="u1", resume_id=resume.id, content=b"newer", document_format="text"
    )

    [user_message, _] = context_store.list_messages("u1", "c1", limit=10)
    assert user_message.resource_refs[0].resource_id == first.id != second.id


def test_a_version_of_another_user_fails_before_anything_is_written(tmp_path):
    store, _resume, version = _seed(tmp_path)
    store.create_target_role(user_id="u2", title="PM", priority=1)
    decisions = RecordingDecisionMaker()
    runtime, context_store = _runtime(tmp_path, decisions, store)
    ContextManager(context_store).upsert_profile(CareerProfileContext(user_id="u2"))
    events = []

    with pytest.raises(InputResourceNotFoundError):
        runtime.run_turn(
            user_id="u2",
            conversation_id="c-other",
            user_message="分析这份简历",
            input_resources=(TurnInputResource(kind="resume_version", id=version.id),),
            event_sink=events.append,
        )

    assert decisions.contexts == []
    assert context_store.list_messages("u2", "c-other", limit=10) == ()
    failed = [event for event in events if isinstance(event, TurnFailedEvent)]
    assert [event.code for event in failed] == ["INPUT_RESOURCE_NOT_FOUND"]
    assert version.id not in failed[0].message


def test_a_long_resume_is_clipped_and_an_unreadable_one_is_flagged(tmp_path):
    long_text = ("一行经历。" * 40 + "\n") * 200
    store, _resume, long_version = _seed(tmp_path, long_text.encode("utf-8"))
    role = store.list_target_roles(user_id="u1")[0]
    _, scanned = store.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="扫描件",
        content=b"\xff\xfe\x00 not utf-8",
        document_format="text",
    )
    decisions = RecordingDecisionMaker(AgentDecision(action="final", message="收到。"))
    runtime, _ = _runtime(tmp_path, decisions, store)

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="对比这两版",
        input_resources=(
            TurnInputResource(kind="resume_version", id=long_version.id),
            TurnInputResource(kind="resume_version", id=scanned.id),
        ),
    )

    clipped, unreadable = decisions.contexts[0].attached_resumes
    assert clipped.excerpt_truncated and len(clipped.excerpt) <= ATTACHED_RESUME_EXCERPT_CHARS
    assert unreadable.text_unavailable and unreadable.excerpt is None


def test_the_excerpt_budget_is_per_turn_and_shared_across_attachments(tmp_path):
    store, _resume, first = _seed(tmp_path, ("经历A。" * 3000).encode("utf-8"))
    role = store.list_target_roles(user_id="u1")[0]
    versions = [first]
    for index in range(7):
        _, version = store.import_document(
            user_id="u1",
            target_role_id=role.id,
            name=f"简历{index}",
            content=(f"经历{index}。" * 3000).encode("utf-8"),
            document_format="text",
        )
        versions.append(version)
    decisions = RecordingDecisionMaker(AgentDecision(action="final", message="收到。"))
    runtime, _ = _runtime(tmp_path, decisions, store)

    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="对比这八份",
        input_resources=tuple(
            TurnInputResource(kind="resume_version", id=version.id) for version in versions
        ),
    )

    attached = decisions.contexts[0].attached_resumes
    assert len(attached) == 8 and all(item.excerpt_truncated for item in attached)
    assert sum(len(item.excerpt) for item in attached) <= ATTACHED_RESUME_EXCERPT_CHARS


def test_a_short_attachment_hands_its_unused_share_to_the_long_ones():
    assert excerpt_budgets((100, 50_000, 20_000), 12_000) == (100, 5_950, 5_950)
    assert excerpt_budgets((0, 0), 12_000) == (0, 0)
    assert sum(excerpt_budgets((9_000,) * 8, 12_000)) <= 12_000


def test_attachments_are_refused_while_a_workflow_owns_the_conversation(tmp_path):
    store, _resume, version = _seed(tmp_path)
    decisions = RecordingDecisionMaker()
    runtime, context_store = _runtime(tmp_path, decisions, store)
    manager = ContextManager(context_store)
    manager.commit_workflow_turn(
        context=manager.load_for_workflow_turn(
            user_id="u1",
            conversation_id="c1",
            task=ConversationTaskState(active_workflow="mock_interview", run_id="mock-1"),
        ),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
            phase="mock_interview_answer_required",
        ),
    )
    events = []

    with pytest.raises(InputResourceRejectedError):
        runtime.run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="顺便看看这份简历",
            input_resources=(TurnInputResource(kind="resume_version", id=version.id),),
            event_sink=events.append,
        )

    assert decisions.contexts == []
    failed = [event for event in events if isinstance(event, TurnFailedEvent)]
    assert [event.code for event in failed] == ["INPUT_RESOURCE_REJECTED"]
    assert "退出" in failed[0].message and version.id not in failed[0].message
    task = manager.get_task(user_id="u1", conversation_id="c1")
    assert task.active_workflow == "mock_interview" and task.run_id == "mock-1"
