from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from career_agent.agent.main_agent_contracts import (
    ConversationResourceReference,
    ToolObservation,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.services.episode_consolidation import (
    EpisodeDraftCoverageError,
    drafts_from_tool_results,
)
from career_agent.services.episode_reconciliation import (
    EpisodeReconciler,
    EpisodeReconciliationLimitError,
    EpisodeReconciliationResult,
)
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


class EmptyApplicationSource:
    def list(self, **kwargs):
        return ()


class EmptyInterviewSource:
    def list(self, **kwargs):
        return ()


class EmptyJobResearchSource:
    def list_reports(self, **kwargs):
        return ()


class CompletedMockSource:
    def __init__(self) -> None:
        now = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.session = SimpleNamespace(
            id="mock-1",
            status="completed",
            completed_at=now,
            updated_at=now,
        )
        self.report = SimpleNamespace(
            id="report-1",
            session_id="mock-1",
            summary="STAR 回答缺少可量化结果。",
        )
        self.report_reads = 0

    def list_sessions(self, **kwargs):
        return (self.session,)

    def list_reports(self, **kwargs):
        self.report_reads += 1
        return (self.report,)


def test_committed_tool_results_become_one_episode_per_domain_entity() -> None:
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    drafts = drafts_from_tool_results(
        user_id="u1",
        conversation_id="conversation-1",
        occurred_at=now,
        tool_results=(
            ToolObservation(
                tool_name="create_application",
                state="application_ready",
                message="已创建投递记录。",
                execution_outcome="committed",
                payload={
                    "application_id": "application-1",
                    "company_name": "Example",
                    "title": "ML Engineer",
                },
            ),
            ToolObservation(
                tool_name="research_job",
                state="job_research_ready",
                message="岗位研究已完成。",
                execution_outcome="committed",
                payload={
                    "run_id": "research-1",
                    "research": {"summary": "公司正在扩展 AI 平台团队。"},
                },
                resource_ref=ConversationResourceReference(
                    kind="job_research_report",
                    resource_id="report-1",
                    title="Example 公司研究",
                    status_at_delivery="current",
                    anchored_by_other_job=False,
                ),
            ),
            ToolObservation(
                tool_name="update_application_status",
                state="application_not_found",
                message="没有找到投递记录。",
                execution_outcome="not_committed",
            ),
        ),
    )

    assert [(draft.kind, draft.source_run_id) for draft in drafts] == [
        ("application", "application-1"),
        ("job_research", "research-1"),
    ]
    assert drafts[1].summary == "公司正在扩展 AI 平台团队。"
    assert drafts[1].resource_refs[0].resource_id == "report-1"


def test_committed_episodic_tool_cannot_silently_skip_its_source_pointer() -> None:
    with pytest.raises(EpisodeDraftCoverageError, match="application_id"):
        drafts_from_tool_results(
            user_id="u1",
            conversation_id="conversation-1",
            tool_results=(
                ToolObservation(
                    tool_name="create_application",
                    state="application_ready",
                    message="已创建投递记录。",
                    execution_outcome="committed",
                    payload={"title": "ML Engineer"},
                ),
            ),
        )


def test_resume_intent_and_tailoring_successes_become_episodes() -> None:
    drafts = drafts_from_tool_results(
        user_id="u1",
        conversation_id="conversation-1",
        tool_results=(
            ToolObservation(
                tool_name="confirm_resume_analysis",
                state="resume_analysis_confirmed",
                message="已确认并保存 2 段职业经历。",
                execution_outcome="committed",
                payload={"analysis_id": "analysis-1"},
            ),
            ToolObservation(
                tool_name="confirm_job_intent",
                state="job_intent_recorded",
                message="已保存目标岗位和城市。",
                execution_outcome="committed",
                payload={"intent_episode_id": "intent-1"},
            ),
            ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_tailoring_finalized",
                message="已生成新的不可变 Markdown 简历版本。",
                execution_outcome="committed",
                payload={"draft_id": "draft-1", "created": True},
            ),
            ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_tailoring_finalized",
                message="已返回原结果。",
                execution_outcome="committed",
                payload={"draft_id": "draft-2", "created": False},
            ),
        ),
    )

    assert [(draft.kind, draft.source_run_id) for draft in drafts] == [
        ("resume_analysis", "analysis-1"),
        ("intent_confirmation", "intent-1"),
        ("resume_tailoring", "draft-1"),
    ]


def test_the_runtime_sweeps_each_user_once_per_process_not_once_per_turn() -> None:
    class CountingReconciler:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def reconcile_user(self, *, user_id: str):
            self.calls.append(user_id)
            return EpisodeReconciliationResult(scanned=0, inserted=0)

    reconciler = CountingReconciler()
    runtime = MainAgentRuntime.__new__(MainAgentRuntime)
    runtime._episode_reconciler = reconciler
    runtime._reconciled_users = set()
    runtime._episode_reconcile_guard = Lock()
    runtime._episode_reconcile_locks = {}

    for user_id in ("u1", "u1", "u1", "u2", "u1"):
        runtime._reconcile_episodes(user_id)

    # The sweep is a full scan of four domain stores plus one report read per
    # mock session, so repeating it per turn would make every turn pay for the
    # whole history. Seam writes keep L1 current after the first pass.
    assert reconciler.calls == ["u1", "u2"]


def test_a_runtime_without_a_reconciler_still_takes_turns() -> None:
    runtime = MainAgentRuntime.__new__(MainAgentRuntime)
    runtime._episode_reconciler = None
    runtime._reconciled_users = set()
    runtime._episode_reconcile_guard = Lock()
    runtime._episode_reconcile_locks = {}

    runtime._reconcile_episodes("u1")


def test_a_failed_turn_invalidates_the_user_sweep_guard() -> None:
    class CountingReconciler:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def reconcile_user(self, *, user_id: str):
            self.calls.append(user_id)
            return EpisodeReconciliationResult(scanned=0, inserted=0)

    reconciler = CountingReconciler()
    runtime = MainAgentRuntime.__new__(MainAgentRuntime)
    runtime._episode_reconciler = reconciler
    runtime._reconciled_users = set()
    runtime._episode_reconcile_guard = Lock()
    runtime._episode_reconcile_locks = {}

    runtime._reconcile_episodes("u1")
    runtime._invalidate_episode_reconciliation("u1")
    runtime._reconcile_episodes("u1")

    assert reconciler.calls == ["u1", "u1"]


def test_one_users_sweep_does_not_block_another_user() -> None:
    first_started = Event()
    release_first = Event()

    class BlockingReconciler:
        def reconcile_user(self, *, user_id: str):
            if user_id == "u1":
                first_started.set()
                assert release_first.wait(timeout=2)
            return EpisodeReconciliationResult(scanned=0, inserted=0)

    runtime = MainAgentRuntime.__new__(MainAgentRuntime)
    runtime._episode_reconciler = BlockingReconciler()
    runtime._reconciled_users = set()
    runtime._episode_reconcile_guard = Lock()
    runtime._episode_reconcile_locks = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(runtime._reconcile_episodes, "u1")
        assert first_started.wait(timeout=2)
        second = executor.submit(runtime._reconcile_episodes, "u2")
        second.result(timeout=1)
        release_first.set()
        first.result(timeout=1)


def test_reconciliation_replays_a_completed_domain_row_exactly_once(tmp_path) -> None:
    episodes = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    mock_source = CompletedMockSource()
    reconciler = EpisodeReconciler(
        episodes=episodes,
        applications=EmptyApplicationSource(),
        interviews=EmptyInterviewSource(),
        mock_interviews=mock_source,
        job_research=EmptyJobResearchSource(),
    )

    first = reconciler.reconcile_user(user_id="u1")
    second = reconciler.reconcile_user(user_id="u1")

    assert (first.scanned, first.inserted) == (1, 1)
    assert (second.scanned, second.inserted) == (1, 0)
    episode = episodes.get_by_source(
        user_id="u1",
        kind="mock_interview",
        source_run_id="mock-1",
    )
    assert episode is not None
    assert episode.summary == "STAR 回答缺少可量化结果。"
    assert len(episodes.search(user_id="u1", query="STAR")) == 1
    assert mock_source.report_reads == 2


def test_reconciliation_does_not_add_career_scope_bindings(tmp_path) -> None:
    episodes = SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3")
    mock_source = CompletedMockSource()
    reconciler = EpisodeReconciler(
        episodes=episodes,
        applications=EmptyApplicationSource(),
        interviews=EmptyInterviewSource(),
        mock_interviews=mock_source,
        job_research=EmptyJobResearchSource(),
    )
    assert reconciler.reconcile_user(user_id="u1").inserted == 1
    with episodes._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM career_episode_memory_bindings"
        ).fetchone()[0] == 0


def test_reconciliation_fails_instead_of_silently_truncating_a_source(
    tmp_path,
) -> None:
    class OverflowApplications:
        def list(self, *, limit: int, **kwargs):
            return (object(),) * limit

    reconciler = EpisodeReconciler(
        episodes=SQLiteCareerEpisodeStore(tmp_path / "context.sqlite3"),
        applications=OverflowApplications(),
        interviews=EmptyInterviewSource(),
        mock_interviews=CompletedMockSource(),
        job_research=EmptyJobResearchSource(),
    )

    with pytest.raises(
        EpisodeReconciliationLimitError,
        match="refusing a silently incomplete",
    ):
        reconciler.reconcile_user(user_id="u1")
