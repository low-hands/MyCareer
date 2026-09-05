"""Both ways to read a saved JD reach the same bytes.

The contract that makes "ask the agent, or go look yourself" a real choice
rather than one path with a detour. It was asserted in two docstrings and
checked nowhere: a later change to either presenter would let the two surfaces
drift while both files still claimed they could not.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.api.reads import WorkspaceReader
from career_agent.storage.jobs import SQLiteJobPostingRepository


NOW = datetime(2026, 9, 5, 2, 0, tzinfo=timezone.utc)

# Structure on purpose: a JD reflowed into one paragraph is a different
# document to read, and the two surfaces must not disagree about that either.
JD_TEXT = "岗位职责：\n1. 负责检索系统的端到端优化\n\n任职要求：\n1. 熟悉大模型应用"


@pytest.fixture
def saved(tmp_path):
    store = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    detail = JobDetail(
        source_name="boss",
        source_job_id="job-a",
        source_url="https://www.zhipin.com/job_detail/job-a.html",
        title="AI Engineer",
        company_name="Acme",
        description=JD_TEXT,
        captured_at=NOW,
        provenance=Provenance(
            source_name="boss",
            source_job_id="job-a",
            source_url="https://www.zhipin.com/job_detail/job-a.html",
            captured_at=NOW,
            operation="detail",
            adapter_version="test-v1",
        ),
    )
    return store, store.save_captured_detail(user_id="u1", detail=detail).posting.id


def test_the_agent_hands_back_the_stored_snapshot_without_restating_it(saved):
    """No model sits between the file and the reader on this path.

    ``get_saved_job`` once answered "他的完整jd" with the agent's own analysis,
    which is the complaint this presenter exists to answer. Returning the
    snapshot verbatim is also what lets the other surface exist at all.
    """
    store, job_id = saved
    registry = MainAgentToolRegistry(job_repository=store)

    observation = registry.invoke_atomic_tool(
        "get_saved_job", {"user_id": "u1", "job_posting_id": job_id}
    )
    delivered = MainAgentRuntime._assistant_message(observation)

    assert observation.state == "saved_job_ready"
    assert delivered == JD_TEXT


def test_both_surfaces_read_one_document_not_two_renderings_of_it(saved, tmp_path):
    """The claim both docstrings make, now checked.

    If either side starts summarising, reformatting, or truncating, this fails —
    which is the only thing standing between "two ways to reach a document" and
    "two versions of it".
    """
    store, job_id = saved
    registry = MainAgentToolRegistry(job_repository=store)

    through_agent = MainAgentRuntime._assistant_message(
        registry.invoke_atomic_tool(
            "get_saved_job", {"user_id": "u1", "job_posting_id": job_id}
        )
    )
    # The endpoint's own method, not a stand-in for it: asserting against the
    # stored row would only prove the agent path, and leave the half that was
    # actually added later unchecked.
    reader = object.__new__(WorkspaceReader)
    reader._jobs = store
    through_ui = reader.job_detail(user_id="u1", job_posting_id=job_id).jd_text

    assert through_ui == through_agent
    assert through_ui.count("\n") == JD_TEXT.count("\n")
