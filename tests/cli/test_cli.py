from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO

from career_agent.agent.main_agent_contracts import AgentDecision, ToolCall, ToolObservation
from career_agent.agent.main_agent_runtime import MainAgentTurnResult
from career_agent.domain.resume import ResumeArtifactDelivery, ResumeArtifactReference
from career_agent.cli import EXIT_ARGUMENT_ERROR, EXIT_WORKFLOW_ERROR, main


class TTYBuffer(StringIO):
    def isatty(self) -> bool:
        return True


class Runtime:
    def __init__(self, turn):
        self.turn = turn
        self.calls = []
        self.closed = False

    def run_turn(self, *, user_id, conversation_id, user_message):
        self.calls.append((user_id, conversation_id, user_message))
        return self.turn

    def close(self):
        self.closed = True


def test_trajectory_cli_reports_quality_as_an_independent_axis() -> None:
    output = StringIO()

    code = main(
        [
            "eval",
            "trajectories",
            "--scenario",
            "a_report_that_scrolled_out_of_the_catalogue_is_not_faked",
        ],
        stdout=output,
        stderr=StringIO(),
    )
    payload = json.loads(output.getvalue())
    result = payload["results"][0]

    assert code == 0
    assert payload["behaviour_failed"] == 0
    assert payload["quality_failed"] == 0
    assert result["quality_status"] == "passed"
    assert result["quality_samples_passed"] == 5
    assert result["quality_sample_count"] == 5
    assert result["quality_pass_rate"] == 1.0
    assert result["quality_min_pass_rate"] == 0.6
    assert result["quality_wilson_95"] == [0.565518, 1.0]


def test_trajectory_cli_keeps_intermittent_hard_gaps_red() -> None:
    output = StringIO()

    code = main(
        ["eval", "trajectories"],
        stdout=output,
        stderr=StringIO(),
    )
    payload = json.loads(output.getvalue())
    intermittent = {
        result["scenario"]
        for result in payload["results"]
        if result["known_gap_status"] == "intermittent"
    }

    assert code == EXIT_ARGUMENT_ERROR
    assert payload["behaviour_failed"] == 2
    assert intermittent == {
        "a_report_made_this_turn_without_an_index_cannot_be_named",
        "an_uncertain_calendar_write_is_not_reissued_or_claimed",
    }


def test_chat_publishes_no_decision_for_a_turn_the_model_never_decided() -> None:
    """The JSON must not claim a call that never happened.

    ``decision.tool_name`` is machine-readable output. For the two ingresses
    that fabricate an ``AgentDecision`` — the bound interaction receipt and the
    mock interview workflow continuation — publishing those values, in a form
    another program would act on, that the model chose a capability it was
    never even consulted about. What did happen is in ``tool_results``.
    """
    context = type("Context", (), {"task": None})()
    turn = MainAgentTurnResult(
        decision_source="runtime",
        decision=AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="handle_mock_interview_input", arguments={}),
        ),
        context=context,
        assistant_message="下一题。",
        tool_result=ToolObservation(
            tool_name="handle_mock_interview_input",
            state="mock_interview_running",
            message="下一题。",
        ),
    )
    output = StringIO()

    code = main(
        ["chat", "--user-id", "u1", "--session-id", "s1", "--message", "我的回答"],
        runtime_factory=lambda args: Runtime(turn),
        stdout=output,
        stderr=StringIO(),
    )
    payload = json.loads(output.getvalue())

    assert code == 0
    assert payload["decision"] == {
        "source": "runtime",
        "action": None,
        "tool_name": None,
    }
    # The turn is still fully reported — through the results, which are real.
    assert payload["tool_result"]["state"] == "mock_interview_running"


def test_chat_forwards_message_to_runtime_and_emits_one_json_object() -> None:
    context = type("Context", (), {"task": None})()
    turn = MainAgentTurnResult(
        decision_source="model",
        decision=AgentDecision(action="tool_call", tool_call=ToolCall(name="open_job_search", arguments={"keyword": "AI Engineer"})),
        context=context,
        assistant_message="已准备打开 BOSS 搜索“AI Engineer”。",
        tool_result=ToolObservation(
            tool_name="open_job_search",
            state="job_search_page_ready",
            message="已准备打开 BOSS 搜索“AI Engineer”。",
            next_action="browse_and_save_job",
            payload={
                "platform": "boss",
                "keyword": "AI Engineer",
                "city": None,
                "client_action": {
                    "type": "open_url",
                    "url": "https://www.zhipin.com/web/geek/job?query=AI+Engineer",
                    "label": "在 BOSS 搜索 AI Engineer",
                },
            },
        ),
    )
    runtime = Runtime(turn)
    output = StringIO()

    code = main(
        ["chat", "--user-id", "u1", "--session-id", "s1", "--message", "Find work"],
        runtime_factory=lambda args: runtime,
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert runtime.calls == [("u1", "s1", "Find work")]
    assert runtime.closed is True
    assert payload["assistant_message"] == "已准备打开 BOSS 搜索“AI Engineer”。"
    assert payload["decision"] == {
        "source": "model",
        "action": "tool_call",
        "tool_name": "open_job_search",
    }
    assert payload["tool_result"]["state"] == "job_search_page_ready"
    assert payload["tool_result"]["payload"]["client_action"]["url"] == (
        "https://www.zhipin.com/web/geek/job?query=AI+Engineer"
    )


def test_chat_emits_artifact_metadata_without_attachment_bytes() -> None:
    context = type("Context", (), {"task": None})()
    reference = ResumeArtifactReference(
        id="resume_artifact_1",
        user_id="u1",
        resume_version_id="resume_version_1",
        filename="AI Resume-v2.md",
        media_type="text/markdown; charset=utf-8",
        byte_size=22,
        created_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    turn = MainAgentTurnResult(
        decision_source="model",
        decision=AgentDecision(action="final", message="文件已准备好。"),
        context=context,
        assistant_message="文件已准备好。",
        tool_result=ToolObservation(
            tool_name="export_resume_artifact",
            state="resume_artifact_ready",
            message="文件已准备好。",
            payload={"artifact_id": reference.id},
        ),
        artifacts=(
            ResumeArtifactDelivery(
                reference=reference,
                content=b"PRIVATE RESUME BYTES!",
            ),
        ),
    )
    output = StringIO()

    code = main(
        [
            "chat",
            "--user-id",
            "u1",
            "--session-id",
            "s1",
            "--message",
            "下载简历",
        ],
        runtime_factory=lambda args: Runtime(turn),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["artifacts"][0]["id"] == reference.id
    assert payload["artifacts"][0]["filename"] == reference.filename
    assert "PRIVATE RESUME BYTES" not in output.getvalue()
    assert "content" not in payload["artifacts"][0]


def test_chat_emits_all_tool_results_in_execution_order() -> None:
    first = ToolObservation(
        tool_name="list_resumes",
        state="resumes_found",
        message="找到一份简历。",
        payload={"items": [{"selection_index": 1, "name": "AI Resume"}]},
    )
    second = ToolObservation(
        tool_name="get_resume_metadata",
        state="resume_metadata_ready",
        message="已读取简历元数据。",
        payload={"versions": [{"selection_index": 1, "version_number": 2}]},
    )
    turn = MainAgentTurnResult(
        decision_source="model",
        decision=AgentDecision(action="final", message=None),
        context=type("Context", (), {"task": None})(),
        assistant_message=second.message,
        tool_result=second,
        tool_results=(first, second),
    )
    output = StringIO()

    code = main(
        [
            "chat",
            "--user-id",
            "u1",
            "--session-id",
            "s1",
            "--message",
            "查看简历",
        ],
        runtime_factory=lambda args: Runtime(turn),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert [item["tool_name"] for item in payload["tool_results"]] == [
        "list_resumes",
        "get_resume_metadata",
    ]
    assert payload["tool_result"] == payload["tool_results"][-1]


def test_chat_reports_a_runtime_failure_as_one_json_object() -> None:
    output = StringIO()

    code = main(
        ["chat", "--user-id", "u1", "--session-id", "s1", "--message", "Find work"],
        runtime_factory=lambda args: (_ for _ in ()).throw(Exception("boom")),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 6
    assert payload["state"] == "failed"
    assert payload["error_code"] == "CHAT_ERROR"
