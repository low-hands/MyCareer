from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO

from career_agent.agent.job_discovery_contracts import JDAnalysis
from career_agent.agent.job_discovery_gateway import GatewayJobItem, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import AgentDecision, ToolCall, ToolObservation
from career_agent.agent.main_agent_runtime import MainAgentTurnResult
from career_agent.domain.resume import ResumeArtifactDelivery, ResumeArtifactReference
from career_agent.cli import EXIT_WORKFLOW_ERROR, main


class TTYBuffer(StringIO):
    def isatty(self) -> bool:
        return True


class FakeGateway:
    def __init__(self, result: JobDiscoveryGatewayResult):
        self.result = result
        self.request = None

    def start(self, request):
        self.request = request
        return self.result

    def research(self, request):
        return self.start(request)

    def select(self, *, run_id, result_ref, user_id=None):
        return self.result

    def status(self, *, run_id):
        return self.result

    def analyze_provided_jd(self, *, run_id, result_ref, jd_text, user_id=None):
        self.jd_text = jd_text
        return self.result


def result(state: str = "selection_required") -> JobDiscoveryGatewayResult:
    return JobDiscoveryGatewayResult(
        run_id="run-1",
        state=state,
        message="Select one search result to continue.",
        items=(GatewayJobItem(result_ref="r1", title="AI Engineer", company_name="Acme", city="Shanghai", salary="30-50K"),),
        next_action="select_result",
    )


def test_discover_emits_one_parseable_json_object_without_tty() -> None:
    gateway = FakeGateway(result())
    output = StringIO()

    code = main(
        ["discover", "--user-id", "u1", "--target-role", "AI Engineer", "--boss-data-dir", "/tmp/boss", "--json"],
        gateway_factory=lambda args: gateway,
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["state"] == "selection_required"
    assert payload["items"][0]["title"] == "AI Engineer"
    assert gateway.request.target_role == "AI Engineer"
    assert "\033[" not in output.getvalue()


def test_discover_uses_plain_human_output_on_tty() -> None:
    output = TTYBuffer()

    code = main(
        ["discover", "--user-id", "u1", "--target-role", "AI Engineer", "--boss-data-dir", "/tmp/boss"],
        gateway_factory=lambda args: FakeGateway(result()),
        stdout=output,
        stderr=StringIO(),
    )

    assert code == 0
    assert "State: selection_required" in output.getvalue()
    assert "1. AI Engineer — Acme" in output.getvalue()
    assert "\033[" not in output.getvalue()


def test_discover_returns_actionable_json_for_workflow_failure() -> None:
    failed = JobDiscoveryGatewayResult(
        run_id="run-2",
        state="failed",
        message="Job Discovery could not complete.",
        error_code="AGENT_WORKER_INVALID_RESPONSE",
        error_stage="candidate_triage",
        error_detail='[{"loc":["selections",0,"result_ref"],"type":"missing"}]',
    )
    output = StringIO()

    code = main(
        ["discover", "--user-id", "u1", "--target-role", "AI Engineer", "--boss-data-dir", "/tmp/boss", "--non-interactive"],
        gateway_factory=lambda args: FakeGateway(failed),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == EXIT_WORKFLOW_ERROR
    assert payload["error_code"] == "AGENT_WORKER_INVALID_RESPONSE"
    assert payload["error_stage"] == "candidate_triage"
    assert "result_ref" in payload["error_detail"]


def test_select_emits_json_for_a_persisted_run() -> None:
    output = StringIO()

    code = main(
        ["select", "--user-id", "u1", "--run-id", "run-1", "--result-ref", "boss:r1", "--boss-data-dir", "/tmp/boss", "--run-store", "/tmp/runs.sqlite3", "--json"],
        gateway_factory=lambda args: FakeGateway(result("analysis_ready")),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["state"] == "analysis_ready"


def test_analyze_jd_reads_stdin_without_boss_options() -> None:
    gateway = FakeGateway(result("analysis_ready"))
    output = StringIO()

    code = main(
        ["analyze-jd", "--user-id", "u1", "--run-id", "run-1", "--result-ref", "r1", "--jd-stdin", "--json"],
        gateway_factory=lambda args: gateway,
        stdin=StringIO("Build reliable LLM systems."),
        stdout=output,
        stderr=StringIO(),
    )

    assert code == 0
    assert gateway.jd_text == "Build reliable LLM systems."


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


def test_chat_forwards_message_to_runtime_and_emits_one_json_object() -> None:
    context = type("Context", (), {"task": None})()
    turn = MainAgentTurnResult(
        decision=AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={})),
        context=context,
        assistant_message="Select a result.",
        tool_result=JobDiscoveryGatewayResult(run_id="internal-run-do-not-leak", state="selection_required", message="Select a result.", items=(GatewayJobItem(result_ref="opaque-result-ref-do-not-leak", title="AI Engineer", company_name="Acme"),)),
    )
    runtime = Runtime(turn)
    output = StringIO()

    code = main(
        ["chat", "--user-id", "u1", "--session-id", "s1", "--message", "Find work", "--boss-data-dir", "/tmp/boss"],
        runtime_factory=lambda args: runtime,
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert runtime.calls == [("u1", "s1", "Find work")]
    assert runtime.closed is True
    assert payload["assistant_message"] == "Select a result."
    assert payload["decision"] == {"action": "tool_call", "tool_name": "job_discovery"}
    assert payload["tool_result"]["state"] == "selection_required"
    assert payload["tool_result"]["items"] == [{"selection_index": 1, "title": "AI Engineer", "company_name": "Acme", "city": None, "salary": None, "rationale": None, "cautions": []}]
    assert "internal-run-do-not-leak" not in output.getvalue()
    assert "opaque-result-ref-do-not-leak" not in output.getvalue()


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
            "--boss-data-dir",
            "/tmp/boss",
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
            "--boss-data-dir",
            "/tmp/boss",
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


def test_chat_keeps_structured_analysis_with_human_summary() -> None:
    analysis = JDAnalysis(result_ref="r1", job_summary="LLM 应用落地。", responsibilities=("建设 LLM 应用",), required_skills=("Python",), preferred_qualifications=("RAG 经验",), clarification_questions=("经验年限不明确",))
    turn = MainAgentTurnResult(
        decision=AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={})),
        context=type("Context", (), {"task": None})(),
        assistant_message="岗位摘要\nLLM 应用落地。\n\n工作职责\n- 建设 LLM 应用\n\n必备技能\n- Python\n\n加分项\n- RAG 经验\n\n待确认问题\n- 经验年限不明确",
        tool_result=JobDiscoveryGatewayResult(run_id="internal-run-do-not-leak", state="analysis_ready", message="Analysis ready.", analysis=analysis),
    )
    output = StringIO()

    code = main(
        ["chat", "--user-id", "u1", "--session-id", "s1", "--message", "看第一个", "--boss-data-dir", "/tmp/boss"],
        runtime_factory=lambda args: Runtime(turn),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["assistant_message"] == turn.assistant_message
    assert payload["tool_result"]["analysis"] == {"job_summary": "LLM 应用落地。", "responsibilities": ["建设 LLM 应用"], "required_skills": ["Python"], "preferred_qualifications": ["RAG 经验"], "clarification_questions": ["经验年限不明确"]}
    assert "internal-run-do-not-leak" not in output.getvalue()

    output = StringIO()

    code = main(
        ["chat", "--user-id", "u1", "--session-id", "s1", "--message", "Find work", "--boss-data-dir", "/tmp/boss"],
        runtime_factory=lambda args: (_ for _ in ()).throw(Exception("boom")),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 6
    assert payload["state"] == "failed"
    assert payload["error_code"] == "CHAT_ERROR"
