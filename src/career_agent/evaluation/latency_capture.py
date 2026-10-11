"""Capture live production-runtime latency in an isolated seeded workspace.

Unlike runtime_quality, this uses real local business stores and specialist
workers. The authored source materials are intentionally long and public in
this repository. No user account, mailbox or calendar is connected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter, sleep

FIXTURE = Path(__file__).resolve().parents[3] / "evals/latency/scenarios.json"


def capture(
    root: Path, *, conversations: int, fixture: Path = FIXTURE,
    inter_turn_delay_seconds: float = 0.0,
) -> dict:
    if (
        not math.isfinite(inter_turn_delay_seconds)
        or not 0 <= inter_turn_delay_seconds <= 60
    ):
        raise ValueError("inter-turn delay must be between 0 and 60 seconds")
    from dotenv import load_dotenv
    load_dotenv()
    from career_agent.cli import build_parser, build_main_agent_runtime
    from career_agent.agent.context.deployment import ContextDeploymentConfig, ConversationSummaryAgentConfig
    from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
    from career_agent.agent.providers.conversation_summary import OpenAIConversationSummaryWorker
    from career_agent.agent.contracts.profile import CareerProfileContext
    from career_agent.agent.contracts.resources import ConversationMessageContext
    from career_agent.agent.contracts.task_state import ConversationTaskState
    from career_agent.api.single_worker import SingleWorkerLock, lock_path_for
    from career_agent.domain.job_discovery import JobDetail, Provenance
    from career_agent.storage.context import CareerContextStore
    from career_agent.storage.jobs import SQLiteJobPostingRepository
    from career_agent.storage.resumes import ResumeStore, StoredResumeText

    # A fresh directory prevents stale traces and protects existing workspaces.
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    args = build_parser().parse_args(["chat", "--user-id", "latency-user", "--session-id", "unused", "--message", "unused"])
    for name in vars(args):
        if name.endswith("_store"):
            setattr(args, name, str(root / (name.replace("_", "-") + ".sqlite3")))
    args.run_events_store = str(root / "run-events.sqlite3")
    now = datetime.now(timezone.utc)
    resumes = ResumeStore(Path(args.resume_store))
    target = resumes.create_target_role(user_id="latency-user", title="AI 应用后端工程师", priority=1)
    _, version = resumes.import_document(user_id="latency-user", content=payload["resume"].encode(), document_format="markdown", name="陈明简历", target_role_id=target.id)
    resumes.save_version_text(user_id="latency-user", resume_version_id=version.id,
                              text=StoredResumeText(pages=(payload["resume"],), method="plain"))
    jobs = SQLiteJobPostingRepository(Path(args.job_store))
    jobs.save_captured_detail(user_id="latency-user", detail=JobDetail(
        source_name="latency-fixture", title="AI 应用后端工程师", company_name="示例云科技",
        description=payload["jd"], city="上海", captured_at=now, content_origin="user_provided",
        provenance=Provenance(source_name="latency-fixture", captured_at=now, operation="seed", adapter_version="1")))
    store = CareerContextStore(Path(args.context_store))
    store.upsert_profile(CareerProfileContext(user_id="latency-user"))
    summary_config = ConversationSummaryAgentConfig.from_env(
        main_config=OpenAICompatibleAgentConfig.from_env(prefix="MAIN_AGENT"),
        main_context_window_tokens=ContextDeploymentConfig.from_env().main_context_window_tokens,
    )
    report = {"version": 1, "measurement": payload["measurement"], "started_at": now.isoformat(),
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py")))).hexdigest(),
              "models": {k: os.environ.get(k) for k in ("MAIN_AGENT_MODEL", "RESUME_ANALYSIS_AGENT_MODEL", "CAREER_EMBEDDING_MODEL", "CONVERSATION_SUMMARY_AGENT_MODEL")},
              "summary_prompt_sha256": hashlib.sha256(OpenAIConversationSummaryWorker._budgeted_system_prompt(summary_config.max_output_tokens).encode()).hexdigest(),
              "summary_config": {"model": summary_config.provider.model,
                                 "disable_thinking": summary_config.disable_thinking,
                                 "timeout_seconds": summary_config.provider.timeout_seconds,
                                 "max_output_tokens": summary_config.max_output_tokens},
              "history_turns": len(payload["history"]), "conversations": conversations,
              "inter_turn_delay_seconds": inter_turn_delay_seconds, "turns": []}
    # Keep the manifest durable if a process is interrupted mid-capture.
    def save():
        (root / "capture.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    with SingleWorkerLock(lock_path_for(root)):
        runtime = build_main_agent_runtime(args)
        try:
            for number in range(conversations):
                conversation_id = f"latency-{number + 1}"
                for user, assistant in payload["history"]:
                    store.commit_turn(user_id="latency-user", conversation_id=conversation_id,
                                      task=ConversationTaskState(),
                                      user_message=ConversationMessageContext(role="user", content=user.format(resume=payload["resume"], jd=payload["jd"]), created_at=now),
                                      assistant_message=ConversationMessageContext(role="assistant", content=assistant, created_at=now))
                for index, message in enumerate(payload["turns"]):
                    gap_ms = 0
                    if index and inter_turn_delay_seconds:
                        gap_started = perf_counter()
                        sleep(inter_turn_delay_seconds)
                        gap_ms = round((perf_counter() - gap_started) * 1000)
                    started = perf_counter()
                    row = {"conversation": number + 1, "turn": index + 1,
                           "inter_turn_gap_ms": gap_ms}
                    try:
                        result = runtime.run_turn(user_id="latency-user", conversation_id=conversation_id, user_message=message)
                        row["status"] = "returned"
                        row["interrupted"] = bool(result.context.task.pending_interaction)
                    except Exception as error:
                        row.update(status="failed", error_code=getattr(error, "code", type(error).__name__))
                    row["wall_ms"] = round((perf_counter() - started) * 1000)
                    report["turns"].append(row)
                    save()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    # Owner input cannot be replaced by later benchmark prompts.
                    if row.get("interrupted") or row["status"] == "failed":
                        break
        finally:
            runtime.close()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--conversations", type=int, default=3)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--inter-turn-delay-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not 1 <= args.conversations <= 20:
        parser.error("conversations must be between 1 and 20")
    capture(args.work_root, conversations=args.conversations, fixture=args.fixture,
            inter_turn_delay_seconds=args.inter_turn_delay_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
