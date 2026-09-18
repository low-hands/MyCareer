from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from compaction_smoke_093 import FactSummaryWorker, MeteredSummaryWorker, run_trajectory
from career_agent.agent.context_deployment_config import (
    ContextDeploymentConfig,
    ConversationSummaryAgentConfig,
    validate_model_window,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
    SummaryMessage,
)
from career_agent.agent.main_agent_contracts import (
    DECISION_OBSERVATION_BODY_LIMIT,
    ConversationTaskState,
    DecisionObservation,
    MainAgentContext,
)
from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)
from career_agent.agent.token_budget import message_token_count
from career_agent.storage.context import CareerContextStore


def commit(manager: ContextManager, text: str = "synthetic") -> MainAgentContext:
    context = manager.load_for_turn(user_id="u", conversation_id="c", user_message=text)
    manager.commit_turn(
        context=context, task=ConversationTaskState(), assistant_message="acknowledged"
    )
    return context


def test_huge_first_message_compacts_committed_exchange_without_waiting_for_eight_rows(
    tmp_path: Path,
) -> None:
    worker = MeteredSummaryWorker(FactSummaryWorker())
    store = CareerContextStore(tmp_path / "early.sqlite3")
    manager = ContextManager(store, summary_worker=worker)
    manager.configure_request_token_estimator(
        lambda context: (22000 + message_token_count(context.user_message), 32000),
        static_input_tokens=15000,
        max_input_tokens=32000,
    )
    text = "long " * 3200
    context = manager.load_for_turn(user_id="u", conversation_id="c", user_message=text)
    assert worker.sequences == []  # The current input is not historical source yet.
    assert context.user_message == text
    manager.commit_turn(
        context=context, task=ConversationTaskState(), assistant_message="acknowledged"
    )
    assert worker.sequences == [(1, 2)]
    summary = store.get_conversation_summary(user_id="u", conversation_id="c")
    assert summary is not None and summary.through_sequence == 2
    # Source clipping for the summary does not rewrite the durable transcript.
    assert store.list_message_records("u", "c", limit=2)[0].message.content == text
    final = manager.load_for_turn(user_id="u", conversation_id="c", user_message="next")
    assert final.through_sequence == 2 and final.recent_messages == ()
    assert worker.sequences == [(1, 2)]


@pytest.mark.parametrize(
    "occupancy,expected", [(0.749, []), (0.75, [(1, 2)]), (1.1, [(1, 2)])]
)
def test_partial_prefix_requires_measured_complete_request_pressure(
    tmp_path: Path,
    occupancy: float,
    expected: list[tuple[int, ...]],
) -> None:
    store = CareerContextStore(tmp_path / "threshold.sqlite3")
    commit(ContextManager(store))
    worker = MeteredSummaryWorker(FactSummaryWorker())
    manager = ContextManager(store, summary_worker=worker)
    manager.configure_request_token_estimator(
        lambda context: (int(occupancy * 32000), 32000)
    )
    manager.load_for_turn(user_id="u", conversation_id="c", user_message="next")
    assert worker.sequences == expected


class RecoveringSummaryWorker:
    def __init__(self) -> None:
        self.sequences: list[tuple[int, ...]] = []

    def summarize(
        self,
        *,
        previous: ConversationSummaryContent | None,
        messages: tuple[SummaryMessage, ...],
    ) -> ConversationSummaryContent:
        self.sequences.append(tuple(message.sequence for message in messages))
        if len(self.sequences) <= 3:
            raise AgentWorkerError(
                "CONVERSATION_SUMMARY_REJECTED_503", "Unavailable", retryable=True
            )
        return FactSummaryWorker().summarize(previous=previous, messages=messages)


def test_default_batch_backoff_recovers_same_prefix_without_watermark_skips(
    tmp_path: Path,
) -> None:
    store = CareerContextStore(tmp_path / "recovery.sqlite3")
    writer = ContextManager(store)
    for index in range(15):
        commit(writer, f"synthetic-{index}")
    clock = [datetime(2026, 9, 17, tzinfo=timezone.utc)]
    worker = RecoveringSummaryWorker()
    manager = ContextManager(store, summary_worker=worker, clock=lambda: clock[0])
    manager.configure_request_token_estimator(lambda context: (16000, 32000))
    for _ in range(6):
        manager.load_for_turn(user_id="u", conversation_id="c", user_message="next")
    assert worker.sequences == [tuple(range(1, 9))] * 3
    assert store.get_conversation_summary(user_id="u", conversation_id="c") is None
    # During backoff the bounded recent projection may have a named gap, but
    # it is not a false summary watermark or deleted history: exact page-in works.
    span = store.read_conversation_span(
        user_id="u", conversation_id="c", from_sequence=1, through_sequence=8
    )
    assert [message.sequence for message in span.messages] == list(range(1, 9))
    assert span.messages[0].content == "synthetic-0"
    assert (
        store.read_conversation_span(
            user_id="other-owner",
            conversation_id="c",
            from_sequence=1,
            through_sequence=8,
        ).messages
        == ()
    )
    clock[0] += timedelta(minutes=9, seconds=59)
    manager.load_for_turn(user_id="u", conversation_id="c", user_message="next")
    assert len(worker.sequences) == 3
    clock[0] += timedelta(seconds=1)
    recovered = manager.load_for_turn(
        user_id="u", conversation_id="c", user_message="next"
    )
    assert worker.sequences == [tuple(range(1, 9))] * 4
    assert recovered.through_sequence == 8
    assert recovered.recent_from_sequence == 9
    assert len(recovered.recent_messages) == 22
    assert ("u", "c") not in manager._compaction_failures
    manager.load_for_turn(user_id="u", conversation_id="c", user_message="next")
    assert len(worker.sequences) == 4
    assert len(store.list_message_records("u", "c", limit=30)) == 30


@pytest.mark.parametrize("value", [0, -1, True, 1024.5])
def test_declared_window_rejects_non_positive_or_non_integral_budgets(
    value: int,
) -> None:
    with pytest.raises(AgentConfigurationError):
        validate_model_window(
            input_tokens=value,
            output_tokens=1200,
            context_window_tokens=65536,
            prefix="TEST",
        )


@pytest.mark.parametrize("value", [True, 16.5])
def test_context_direct_construction_rejects_non_integral_counts(value: int) -> None:
    with pytest.raises(ValueError):
        ContextDeploymentConfig(recent_message_limit=value)
    with pytest.raises(ValueError):
        ContextDeploymentConfig(summary_batch_size=value)


@pytest.mark.parametrize(
    "input_tokens,output_tokens,window,timeout",
    [
        (2_000_001, 1200, 2_000_000, 30),
        (32000, 1200.5, 65536, 30),
        (32000, 1200, 2_000_001, 30),
        (32000, 1200, 65536.5, 30),
        (32000, 1200, 65536, float("nan")),
        (32000, 1200, 65536, 121),
    ],
)
def test_summary_direct_construction_has_same_bounded_policy(
    input_tokens: int,
    output_tokens: int,
    window: int,
    timeout: float,
) -> None:
    provider = OpenAICompatibleAgentConfig(
        endpoint="https://synthetic.test/v1/chat/completions",
        api_key="synthetic-key",
        model="synthetic",
        max_input_tokens=input_tokens,
        timeout_seconds=timeout,
    )
    with pytest.raises(ValueError):
        ConversationSummaryAgentConfig(
            provider=provider,
            max_output_tokens=output_tokens,
            context_window_tokens=window,
        )


def test_invalid_summary_numeric_env_value_is_not_chained_into_logs() -> None:
    provider = OpenAICompatibleAgentConfig(
        endpoint="https://synthetic.test/v1/chat/completions",
        api_key="synthetic-key",
        model="synthetic",
    )
    with pytest.raises(AgentConfigurationError) as raised:
        ConversationSummaryAgentConfig.from_env(
            main_config=provider,
            environ={
                "CONVERSATION_SUMMARY_AGENT_BASE_URL": "https://synthetic.test/v1",
                "CONVERSATION_SUMMARY_AGENT_API_KEY": "synthetic-key",
                "CONVERSATION_SUMMARY_AGENT_MODEL": "synthetic",
                "CONVERSATION_SUMMARY_AGENT_MAX_INPUT_TOKENS": "synthetic-private-value",
            },
        )
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    assert "synthetic-private-value" not in str(raised.value)


def test_absent_summary_namespace_does_not_hide_invalid_inherited_window() -> None:
    provider = OpenAICompatibleAgentConfig(
        endpoint="https://synthetic.test/v1/chat/completions",
        api_key="synthetic-key",
        model="synthetic",
    )
    with pytest.raises(AgentConfigurationError, match="exceed"):
        ConversationSummaryAgentConfig.from_env(
            main_config=replace(provider, max_input_tokens=64000),
            main_context_window_tokens=65000,
            environ={},
        )


@pytest.mark.parametrize("recent,batch", [(8, 4), (16, 8), (64, 32)])
@pytest.mark.parametrize("full_observation", [False, True])
def test_production_request_estimator_preserves_reserves_and_measures_observation(
    tmp_path: Path,
    recent: int,
    batch: int,
    full_observation: bool,
) -> None:
    store = CareerContextStore(tmp_path / "request.sqlite3")
    writer = ContextManager(store)
    for index in range(12):
        commit(writer, f"合成记录{index}：" + "中文有界历史消息。" * 1000)
    manager = ContextManager(
        store, recent_message_limit=recent, summary_batch_size=batch
    )
    specs: tuple[dict[str, object], ...] = tuple(
        {
            "type": "function",
            "function": {
                "name": f"synthetic_tool_{index}",
                "description": "Synthetic read-only tool description. " * 12,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(69)
    )
    config = OpenAICompatibleAgentConfig(
        endpoint="https://synthetic.test/v1/chat/completions",
        api_key="synthetic-key",
        model="synthetic",
    )

    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError("budget estimation must not send a model request")

    with OpenAI(
        api_key="synthetic-key",
        base_url="https://synthetic.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(unexpected)),
    ) as client:
        maker = OpenAICompatibleMainAgentDecisionMaker(
            config, client=client, max_output_tokens=16384
        )
        static, limit = maker.static_request_token_usage(specs)
        manager.configure_request_token_estimator(
            lambda context: maker.request_token_usage(context, specs),
            static_input_tokens=static,
            max_input_tokens=limit,
        )
        context = manager.load_for_turn(
            user_id="u", conversation_id="c", user_message="新合成请求。" * 3000
        )
        body_chars = DECISION_OBSERVATION_BODY_LIMIT if full_observation else 3000
        body = "观察" * (body_chars // 2)
        observation = DecisionObservation(
            tool_name="synthetic_tool_0",
            state="ready",
            message="合成观察已就绪",
            body=body,
        )
        observed = context.model_copy(update={"tool_observations": (observation,)})
        before, _ = maker.request_token_usage(context, specs)
        after, _ = maker.request_token_usage(observed, specs)
    assert after > before > static
    assert before <= limit == 32000
    if full_observation:
        # Existing character-bounded observations are not token-bounded. This
        # maximum CJK body exceeds the input target under 8/4 as well as 16/8;
        # increasing the recent-row count must not hide that measured pressure.
        assert after > limit
    else:
        assert after <= limit
    assert (
        after + 16384 <= 65536
    )  # Declared window, not an inferred provider capability.
    assert observed.tool_observations[0].body == body
    assert observed.user_message == context.user_message
    assert (
        sum(message_token_count(message.content) for message in context.recent_messages)
        <= manager._recent_context_tokens
    )
    assert manager._recent_context_tokens == int((limit - static) * 0.4)
    assert manager._user_message_tokens == int((limit - static) * 0.2)


def test_smoke_failure_counts_attempts_without_claiming_history_was_deleted(
    tmp_path: Path,
) -> None:
    result = run_trajectory(
        tmp_path / "failed.sqlite3",
        recent=16,
        batch=8,
        worker=RecoveringSummaryWorker(),
    )
    assert result.compaction_attempts == 3
    assert result.compaction_calls == 0
    assert result.summary_failure_codes == ("CONVERSATION_SUMMARY_REJECTED_503",) * 3
    assert result.history_reachable
    assert not result.projection_contiguous
    assert not result.summary_available
    assert result.watermark == 0
    assert result.duplicate_summary_calls == 2
    assert result.page_in_calls == 5
    assert not result.passed
