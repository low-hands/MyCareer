import json
from datetime import datetime, timezone

import pytest

from career_agent.agent.decision_messages import (
    CONTROL_CONTEXT_LABEL,
    CONTROL_REMINDER_TAG,
    DATA_CONTEXT_LABEL,
    STABLE_DATA_CONTEXT_LABEL,
    TURN_OBSERVATION_LABEL,
    project_decision_messages,
)
from career_agent.agent.main_agent_contracts import CandidateContextItem, CareerProfileContext, ConversationMessageContext, ConversationResourceReference, ConversationTaskState, CurrentTargetContext, DecisionObservation, MainAgentContext, OpenJobSearchToolArguments
from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    AgentWorkerError,
)
from career_agent.agent.openai_compatible_main_agent import OpenAICompatibleMainAgentDecisionMaker


def _spotlight_json(content: str, *, label: str) -> dict:
    lines = content.splitlines()
    assert lines[0] == label
    assert lines[1].startswith('<untrusted-data nonce="')
    assert lines[-1].startswith('</untrusted-data nonce="')
    assert lines[1][1:] == lines[-1][2:]
    return json.loads("\n".join(lines[2:-1]))


def _control_json(content: str) -> dict:
    lines = content.splitlines()
    assert lines[0] == f"<{CONTROL_REMINDER_TAG}>"
    assert lines[1] == CONTROL_CONTEXT_LABEL
    assert lines[-1] == f"</{CONTROL_REMINDER_TAG}>"
    return json.loads("\n".join(lines[2:-1]))


class Completions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": json.dumps({"action": "ask_user", "message": "Which city should I search?"})})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class Client:
    def __init__(self) -> None:
        self.completions = Completions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_prompt_cache_policy_is_loaded_explicitly_from_environment() -> None:
    config = OpenAICompatibleAgentConfig.from_env(
        environ={
            "MAIN_AGENT_BASE_URL": "https://compatible.example.test/v1",
            "MAIN_AGENT_API_KEY": "test",
            "MAIN_AGENT_MODEL": "provider-model-alias",
            "MAIN_AGENT_PROMPT_CACHE": "explicit",
        },
        prefix="MAIN_AGENT",
    )

    assert config.prompt_cache == "explicit"


def test_prompt_cache_defaults_to_implicit() -> None:
    config = OpenAICompatibleAgentConfig.from_env(
        environ={
            "MAIN_AGENT_BASE_URL": "https://compatible.example.test/v1",
            "MAIN_AGENT_API_KEY": "test",
            "MAIN_AGENT_MODEL": "provider-model-alias",
        },
        prefix="MAIN_AGENT",
    )

    assert config.prompt_cache == "implicit"


def test_invalid_prompt_cache_policy_fails_configuration() -> None:
    with pytest.raises(AgentConfigurationError, match="PROMPT_CACHE"):
        OpenAICompatibleAgentConfig.from_env(
            environ={
                "MAIN_AGENT_BASE_URL": "https://compatible.example.test/v1",
                "MAIN_AGENT_API_KEY": "test",
                "MAIN_AGENT_MODEL": "provider-model-alias",
                "MAIN_AGENT_PROMPT_CACHE": "auto-detect",
            },
            prefix="MAIN_AGENT",
        )


def test_main_agent_decision_maker_separates_control_data_and_native_chat() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"),
        client=client,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1", default_city="Shanghai"),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="internal-run-do-not-leak",
            selected_result_ref="opaque-selected-ref-do-not-leak",
            candidates=(CandidateContextItem(result_ref="opaque-candidate-ref-do-not-leak", title="AI Engineer", company_name="Acme", city="Shanghai"),),
        ),
        conversation_summary=ConversationSummaryContent(
            user_goals=("Find an AI Engineer role",),
            confirmed_decisions=("Use the current resume",),
            active_constraints=("Do not apply automatically",),
        ),
        recent_messages=(
            ConversationMessageContext(
                role="user",
                content="Earlier user words.",
                created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
            ),
            ConversationMessageContext(
                role="assistant",
                content="Earlier assistant words.",
                created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
                resource_refs=(
                    ConversationResourceReference(
                        kind="job_research_report",
                        resource_id="report-1",
                        status_at_delivery="current",
                        anchored_by_other_job=False,
                        title="External report title",
                    ),
                ),
            ),
        ),
        user_message="Help me find work.",
    )

    decision = maker.decide(context, ({"type": "function", "function": {"name": "open_job_search", "description": "test", "parameters": OpenJobSearchToolArguments.model_json_schema()}},))

    messages = client.completions.kwargs["messages"]
    system_content = messages[0]["content"]
    stable_content = messages[1]["content"]
    stable_payload = _spotlight_json(
        stable_content,
        label=STABLE_DATA_CONTEXT_LABEL,
    )
    control_content = messages[4]["content"]
    control = _control_json(control_content)
    data_content = messages[5]["content"]
    assert data_content.startswith(DATA_CONTEXT_LABEL + "\n")
    payload = _spotlight_json(data_content, label=DATA_CONTEXT_LABEL)
    assert decision.action == "ask_user"
    # Profile facts are complete deterministic Markdown files in the stable
    # prefix. Query-sensitive evidence alone occupies career_memory.
    profile_files = stable_payload["career_profile"]
    assert '- Default city: "Shanghai"' in profile_files["memory/profile.md"]
    assert payload["career_memory"] == {}
    assert not {
        "salary_expectation",
        "experience",
        "education",
    } & payload["career_memory"].keys()
    assert "resume_text" not in payload
    assert "open_job_search" not in system_content
    assert (
        "single <system-reminder> after native prior turns"
    ) in system_content
    assert CONTROL_CONTEXT_LABEL not in system_content
    assert "mock_interview" not in system_content
    assert control["task"]["active_workflow"] == "mock_interview"
    assert "candidates" not in control["task"]
    assert "Acme" not in system_content
    assert "External report title" not in system_content
    assert "Do not apply automatically" not in system_content
    assert client.completions.kwargs["tools"][0]["function"]["name"] == "open_job_search"
    assert set(client.completions.kwargs["tools"][0]["function"]["parameters"]["properties"]) == {"platform", "keyword", "city"}
    raw_context = data_content
    assert "internal-run-do-not-leak" not in raw_context
    assert "opaque-selected-ref-do-not-leak" not in raw_context
    assert "opaque-candidate-ref-do-not-leak" not in raw_context
    assert payload["task"]["candidates"][0]["selection_index"] == 1
    assert control["preferences"]["boss_search"] == "explicit_request_only"
    assert stable_payload["conversation_summary"]["confirmed_decisions"] == [
        "Use the current resume"
    ]
    assert (
        stable_payload["conversation_summary"]["omitted_active_constraint_count"] == 0
    )
    assert "tool_observations" not in control
    assert "tool_observations" not in payload
    assert payload["recent_resources"][0]["title"] == "External report title"
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
        "user",
        "user",
    ]
    assert messages[2]["content"] == "Earlier user words."
    assert messages[3]["content"].startswith("Earlier assistant words.\n\n")
    assert "[runtime resources: report_" in messages[3]["content"]
    assert "job_research_report]" in messages[3]["content"]
    assert "External report title" not in messages[3]["content"]
    assert messages[-1] == {"role": "user", "content": "Help me find work."}


def test_role_scoped_intent_stays_separate_in_current_targets() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            default_city="杭州",
            current_targets=(
                CurrentTargetContext(
                    title="ML Engineer",
                    priority=1,
                    salary_expectation="40-50k",
                    experience="5-7 years",
                ),
                CurrentTargetContext(
                    title="Product Manager",
                    priority=2,
                    city="上海",
                    salary_expectation="30-40k",
                    education="本科",
                ),
            ),
        ),
        user_message="比较我的两个方向",
    )

    career_profile = context.model_context()["career_profile"]
    profile_file = career_profile["memory/profile.md"]
    targets_file = career_profile["memory/current_targets.md"]

    assert '- Default city: "杭州"' in profile_file
    assert targets_file.index('Title: "ML Engineer"') < targets_file.index(
        'Title: "Product Manager"'
    )
    assert '- Salary expectation: "40-50k"' in targets_file
    assert '- Experience: "5-7 years"' in targets_file
    assert '- City: "上海"' in targets_file
    assert '- Salary expectation: "30-40k"' in targets_file
    assert '- Education: "本科"' in targets_file


def test_untrusted_data_uses_a_session_stable_matching_spotlight_nonce() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )
    projection = project_decision_messages(context)
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=Client(),
    )
    nonce = maker._spotlight_nonce(context)

    first = projection.messages(
        system_prompt="policy", spotlight_nonce=nonce
    )[1]["content"]
    second = projection.messages(
        system_prompt="policy", spotlight_nonce=maker._spotlight_nonce(context)
    )[1]["content"]

    assert _spotlight_json(
        first, label=STABLE_DATA_CONTEXT_LABEL
    ) == _spotlight_json(
        second,
        label=STABLE_DATA_CONTEXT_LABEL,
    )
    assert first.splitlines()[1] == second.splitlines()[1]
    other = context.model_copy(update={"conversation_id": "c2"})
    assert maker._spotlight_nonce(other) != nonce


def test_request_token_usage_counts_tools_and_weights_cjk() -> None:
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
            max_input_tokens=4096,
        ),
        client=Client(),
    )
    english = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="a" * 400,
    )
    chinese = english.model_copy(update={"user_message": "中" * 400})
    small, limit = maker.request_token_usage(english, ())
    with_tool, _ = maker.request_token_usage(
        english,
        (
            {
                "type": "function",
                "function": {
                    "name": "large_tool",
                    "description": "d" * 4000,
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )
    cjk, _ = maker.request_token_usage(chinese, ())

    assert limit == 4096
    assert with_tool > small + 900
    assert cjk > small + 150


def test_configured_implicit_cache_uses_a_stable_key_without_endpoint_sniffing() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://compatible.example.test/v1/chat/completions",
            api_key="test",
            model="renamed-model",
            prompt_cache="implicit",
        ),
        client=client,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="first",
    )
    maker.decide(context, ())
    first = client.completions.kwargs["extra_body"]["prompt_cache_key"]
    maker.decide(context.model_copy(update={"user_message": "second"}), ())
    second = client.completions.kwargs["extra_body"]["prompt_cache_key"]

    assert first == second


def test_configured_explicit_cache_applies_breakpoint_without_model_sniffing() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://compatible.example.test/v1/chat/completions",
            api_key="test",
            model="provider-model-alias",
            prompt_cache="explicit",
        ),
        client=client,
    )
    maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="continue",
        ),
        (),
    )

    system_block = client.completions.kwargs["messages"][0]["content"][0]
    assert system_block["prompt_cache_breakpoint"] == {"mode": "explicit"}
    stable_block = client.completions.kwargs["messages"][1]["content"][0]
    assert stable_block["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert client.completions.kwargs["extra_body"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }


def test_disabled_cache_is_visible_and_adds_no_provider_specific_fields() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://api.openai.com/v1/chat/completions",
            api_key="test",
            model="gpt-6",
            prompt_cache="disabled",
        ),
        client=client,
    )

    maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="continue",
        ),
        (),
    )

    assert "extra_body" not in client.completions.kwargs
    assert maker.cache_configuration() == {
        "prompt_cache_mode": "disabled",
        "prompt_cache_key_applied": False,
        "prompt_cache_breakpoint_applied": False,
        "prompt_cache_stable_slots": (
            "career_identity",
            "conversation_summary",
        ),
    }


def test_cache_usage_is_exposed_as_a_hit_ratio() -> None:
    client = Client()
    original_create = client.completions.create

    def create(**kwargs):
        response = original_create(**kwargs)
        details = type("Details", (), {"cached_tokens": 750})()
        response.usage = type(
            "Usage",
            (),
            {"prompt_tokens": 1000, "prompt_tokens_details": details},
        )()
        return response

    client.completions.create = create
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )
    maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="continue",
        ),
        (),
    )

    assert maker.consume_cache_metrics() == {
        "cache_metrics_reported": True,
        "cache_metrics_sample_count": 1,
        "cache_metrics_unreported_count": 0,
        "cache_metrics_unreported_ratio": 0.0,
        "input_units": 1000,
        "cached_input_units": 750,
        "cache_read_input_tokens": 750,
        "cache_hit_ratio": 0.75,
    }
    assert maker.consume_cache_metrics() == {}


def test_anthropic_cache_read_input_tokens_are_reported_exactly() -> None:
    client = Client()
    original_create = client.completions.create

    def create(**kwargs):
        response = original_create(**kwargs)
        response.usage = type(
            "Usage",
            (),
            {
                "input_tokens": 200,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 700,
            },
        )()
        return response

    client.completions.create = create
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )

    maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="continue",
        ),
        (),
    )

    metrics = maker.consume_cache_metrics()
    assert metrics["cache_read_input_tokens"] == 700
    assert metrics["cache_creation_input_tokens"] == 100
    assert metrics["uncached_input_tokens"] == 200
    assert metrics["input_units"] == 1000
    assert metrics["cache_hit_ratio"] == 0.7


def test_missing_cached_token_usage_is_distinct_from_a_zero_hit_rate() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
            prompt_cache="implicit",
        ),
        client=client,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="continue",
    )
    base_create = client.completions.create

    def create_without_cached_tokens(**kwargs):
        response = base_create(**kwargs)
        response.usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 1000,
                "prompt_tokens_details": type("Details", (), {})(),
            },
        )()
        return response

    client.completions.create = create_without_cached_tokens

    maker.decide(context, ())
    missing = maker.consume_cache_metrics()

    def create(**kwargs):
        response = base_create(**kwargs)
        details = type("Details", (), {"cached_tokens": 0})()
        response.usage = type(
            "Usage",
            (),
            {"prompt_tokens": 1000, "prompt_tokens_details": details},
        )()
        return response

    client.completions.create = create
    maker.decide(context, ())
    zero_hit = maker.consume_cache_metrics()

    assert missing == {
        "cache_metrics_reported": False,
        "cache_metrics_sample_count": 1,
        "cache_metrics_unreported_count": 1,
        "cache_metrics_unreported_ratio": 1.0,
        "input_units": 1000,
    }
    assert zero_hit["cache_metrics_reported"] is True
    assert zero_hit["cached_input_units"] == 0
    assert zero_hit["cache_hit_ratio"] == 0.0
    assert zero_hit["cache_metrics_sample_count"] == 2
    assert zero_hit["cache_metrics_unreported_ratio"] == 0.5


def test_static_request_serialization_is_memoized_for_the_tool_universe(
    monkeypatch,
) -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )
    specs = (
        {
            "type": "function",
            "function": {
                "name": "large_tool",
                "description": "d" * 4000,
                "parameters": {"type": "object", "properties": {}},
            },
        },
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="continue",
    )
    original_dumps = json.dumps
    static_serializations = 0

    def counting_dumps(value, *args, **kwargs):
        nonlocal static_serializations
        if "large_tool" in repr(value):
            static_serializations += 1
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(
        "career_agent.agent.openai_compatible_main_agent.json.dumps",
        counting_dumps,
    )

    maker.request_token_usage(context, specs)
    first_count = static_serializations
    maker.request_token_usage(
        context.model_copy(update={"user_message": "a new turn"}), specs
    )

    assert first_count == 1
    assert static_serializations == first_count


def test_recent_message_clipping_is_visible_in_native_history() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        recent_messages=(
            ConversationMessageContext(
                role="user",
                content="partial sentence",
                content_clipped=True,
                created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
            ),
        ),
        user_message="continue",
    )

    messages = project_decision_messages(context).messages(
        system_prompt="policy", spotlight_nonce="nonce"
    )

    assert messages[2]["content"].endswith("content_clipped=true]")


@pytest.mark.parametrize("clipped", [True, False])
def test_current_message_clipping_is_marked_like_a_clipped_window_message(
    clipped: bool,
) -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="partial request",
        user_message_source=(
            "partial request and the tail that was cut" if clipped else None
        ),
        user_message_clipped=clipped,
    )

    messages = project_decision_messages(context).messages(
        system_prompt="policy", spotlight_nonce="nonce"
    )

    current = messages[-1]
    assert current["role"] == "user"
    assert current["content"].startswith("partial request")
    assert current["content"].endswith(
        "\n\n[runtime message metadata: content_clipped=true]"
    ) is clipped
    assert all("the tail that was cut" not in m["content"] for m in messages)


def test_dynamic_control_does_not_change_the_static_system_message() -> None:
    base = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )
    active = base.model_copy(
        update={
            "task": ConversationTaskState(
                active_workflow="mock_interview",
                run_id="run-1",
                phase="mock_interview_running",
            )
        }
    )

    cold_messages = project_decision_messages(base).messages(
        system_prompt="static policy",
        spotlight_nonce="nonce",
    )
    active_messages = project_decision_messages(active).messages(
        system_prompt="static policy",
        spotlight_nonce="nonce",
    )

    assert cold_messages[0] == active_messages[0] == {
        "role": "system",
        "content": "static policy",
    }
    assert cold_messages[-3]["content"] != active_messages[-3]["content"]


def test_new_task_projection_fields_require_an_explicit_authority_classification() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )

    class ContextWithUnclassifiedTaskField:
        recent_messages = context.recent_messages
        user_message = context.user_message

        @staticmethod
        def reference_handles():
            return context.reference_handles()

        @staticmethod
        def model_context():
            projected = context.model_context()
            projected["task"]["new_display_candidates"] = [
                {"selection_index": 1, "title": "Must stay out of system"}
            ]
            return projected

    with pytest.raises(
        ValueError,
        match=r"classification is stale; unknown=\['new_display_candidates'\]",
    ):
        project_decision_messages(ContextWithUnclassifiedTaskField())


def test_missing_task_projection_fields_report_a_stale_classification() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )

    class ContextWithMissingTaskField:
        recent_messages = context.recent_messages
        user_message = context.user_message

        @staticmethod
        def reference_handles():
            return context.reference_handles()

        @staticmethod
        def model_context():
            projected = context.model_context()
            task = dict(projected["task"])
            task.pop("phase")
            return {**projected, "task": task}

    with pytest.raises(
        ValueError,
        match=r"classification is stale; unknown=\[\], missing=\['phase'\]",
    ):
        project_decision_messages(ContextWithMissingTaskField())


def test_main_agent_parses_native_tool_call() -> None:
    client = Client()
    function = type("Function", (), {"name": "open_job_search", "arguments": '{"keyword":"AI Engineer"}'})()
    tool_call = type("ToolCall", (), {"function": function})()
    message = type("Message", (), {"content": None, "tool_calls": [tool_call]})()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type("Response", (), {"choices": [choice]})()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"),
        client=client,
    )

    decision = maker.decide(MainAgentContext(conversation_id="c1", profile=CareerProfileContext(user_id="u1"), user_message="Find work."), ("open_job_search",))

    assert decision.action == "tool_call"
    assert decision.tool_call.name == "open_job_search"
    assert decision.tool_call.arguments == {"keyword": "AI Engineer"}


def test_observation_control_and_readable_text_are_split_by_authority() -> None:
    client = Client()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        tool_observations=(
            DecisionObservation(
                tool_name="find_saved_jobs",
                state="failed",
                message="UNTRUSTED TOOL MESSAGE",
                body="UNTRUSTED TOOL BODY",
                arguments={"query": "MODEL AUTHORED ARGUMENT"},
                facts={"retryable": True},
                next_action="UNTRUSTED TOOL SUGGESTION",
            ),
        ),
        user_message="继续",
    )

    maker.decide(context, ())

    messages = client.completions.kwargs["messages"]
    system_content = messages[0]["content"]
    data_content = messages[2]["content"]
    assistant_call = messages[-2]
    tool_result = messages[-1]
    turn_content = tool_result["content"]
    assert turn_content.startswith(TURN_OBSERVATION_LABEL + "\n")
    assert messages[-3] == {"role": "user", "content": "继续"}
    assert assistant_call["role"] == "assistant"
    assert assistant_call["tool_calls"][0]["function"]["name"] == "find_saved_jobs"
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == assistant_call["tool_calls"][0]["id"]
    for fragment in (
        "UNTRUSTED TOOL MESSAGE",
        "UNTRUSTED TOOL BODY",
        "UNTRUSTED TOOL SUGGESTION",
    ):
        assert fragment not in system_content
        assert fragment not in data_content
        assert fragment in turn_content
    assert "MODEL AUTHORED ARGUMENT" not in system_content
    assert "MODEL AUTHORED ARGUMENT" not in data_content
    assert "MODEL AUTHORED ARGUMENT" not in turn_content
    assert "MODEL AUTHORED ARGUMENT" in assistant_call["tool_calls"][0]["function"]["arguments"]


def test_main_agent_treats_plain_prose_without_tool_call_as_final() -> None:
    client = Client()
    message = type(
        "Message",
        (),
        {"content": "Hi! What would you like help with today?", "tool_calls": []},
    )()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type(
        "Response", (), {"choices": [choice]}
    )()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )

    decision = maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="hi",
        ),
        (),
    )

    assert decision.action == "final"
    assert decision.message == "Hi! What would you like help with today?"


@pytest.mark.parametrize("action", ("ask_user", "final"))
@pytest.mark.parametrize("alias", ("content", "text"))
def test_main_agent_accepts_content_as_the_prose_field_for_non_tool_decisions(
    action,
    alias,
) -> None:
    client = Client()
    message = type(
        "Message",
        (),
        {
            "content": json.dumps(
                {"action": action, alias: "本轮读取额度已用完。"}
            ),
            "tool_calls": [],
        },
    )()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type(
        "Response", (), {"choices": [choice]}
    )()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )

    decision = maker.decide(
        MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="继续",
        ),
        (),
    )

    assert decision.action == action
    assert decision.message == "本轮读取额度已用完。"


def test_main_agent_rejects_malformed_json_instead_of_showing_it_as_prose() -> None:
    client = Client()
    message = type(
        "Message",
        (),
        {"content": '{"action":"tool_call"', "tool_calls": []},
    )()
    choice = type("Choice", (), {"message": message})()
    client.completions.create = lambda **kwargs: type(
        "Response", (), {"choices": [choice]}
    )()
    maker = OpenAICompatibleMainAgentDecisionMaker(
        OpenAICompatibleAgentConfig(
            endpoint="https://example.test/v1/chat/completions",
            api_key="test",
            model="test-model",
        ),
        client=client,
    )

    with pytest.raises(AgentWorkerError) as captured:
        maker.decide(
            MainAgentContext(
                conversation_id="c1",
                profile=CareerProfileContext(user_id="u1"),
                user_message="hi",
            ),
            (),
        )

    assert captured.value.code == "MAIN_AGENT_INVALID_RESPONSE"
