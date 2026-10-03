from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from career_agent.agent.contracts.main_agent import (
    AgentDecision,
    MainAgentContext,
    ToolObservation,
)
from career_agent.domain.resume import ResumeArtifactDelivery


Originator = Literal["model", "user", "runtime"]
"""Who asked for a turn. Derived from its origin variant, never set beside it."""

OriginKind = Literal["model", "interaction", "workflow", "policy"]
"""The union tag and the single source of each variant's label prefix.

The tag is a ``ClassVar`` so callers cannot construct a variant whose tag
disagrees with its type. Origins currently stay inside one process; if they
need to cross a serialization boundary, add an explicit tagged codec rather
than treating this class variable as serialized data.
"""


@dataclass(frozen=True)
class ModelDecision:
    """The model chose this turn's action, and this is the choice it made."""

    decision: AgentDecision

    kind: ClassVar[OriginKind] = "model"
    requested_by: ClassVar[Originator] = "model"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.decision.action}"


@dataclass(frozen=True)
class InteractionReceipt:
    """The user answered an interaction whose contract was sealed when issued."""

    scope: str
    action: str

    kind: ClassVar[OriginKind] = "interaction"
    requested_by: ClassVar[Originator] = "user"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.scope}"


@dataclass(frozen=True)
class RuntimeAction:
    """The user's message went to a workflow the runtime owns."""

    workflow: Literal["job_discovery", "mock_interview"]

    kind: ClassVar[OriginKind] = "workflow"
    requested_by: ClassVar[Originator] = "user"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.workflow}"


@dataclass(frozen=True)
class RuntimePolicyAction:
    """A deterministic policy step taken before model deliberation."""

    policy: Literal[
        "free_text_preference_confirmation",
        "free_text_preference_activation",
        "career_fact_confirmation",
        "job_intent_confirmation",
    ]

    kind: ClassVar[OriginKind] = "policy"
    requested_by: ClassVar[Originator] = "runtime"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.policy}"


TurnOrigin = ModelDecision | InteractionReceipt | RuntimeAction | RuntimePolicyAction
"""How a turn came to exist, without fabricating a model decision."""


class MainAgentTurnResult:
    """One completed turn: how it started, and what it produced.

    Accountability lives in ``origin``. Shared output fields stay on this
    envelope so consumers that only render or commit a turn do not need to
    branch on its ingress.
    """

    def __init__(
        self,
        *,
        origin: TurnOrigin,
        context: MainAgentContext,
        assistant_message: str,
        tool_result: ToolObservation | None = None,
        tool_results: tuple[ToolObservation, ...] = (),
        artifacts: tuple[ResumeArtifactDelivery, ...] = (),
        content_streamed: bool = False,
        model_message: str = "",
        delegated_read_count: int = 0,
        delegated_write_count: int = 0,
        career_memory_scope_keys: tuple[str, ...] = (),
    ) -> None:
        self.origin = origin
        self.context = context
        self.assistant_message = assistant_message
        self.tool_result = tool_result
        self.tool_results = tool_results
        self.artifacts = artifacts
        self.content_streamed = content_streamed
        self.model_message = model_message
        self.delegated_read_count = delegated_read_count
        self.delegated_write_count = delegated_write_count
        self.career_memory_scope_keys = career_memory_scope_keys

    @property
    def requested_by(self) -> Originator:
        return self.origin.requested_by

    @property
    def model_decision(self) -> AgentDecision | None:
        """The model's choice, or ``None`` when the model made none."""

        return self.origin.decision if isinstance(self.origin, ModelDecision) else None
