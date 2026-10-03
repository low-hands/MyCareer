from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, Protocol


logger = logging.getLogger(__name__)


class CheckpointedTurn(Protocol):
    context: Any


class CheckpointLifecycle:
    """Retain resumable questionnaires and discard every settled graph thread."""

    def __init__(
        self,
        *,
        checkpointer: Any,
        thread_id: Callable[..., str],
    ) -> None:
        self._checkpointer = checkpointer
        self._thread_id = thread_id

    def settle(self, result: CheckpointedTurn) -> None:
        context = result.context
        if context.task.pending_questionnaire is not None:
            return
        thread_id = self._thread_id(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
        )
        try:
            self._checkpointer.delete_thread(thread_id)
        except Exception:
            # The business turn is already durable. Cleanup is maintenance and
            # must never rewrite a committed success into a client-visible
            # failure or invite the caller to repeat a side effect.
            logger.exception("failed to delete settled main graph checkpoint")
