from __future__ import annotations

from career_agent.agent.main_agent_contracts import ConversationSpanView


def render_conversation_span(view: ConversationSpanView) -> str:
    """Render an exact, bounded slice of this conversation's durable rows."""
    heading = (
        f"会话消息序号 {view.from_sequence}–{view.through_sequence}："
        f"返回 {view.returned}/{view.total} 条。"
    )
    if view.body_clipped:
        heading += " 本次 observation 正文仍被截断，不能视为完整回读。"
    if not view.messages:
        return heading + "\n\n该范围内没有消息。"
    blocks = [heading]
    for message in view.messages:
        role = "用户" if message.role == "user" else "助手"
        clipped = " · 内容截断" if message.content_clipped else ""
        blocks.append(
            f"### 序号 {message.sequence} · {role}{clipped}\n\n{message.content}"
        )
    if view.returned < view.total:
        blocks.append(
            "该范围还有消息未返回；请从序号 "
            f"{view.messages[-1].sequence + 1} 继续读取。"
        )
    return "\n\n".join(blocks)
