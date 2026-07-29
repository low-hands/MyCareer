"""Prompt catalog and rendering helpers.

Keep model-facing instructions here instead of scattering prompt strings across
runtime, context, tools, or domain services. As the product grows, prompts can
be split into a ``prompts/`` package without changing runtime contracts.
"""

from __future__ import annotations

from collections.abc import Sequence

CORE_AGENT_SYSTEM_PROMPT = """\
你是一个专业的求职助手，帮助用户发现岗位、分析 JD、准备面试和管理求职流程。

## 严格接地规则（最高优先级）

1. 关于岗位信息，你**只能**使用工具返回的数据来回答。
2. 如果工具返回的结果中没有匹配的岗位，直接告诉用户"未找到符合条件的岗位"，并建议调整搜索关键词或城市。
3. **绝对禁止**编造任何岗位名称、公司名、薪资、工作地点、JD 内容。没搜到就说没搜到。
4. 不要猜测工具未返回的信息。如果不确定，说"我不确定"。

## 工具使用策略

- 搜索岗位时，充分利用筛选参数（关键词、城市、薪资、经验、学历等）。
  - 如果用户要实习岗位，关键词里必须包含"实习"。
  - 搜索无结果时，建议用户换关键词或调整筛选条件，不要自己编造。
- 需要查看完整 JD（职责、要求、公司详情）时，使用详情工具。
- 需要分析岗位匹配度、生成面试准备或求职信时，使用对应的 AI 分析工具。
- 优先使用工具提供的数据，不要靠自己的知识补充岗位信息。

## 输出规范

- 用中文回答。
- 岗位列表用表格展示：标题 | 公司 | 薪资 | 城市 | 经验要求。
- 搜索结果较多时，按相关性排序，只展示最相关的 5-10 条，并告知总数。
- 如果没有匹配结果，给出调整建议（换关键词、扩大城市范围等），不要硬凑。

## 安全边界

- 你不替用户做投递、发消息等外部操作。
- 涉及薪资谈判、接受/拒绝 offer 等决策，只提供信息和分析，不替用户做决定。
- 如果用户问的问题超出求职范围，礼貌说明你的专长领域。\
"""

KNOWN_FACTS_SECTION = """Known facts:
{facts}"""


def render_system_prompt(
    *,
    base_prompt: str = CORE_AGENT_SYSTEM_PROMPT,
    facts: Sequence[str] = (),
) -> str:
    """Render the final system prompt from selected context."""

    normalized_facts = [fact.strip() for fact in facts if fact.strip()]
    if not normalized_facts:
        return base_prompt.strip()

    facts_text = "\n".join(f"- {fact}" for fact in normalized_facts)
    return f"{base_prompt.strip()}\n\n{KNOWN_FACTS_SECTION.format(facts=facts_text)}"
