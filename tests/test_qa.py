"""Tests for grounded answer verification (citation + number checking)."""

from __future__ import annotations

from typing import Any

from core.qa import (
    QAService,
    _extract_faction_keywords,
    _numeric_tokens,
    clean_markdown,
    verify_answer,
)

CONTEXT = [
    {
        "chunk_id": 1,
        "announcement_id": 1,
        "revision_id": 1,
        "title": "9月10日1.5.0.9951版本更新公告",
        "announcement_date": "2026-09-10",
        "type": "官方公告",
        "url": "https://example.com/1",
        "content": "联动签到第一期9月10日7:00至10月12日7:00，签到第7天送后缀称号【不凡】。300%积分回馈期间充值时间与消耗通宝享3倍积分，9月14日7:00结束。",
    },
    {
        "chunk_id": 2,
        "announcement_id": 2,
        "revision_id": 2,
        "title": "9月5日活动公告",
        "announcement_date": "2026-09-05",
        "type": "官方公告",
        "url": "https://example.com/2",
        "content": "武林争霸赛第七届季后赛开启，大攻防9月12日晚开始。",
    },
]

# Separate contexts used by the end-to-end answer tests, where the fake
# provider returns a fixed LLM answer: the context must actually contain the
# numbers and times that the fake answer cites, otherwise verify_answer
# correctly drops those sentences.
_SIGNIN_CONTEXT = [
    {
        "chunk_id": 1,
        "announcement_id": 1,
        "revision_id": 1,
        "title": "9月10日版本更新公告",
        "announcement_date": "2026-09-10",
        "type": "官方公告",
        "url": "https://example.com/1",
        "content": "签到第一期9月10日7:00至10月12日7:00。",
    },
]

_UNSUPPORTED_CONTEXT = [
    {
        "chunk_id": 1,
        "announcement_id": 1,
        "revision_id": 1,
        "title": "9月10日版本更新公告",
        "announcement_date": "2026-09-10",
        "type": "官方公告",
        "url": "https://example.com/1",
        "content": "本期无任何名片相关公告。",
    },
]

_PLAIN_CONTEXT = [
    {
        "chunk_id": 1,
        "announcement_id": 1,
        "revision_id": 1,
        "title": "9月10日版本更新公告",
        "announcement_date": "2026-09-10",
        "type": "官方公告",
        "url": "https://example.com/1",
        "content": "签到第一期9月10日7:00至10月12日7:00。",
    },
]


def test_numeric_tokens_equivalence():
    tokens = _numeric_tokens("07:00 68,800通宝 300%")
    assert "7" in tokens and "07" in tokens
    assert "68800" in tokens
    assert "300" in tokens


def test_verify_keeps_supported_sentences_and_strips_markers():
    answer = "签到第一期在10月12日7:00结束[1]。300%积分回馈9月14日结束[1]。"
    kept = verify_answer(answer, CONTEXT)
    assert kept == [
        "签到第一期在10月12日7:00结束。",
        "300%积分回馈9月14日结束。",
    ]
    assert all("[1]" not in line for line in kept)


def test_verify_drops_sentences_with_out_of_range_citation():
    answer = "签到第一期在10月12日结束[5]。"
    assert verify_answer(answer, CONTEXT) == []


def test_verify_drops_sentences_with_unsupported_numbers():
    # 9月30日 never appears in the cited announcement text.
    answer = "签到第一期在9月30日结束[1]。"
    assert verify_answer(answer, CONTEXT) == []


def test_verify_drops_fabricated_item_names():
    # 【双人名片】 does not appear anywhere in the cited announcements.
    answer = "双人名片【双人名片】将于9月30日上线[1]。"
    assert verify_answer(answer, CONTEXT) == []


def test_verify_keeps_real_item_names():
    answer = "签到第7天送后缀称号【不凡】[1]。"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["签到第7天送后缀称号【不凡】。"]


def test_verify_drops_uncited_factual_sentences_but_keeps_transitions():
    answer = "以下是公告要点：\n签到第一期在10月12日结束[1]。\n新坐骑售价68800通宝。"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["以下是公告要点：", "签到第一期在10月12日结束。"]


def test_verify_multi_citation_requires_all_support():
    answer = "签到第一期10月12日结束，大攻防9月12日开始[1][2]。"
    kept = verify_answer(answer, CONTEXT)
    assert len(kept) == 1


def test_verify_exempts_group_headings_with_source_tag():
    answer = (
        "唐门技改分两类。\n"
        "一、已生效调整（9月10日版本更新）\n"
        "1. 奇穴增伤叠加异常已修复[1]；\n"
        "二、计划调整（9月8日资料片预告，尚未上线）\n"
        "1. 神机值体系联动整合[1]。"
    )
    kept = verify_answer(answer, CONTEXT)
    assert kept[0] == "唐门技改分两类。"
    # One blank line before every group heading for visual separation.
    assert kept == [
        "唐门技改分两类。",
        "",
        "一、已生效调整（9月10日版本更新）",
        "1. 奇穴增伤叠加异常已修复；",
        "",
        "二、计划调整（9月8日资料片预告，尚未上线）",
        "1. 神机值体系联动整合。",
    ]


def test_verify_ignores_item_index_numbers():
    answer = "1. 签到第7天送后缀称号[1]；"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["1. 签到第7天送后缀称号；"]


def test_verify_passes_through_when_format_ignored():
    answer = "签到第一期10月12日7:00结束。300%积分回馈9月14日结束。"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["签到第一期10月12日7:00结束。300%积分回馈9月14日结束。"]


def test_clean_markdown_strips_headings_bold_and_code():
    raw = (
        "### 一、联动活动\n"
        "1. **渡厄问心**：参与活动\n"
        "2. 联动签到`第一期`\n"
        "**来源：2026-09-10 版本更新公告**"
    )
    cleaned = clean_markdown(raw)
    assert "###" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "一、联动活动" in cleaned
    assert "渡厄问心：参与活动" in cleaned


class _FakeSearch:
    def __init__(self, context: list | None = None):
        self._context = context if context is not None else CONTEXT

    async def search(self, query: str, **kwargs: Any):
        return self._context


class _FakeProvider:
    def __init__(self, answer: str):
        self.answer = answer

    async def text_chat(self, prompt: str, system_prompt: str = "", **kwargs: Any):
        # Gate and route LLM calls need specific shapes; only the answer LLM
        # reads completion_text as a free-form grounded answer.
        if "入口网关" in system_prompt:
            return type(
                "R", (), {"completion_text": '{"trigger": true, "reason": "test"}'}
            )()
        if "改写为适合检索" in system_prompt:
            return type(
                "R",
                (),
                {
                    "completion_text": '{"rewrite": "签到", "keywords": ["签到"]}',
                },
            )()
        return type("R", (), {"completion_text": self.answer})()


def _service_with_provider(answer: str, context: list | None = None) -> QAService:
    search = _FakeSearch(context if context is not None else CONTEXT)
    service = QAService(search, context=None)
    provider = _FakeProvider(answer)

    async def fake_provider(umo: str | None = None):
        return provider

    service._provider = fake_provider
    return service


async def test_answer_grouped_layout():
    service = _service_with_provider(
        "唐门技改分两类。\n"
        "一、已生效调整（9月10日版本更新）\n"
        "1. 奇穴增伤叠加异常已修复[1]；\n"
        "二、计划调整（9月8日资料片预告，尚未上线）\n"
        "1. 神机值体系联动整合[1]。"
    )
    relevant, answer = await service.answer("唐门技改")
    assert relevant is True
    assert "[1]" not in answer
    assert "（9月10日版本更新）" in answer
    assert "（9月8日资料片预告，尚未上线）" in answer
    assert "来源：" not in answer


async def test_answer_simple_with_inline_source():
    service = _service_with_provider(
        "签到第一期在10月12日7:00结束[1]（9月10日版本更新）。",
        context=_SIGNIN_CONTEXT,
    )
    # Question must contain a trigger keyword ("活动") to pass the gate.
    relevant, answer = await service.answer("签到活动什么时候结束")
    assert relevant is True
    assert "[1]" not in answer
    assert "（9月10日版本更新）" in answer
    assert "10月12日" in answer


async def test_answer_all_sentences_unsupported_returns_fallback():
    """When every sentence is pruned by verify_answer, the fallback snippet
    summary is returned so the user still sees the retrieved context."""
    service = _service_with_provider(
        "双人名片即将于9月30日上线[1]。",
        context=_UNSUPPORTED_CONTEXT,
    )
    # Question must contain a trigger keyword ("公告") to pass the gate.
    relevant, answer = await service.answer("双人名片公告什么时候上线")
    assert relevant is True
    # verify_answer drops the sentence (9月30日 is unsupported); the fallback
    # snippet summary includes the announcement title and a content excerpt.
    assert "现有公告资料" in answer or "没有找到" in answer


async def test_answer_untouched_when_model_ignores_citations():
    service = _service_with_provider(
        "签到第一期在10月12日7:00结束。",
        context=_PLAIN_CONTEXT,
    )
    # Question must contain a trigger keyword ("活动") to pass the gate.
    relevant, answer = await service.answer("签到活动什么时候结束")
    assert relevant is True
    assert answer == "签到第一期在10月12日7:00结束。"


async def test_answer_no_keyword_skips_gate_llm():
    """A question without any trigger keyword is rejected immediately
    without calling the gate LLM, so normal chat stays fast."""
    service = _service_with_provider(
        "签到第一期在10月12日结束。",
        context=_PLAIN_CONTEXT,
    )
    # "签到什么时候结束" has no trigger keyword → gate rejects immediately.
    relevant, answer = await service.answer("签到什么时候结束")
    assert relevant is False
    assert answer == ""


def test_verify_aggregates_across_sibling_chunks_of_same_announcement():
    """A fact in chunk [2] is accepted when the model cites chunk [1] of the
    same announcement — verify_answer checks at announcement scope, not chunk
    scope, so sibling context saves borderline citations."""
    sibling_context = [
        {
            "chunk_id": 10,
            "announcement_id": 42,
            "revision_id": 42,
            "title": "10月13日武学调整公告",
            "announcement_date": "2026-10-13",
            "type": "官方公告",
            "url": "https://example.com/42",
            "content": "凌海诀对非侠士目标造成的伤害提高由38%调整为46%。",
        },
        {
            "chunk_id": 11,
            "announcement_id": 42,
            "revision_id": 42,
            "title": "10月13日武学调整公告",
            "announcement_date": "2026-10-13",
            "type": "官方公告",
            "url": "https://example.com/42",
            "content": "碧海缥缈掌气血回复提高，澹然若海新增调息返还。",
        },
    ]
    # The number 46 only appears in chunk [1] (chunk_id=10), but we cite [2]
    # (chunk_id=11). Old behavior dropped this; new behavior keeps it because
    # both chunks belong to announcement 42.
    answer = "凌海诀对非侠士目标伤害提高至46%[2]。"
    kept = verify_answer(answer, sibling_context)
    assert kept == ["凌海诀对非侠士目标伤害提高至46%。"]


def test_verify_still_drops_numbers_absent_from_entire_announcement():
    """A number not present anywhere in the cited announcement is still
    rejected — the relaxation is per-announcement, not unconditional."""
    context = [
        {
            "chunk_id": 1,
            "announcement_id": 1,
            "revision_id": 1,
            "title": "公告",
            "announcement_date": "2026-10-13",
            "type": "官方公告",
            "url": "https://example.com/1",
            "content": "伤害提高由38%调整为46%。",
        },
    ]
    answer = "伤害提高至99%[1]。"
    assert verify_answer(answer, context) == []


def test_extract_faction_keywords_detects_mentioned_schools():
    assert _extract_faction_keywords("蓬莱技改了什么") == ["蓬莱"]
    assert "万花" in _extract_faction_keywords("万花和蓬莱改了什么")
    assert _extract_faction_keywords("今天天气怎么样") == []


def test_verify_drops_empty_group_headings():
    """A group heading with no body lines after it is pruned from the output
    so the user never sees bare section titles like '一、凌海诀'."""
    context = [
        {
            "chunk_id": 1,
            "announcement_id": 1,
            "revision_id": 1,
            "title": "10月13日武学调整",
            "announcement_date": "2026-10-13",
            "type": "官方公告",
            "url": "https://example.com/1",
            "content": "凌海诀对非侠士目标伤害提高由38%调整为46%。",
        },
    ]
    answer = (
        "蓬莱加强了凌海诀伤害。\n"
        "一、凌海诀（10月13日版本更新）\n"
        "·凌海诀对非侠士目标伤害提高由38%调整为46%[1]。\n"
        "二、凌海诀·悟（10月13日版本更新）"
    )
    kept = verify_answer(answer, context)
    # Section "二" has no body lines and must be pruned; section "一" has
    # content and must be kept.
    assert "一、凌海诀（10月13日版本更新）" in kept
    assert "二、凌海诀·悟（10月13日版本更新）" not in kept


def test_verify_keeps_group_heading_when_followed_by_content():
    """A heading followed by real content in the next section is kept, even
    if the next section's heading comes later."""
    context = [
        {
            "chunk_id": 1,
            "announcement_id": 1,
            "revision_id": 1,
            "title": "公告",
            "announcement_date": "2026-10-13",
            "type": "官方公告",
            "url": "https://example.com/1",
            "content": "凌海诀伤害提高46%。驭羽骋风绝篇伤害降低50%。",
        },
    ]
    answer = (
        "一、凌海诀（10月13日版本更新）\n"
        "·凌海诀伤害提高46%[1]。\n"
        "二、凌海诀·悟（10月13日版本更新）\n"
        "·驭羽骋风绝篇伤害降低50%[1]。"
    )
    kept = verify_answer(answer, context)
    assert "一、凌海诀（10月13日版本更新）" in kept
    assert "二、凌海诀·悟（10月13日版本更新）" in kept


def test_verify_drops_junk_lines_of_bare_punctuation():
    """Lines containing only closing brackets/punctuation are dropped — the
    LLM sometimes emits bare '）' after copying effect descriptions."""
    context = [
        {
            "chunk_id": 1,
            "announcement_id": 1,
            "revision_id": 1,
            "title": "10月13日武学调整",
            "announcement_date": "2026-10-13",
            "type": "官方公告",
            "url": "https://example.com/1",
            "content": "凌海诀对非侠士目标伤害提高由38%调整为46%。",
        },
    ]
    answer = (
        "凌海诀伤害提高了。\n"
        "·凌海诀对非侠士目标伤害提高由38%调整为46%[1]。\n"
        "）\n"
        "）\n"
        "一、凌海诀·悟（10月13日版本更新）\n"
        "）"
    )
    kept = verify_answer(answer, context)
    # Bare "）" lines are dropped; the heading "一" has no body after junk
    # removal, so it is also pruned.
    assert "·凌海诀对非侠士目标伤害提高由38%调整为46%。" in kept
    assert "）" not in kept
    assert "一、凌海诀·悟（10月13日版本更新）" not in kept


def test_verify_no_citation_path_also_drops_junk_lines():
    """When the model ignores citation format entirely, junk-only lines are
    still stripped from the output."""
    context = [
        {
            "chunk_id": 1,
            "announcement_id": 1,
            "revision_id": 1,
            "title": "公告",
            "announcement_date": "2026-10-13",
            "type": "官方公告",
            "url": "https://example.com/1",
            "content": "伤害提高46%。",
        },
    ]
    answer = "凌海诀伤害提高了。\n）\n·具体数值提高了46%。\n）"
    kept = verify_answer(answer, context)
    assert len(kept) == 1
    assert "）" not in kept[0]
    assert "凌海诀伤害提高了。" in kept[0]
    assert "46%" in kept[0]
