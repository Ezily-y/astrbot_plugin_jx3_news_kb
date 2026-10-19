"""LLM relevance routing, query rewriting and grounded answers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any

from .search import SearchService

logger = logging.getLogger(__name__)

# Factions and schools whose names are strong signals for retrieval.
# When a question mentions one, the name is injected into keywords regardless
# of what the route-query LLM returns, so a bad rewrite cannot drop the core
# subject from the search.
_FACTION_KEYWORDS = [
    "蓬莱",
    "万花",
    "七秀",
    "纯阳",
    "天策",
    "少林",
    "藏剑",
    "五毒",
    "明教",
    "唐门",
    "丐帮",
    "苍云",
    "长歌",
    "霸刀",
    "衍天宗",
    "凌雪阁",
    "刀宗",
    "万灵",
    "天山",
    "少林",
]


GATE_SYSTEM_PROMPT = """你是剑网3公告知识库的入口网关。
你的任务是判断用户消息是否应该由"公告知识库"处理，还是应该交给普通聊天机器人处理。
只输出一个 JSON 对象，不要输出 Markdown 或解释。

JSON 字段：
- trigger: 布尔值，是否触发知识库问答。
- reason: 简短说明判断理由（不超过20字）。

触发规则（满足任意一条即 trigger=true）：
1. 消息包含以下必触发关键词（玩家术语也算）：
   - 技改、武学调整、平衡性调整、门派调整
   - 活动（含"活动什么时候""活动奖励""新活动"等）
   - 公告、通知、更新公告、版本更新
   - 资料片、新赛季、新服
   - 维护、停机
   - 名剑大会、JJC、竞技场
   - 副本、团队秘境、大战
   - 外装、外观、披风、挂件、礼盒
   - 奇穴、招式、技能调整
   - 兑换码、礼包、补偿
   - 咸鱼有话说
2. 消息明确询问剑网3官方近期/最新/某日的内容变更或活动信息。
3. 消息提到具体门派名（如蓬莱、万花、七秀、纯阳、天策、少林、藏剑、五毒、明教、唐门、丐帮、苍云、长歌、霸刀、衍天宗、凌雪阁、刀宗、万灵等）并涉及调整、改动、削弱、增强。

不触发的情况：
- 闲聊、问候、与剑网3无关的话题
- 纯游戏操作类请求（如"帮我组队""打副本"）
- 用户只是在聊天中提到游戏术语，但没有询问或查询意图
- 剑网3以外的游戏话题

只输出 JSON，例如：
{"trigger": true, "reason": "包含必触发词'技改'"}
{"trigger": false, "reason": "闲聊，无查询意图"}
"""

ROUTE_SYSTEM_PROMPT = """你负责将用户的剑网3问题改写为适合检索的关键词。
只输出一个 JSON 对象，不要输出 Markdown 或解释。
JSON 字段：
- rewrite: 适合检索的短句，必须使用官方公告用语，不要使用玩家社区简称。
- keywords: 2到6个关键词，必须使用官方公告用语。
- date_from: YYYY-MM-DD 或 null
- date_to: YYYY-MM-DD 或 null
- announcement_type: "官方公告"、"官方新闻" 或 null

重要：玩家常用术语与官方公告用语不同，rewrite 和 keywords 必须翻译为官方用语：
- 技改、平衡性调整 → 武学调整
- 副本、本 → 团队秘境、大战
- PVP、打架 → 名剑大会、阵营攻防
- 外观、皮肤 → 外装、披风、挂件
- 日常、刷本 → 大战、江湖潜行
- 竞技场、JJC → 名剑大会
- 门派调整、门派改动 → 武学调整
- 新资料片 → 资料片
- 维护、更新 → 版本更新
- 活动奖励、兑换 → 活动、兑换
rewrite 中应同时包含用户提到的具体对象（如门派名）和翻译后的官方术语。
例如用户问"蓬莱技改"，rewrite 应为"蓬莱 武学调整"，keywords 应为["蓬莱","武学调整","凌海诀","奇穴"]。
"""

ANSWER_SYSTEM_PROMPT = """你是剑网3官方公告知识库助手。
严格依据用户消息后面的"公告资料"回答，禁止编造。
默认以最新公告为准；如果资料中存在冲突，明确说明最新公告日期。
只有用户询问"是否改过""以前怎样""历史变化"时，才比较新旧公告。
回答使用简体中文纯文本，禁止 Markdown 符号（#、*、`、表格等）。
直接回答，不要开场白和客套语。
禁止输出孤立的标点符号行，例如单独的"）"、"）"、"）\n）"等。每一条效果描述的括号必须与招式名称写在同一行，不能单独成行。

核心原则：详尽完整，不要概括省略。
你必须把公告资料中与用户问题相关的所有具体信息都完整输出，绝对不能只列标题或概括。
用户问的是"加强了什么"、"改了什么"、"有什么活动"时，必须把每一条具体的改动、数值、招式、奇穴、奖励、条件全部列出。
禁止出现只有分组标题没有具体内容的情况。每一个"一、二、三"分组标题后面必须紧跟至少一条以"·"开头或直接描述具体改动的正文行；如果某个分组在资料中没有找到具体的改动条目，就不要列出该分组标题，直接跳过。
每一条改动都必须写明调整前和调整后的完整数值与效果描述。

回答格式：
- 第一行用一句话概括回答要点如xx加强了/削弱了+大致方向，句末标注资料编号如[1]，句后用括号注明来源，如”（10月13日版本更新）”。
- 随后按门派或玩法或主题用”一、二、三”分组，组标题末尾用括号注明该组来源。
- 同一门派下如有多个心法需要分开描述，用”【旗舰通用】””【花间游】””【离经易道】”等方括号标注心法名，不要使用#号。
- 组内每一条改动都必须完整写出，格式示例如下：
  一、万花（9月10日版本更新）
  【旗舰通用】
  ·混选奇穴【厥阴指】（若目标正在运功则进行打断并造成混元性内功伤害，并使其无法释放内功招式，持续4秒，使自身招式伤害提高15%，持续8秒。”离经易道”心法下返还10秒调息时间。）
  调整为：若目标正在运功则进行打断并造成混元性内功伤害，并使其无法释放内功招式，持续4秒，使自身招式伤害提高15%，持续8秒。”离经易道”心法下返还5秒调息时间。
  调息时间降低5秒。
  【花间游】
  ·修复了招式【天工甲士】可以在移动中施展的问题[1]。
  ·招式【钟林毓秀】、【商阳指】附带的持续伤害提高5%[1]。
  ·第六重奇穴【天工甲士】招式【碧波荡】伤害降低10%[1]。
  ·第六重奇穴【洛川神韵】墨迹炸裂时伤害提高100%[1]。
  【离经易道】
  ·第一重奇穴【微潮】治疗量提高10%[1]。
  ·第三重奇穴【知寒】治疗量大幅提高[1]。
  每条必须包含完整名称，具体的数值变化，完整的效果描述。
- 不要省略任何一条相关改动。资料中有几条相关改动就列几条，全部输出。
- 凡包含具体数字、日期、时间的句子，都必须标注资料编号；一句依据多条资料时写作[1][3]。
- 数字、日期、数值保持公告原文的写法，不要换算、改写或省略。
- 完整覆盖资料中与问题相关的信息，包括活动与系统名称，招式与奇穴名称及完整效果描述，物品、道具与外观奖励，日期、时间与次数，数值、档位与货币金额，获取条件与参与资格，设置与操作路径，修复的问题，调整前后的变化，以及适用的门派、体型、区服或账号范围。
- 招式、物品等游戏名称保持公告原文的写法，如【神风宝箱】【神机千变·悟】，不要改写或自创简称。
- 来源注明使用简短描述（如"10月13日版本更新"），不要照抄完整标题；同一来源只在组标题处注明一次。
- 编号用于内容核对，格式为[n]；除括号内注明的来源外，不要写其他来源说明。

如果用户说法与公告原文冲突，先指出公告原文的内容再回答。
如果资料不足，明确说"现有公告资料中没有找到"。
"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;])\s*|\n+")
_GROUP_HEADING_RE = re.compile(r"^[一二三四五六七八九十]+、")
_ITEM_INDEX_RE = re.compile(r"^\d+\s*[\.、]\s*")
_BRACKET_NAME_RE = re.compile(r"【([^【】]{1,24})】")
# Lines that contain only closing punctuation/brackets — artefacts of the
# LLM copying announcement text that used （...） for effect descriptions.
# These carry no information and must not appear in the final output.
_JUNK_LINE_RE = re.compile(r"^[\s）\)】\]》》,，。.、;；:：！!？?\-—·• ]+$")
_NO_FINDING_REPLY = "现有公告资料中没有找到相关内容。"


def clean_markdown(text: str) -> str:
    """Strip raw Markdown markers that would show up verbatim in chat messages."""
    text = _MD_HEADING_RE.sub("", text or "")
    text = text.replace("**", "").replace("`", "")
    return text.strip()


def _numeric_tokens(text: str) -> set[str]:
    """Numeric tokens in several equivalent spellings (07 == 7, 68,800 == 68800)."""
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0).replace(",", "")
        tokens.add(raw)
        stripped = raw.lstrip("0") or "0"
        tokens.add(stripped)
        try:
            tokens.add(str(float(stripped)))
        except ValueError:
            pass
    return tokens


def _extract_faction_keywords(question: str) -> list[str]:
    """Return any faction/school names mentioned in the question."""
    return [name for name in _FACTION_KEYWORDS if name in question]


def verify_answer(answer: str, context_items: list[dict[str, Any]]) -> list[str]:
    """Fact-check a cited answer against the retrieved announcements.

    Sentences are dropped when they cite an out-of-range item, name a
    【game item】 that does not appear in the cited text, or contain numbers
    absent from the cited text. Group headings ("一、...") carry the short
    source tag and are exempt from number/name checks; list-item indices
    ("1. ") are not treated as claim numbers. If the model ignored the
    citation format entirely, the answer passes through unpruned rather than
    losing every factual sentence.

    Number and name checks are evaluated against the full set of chunks
    belonging to the cited announcements (not just the specific chunk
    indices), so a fact mentioned in sibling chunk [1] is not wrongly
    rejected when the model cites chunk [2] of the same announcement.
    """
    text = (answer or "").strip()
    if not _CITATION_RE.search(text):
        # Model ignored the citation format; keep the answer as-is instead of
        # deleting every factual sentence, but still drop junk-only lines.
        if not text:
            return []
        lines = [
            line
            for line in text.splitlines()
            if line.strip() and not _JUNK_LINE_RE.match(line.strip())
        ]
        return ["\n".join(lines)] if lines else []

    # Per-announcement aggregated text: lets us verify facts against every
    # chunk of the cited announcement, not just the single chunk index the
    # model happened to reference.
    by_announcement: dict[int, str] = {}
    for item in context_items:
        aid = int(item.get("announcement_id", 0))
        content = str(item.get("content") or "")
        if aid:
            by_announcement[aid] = by_announcement.get(aid, "") + "\n" + content

    max_index = len(context_items)
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # Drop lines that are only closing punctuation/brackets (e.g. "）").
        if _JUNK_LINE_RE.match(sentence):
            continue
        citations = {int(m.group(1)) for m in _CITATION_RE.finditer(sentence)}
        bare = _CITATION_RE.sub("", sentence).strip()
        if citations and not all(1 <= index <= max_index for index in citations):
            continue

        if _GROUP_HEADING_RE.match(bare):
            kept.append(bare)
            continue

        item_match = _ITEM_INDEX_RE.match(bare)
        body = bare[item_match.end() :] if item_match else bare

        if not citations:
            if not _numeric_tokens(body):
                kept.append(bare)
            continue

        # Aggregate cited text across all chunks of each cited announcement.
        cited_announcement_ids = {
            int(context_items[index - 1].get("announcement_id", 0))
            for index in citations
            if 1 <= index <= max_index
        }
        cited_text = " ".join(
            by_announcement.get(aid, "") for aid in cited_announcement_ids
        )

        numbers = _numeric_tokens(body)
        if numbers and not numbers <= _numeric_tokens(cited_text):
            continue
        names = _BRACKET_NAME_RE.findall(body)
        if names and not all(name in cited_text for name in names):
            continue
        kept.append(bare)

    # Visual separation: one blank line before every group heading.
    spaced: list[str] = []
    for line in kept:
        if _GROUP_HEADING_RE.match(line) and spaced:
            spaced.append("")
        spaced.append(line)

    # Drop empty group headings: a heading followed immediately by another
    # heading or by end-of-list has no body content and must not appear in
    # the final output (the LLM sometimes emits bare section titles).
    pruned: list[str] = []
    for index, line in enumerate(spaced):
        if _GROUP_HEADING_RE.match(line):
            # Peek ahead: if the next non-blank line is another heading or
            # there is no next line, this heading is empty → skip it.
            next_content = None
            for later in spaced[index + 1 :]:
                if later.strip():
                    next_content = later
                    break
            if next_content is None or _GROUP_HEADING_RE.match(next_content):
                continue
        pruned.append(line)

    # Trim leading blank lines left after pruning empty headings.
    while pruned and not pruned[0].strip():
        pruned.pop(0)

    return pruned


class QAService:
    # Keywords that should trigger a live API fetch before searching the local KB.
    # When the user asks about these topics, we pull the latest announcements
    # from the JX3API so the knowledge base is up-to-date.
    LIVE_FETCH_KEYWORDS = [
        "技改",
        "武学调整",
        "平衡性调整",
        "门派调整",
        "更新公告",
        "版本更新",
        "咸鱼有话说",
    ]

    def __init__(
        self,
        search_service: SearchService,
        context: Any | None = None,
        llm_provider_id: str = "",
        max_context_items: int = 64,
        max_answer_length: int = 2400,
    ) -> None:
        self.search = search_service
        self.context = context
        self.llm_provider_id = llm_provider_id
        self.max_context_items = max(1, int(max_context_items))
        self.max_answer_length = max(200, int(max_answer_length))
        self._fetch_callback: Any = None

    def set_fetch_callback(self, callback: Any) -> None:
        """Register a callback to fetch fresh data from the API when needed.

        Args:
            callback: An async callable that accepts ``limit`` (int) and
                triggers a fetch-and-ingest cycle. Typically bound to
                ``plugin.fetch_and_ingest``.
        """
        self._fetch_callback = callback

    def _needs_live_fetch(self, question: str) -> bool:
        """Check if the question exactly matches a live-fetch trigger phrase.

        Only pulls from the API when the question is specifically about one of
        the trigger topics, not just incidentally containing the word.
        The remainder after removing the keyword must be ONLY qualifier words
        (time words, question particles, punctuation), not other nouns.
        """
        lower = question.lower().strip()
        for kw in self.LIVE_FETCH_KEYWORDS:
            kw_lower = kw.lower()
            if kw_lower not in lower:
                continue
            remainder = lower.replace(kw_lower, "").strip()
            if not remainder:
                return True
            # Split remainder into tokens by common punctuation and particles
            tokens = re.split(r"[\s,，。？！、的了吗呢吧么嘛啊呀哇吗呢么]+", remainder)
            # Also split by known qualifier prefixes to catch "看看最新" -> ["看看","最新"]
            split_tokens: list[str] = []
            for t in tokens:
                if not t:
                    continue
                # Break off known qualifier prefixes
                for prefix in ("看看", "查询", "查看", "查找", "搜搜", "最新", "最近"):
                    while t.startswith(prefix) and len(t) > len(prefix):
                        split_tokens.append(prefix)
                        t = t[len(prefix) :]
                if t:
                    split_tokens.append(t)
            tokens = split_tokens
            if not tokens:
                return True
            # Qualifier whitelist: if ALL tokens are in this set, it's a simple query.
            # If any token is NOT in this set, the question mixes topics → no fetch.
            qualifier_words = {
                "最新",
                "最近",
                "今天",
                "昨天",
                "明天",
                "这周",
                "本周",
                "上周",
                "上周",
                "全部",
                "所有",
                "啥",
                "什么",
                "哪些",
                "哪些",
                "多少",
                "几次",
                "没",
                "过",
                "来着",
                "怎么",
                "何时",
                "啥时候",
                "什么时候",
                "有",
                "查",
                "看看",
                "看下",
                "一下",
                "的了",
                "了",
                "吗",
                "呢",
                "吧",
                "么",
                "啊",
                "呀",
                "哇",
                "嘛",
                "是",
                "要",
                "有",
                "更新",
                "维护",
                "公告",
                "新闻",
            }
            if all(t in qualifier_words for t in tokens):
                return True
            # Also allow if each token is short AND in a broader qualifier set
            # (covers "查询" "查看" "找" etc.)
            if all(
                len(t) <= 3
                and any(c in "查搜找看更新了过在们这那这哪有啥什怎么时" for c in t)
                for t in tokens
            ):
                return True
            # Otherwise: multi-topic question → do not fetch
            return False
        return False

    async def _provider(self, umo: str | None = None) -> Any | None:
        if self.context is None:
            return None
        if self.llm_provider_id:
            return self.context.get_provider_by_id(self.llm_provider_id)
        return await self.context.get_using_provider_async(umo)

    async def gate_check(
        self,
        question: str,
        provider: Any | None,
    ) -> dict[str, Any]:
        """First-layer gate: decide whether the knowledge base should handle this message.

        Returns:
            Dict with 'trigger' (bool) and 'reason' (str).
            When trigger is True the message should proceed to route_query + search.

        Keyword match is a prerequisite: messages without any trigger keyword
        are rejected immediately (no LLM call) to avoid latency on normal
        chat. When keywords do match, the gate LLM further verifies the user
        actually has a query intent rather than just mentioning the word
        in passing (e.g. "这个活动不错" vs "最近有什么活动").
        """
        trigger_keywords = [
            "技改",
            "武学调整",
            "平衡性调整",
            "门派调整",
            "活动",
            "公告",
            "通知",
            "更新公告",
            "版本更新",
            "资料片",
            "新赛季",
            "新服",
            "维护",
            "停机",
            "名剑大会",
            "jjc",
            "竞技场",
            "团队秘境",
            "大战",
            "外装",
            "外观",
            "披风",
            "挂件",
            "礼盒",
            "奇穴",
            "招式",
            "兑换码",
            "礼包",
            "补偿",
            "咸鱼有话说",
        ]
        lower = question.lower()
        matched_keywords = [kw for kw in trigger_keywords if kw.lower() in lower]

        # No trigger keyword → reject immediately, no LLM call.
        # This is the fast path for normal chat messages.
        if not matched_keywords:
            return {"trigger": False, "reason": "未命中触发关键词，跳过知识库"}

        # Keywords matched but no provider → trigger anyway so search runs.
        if provider is None:
            return {
                "trigger": True,
                "reason": f"关键词命中: {matched_keywords[0]}（无LLM，默认触发）",
            }

        # Keywords matched → let the gate LLM confirm real query intent.
        keyword_hint = f"\n消息中包含以下关键词：{', '.join(matched_keywords[:5])}。"
        try:
            response = await provider.text_chat(
                prompt=f"用户消息：{question}{keyword_hint}",
                system_prompt=GATE_SYSTEM_PROMPT,
            )
            text = str(getattr(response, "completion_text", "") or "")
            result = extract_json_object(text)
            if result and isinstance(result.get("trigger"), bool):
                return {
                    "trigger": result["trigger"],
                    "reason": str(result.get("reason", "")),
                }
        except Exception:  # noqa: BLE001 - gate failure should not block
            pass

        # LLM output malformed → default to trigger since keywords matched;
        # the search layer will return empty if the KB has nothing relevant.
        return {
            "trigger": True,
            "reason": f"关键词命中: {matched_keywords[0]}，网关LLM未返回有效结果，默认触发",
        }

    async def route_query(
        self,
        question: str,
        provider: Any | None,
    ) -> dict[str, Any]:
        fallback = {
            "rewrite": question,
            "keywords": [question],
            "date_from": None,
            "date_to": None,
            "announcement_type": None,
        }
        if provider is None:
            return fallback
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        response = await provider.text_chat(
            prompt=f"当前时间：{now}\n用户消息：{question}",
            system_prompt=ROUTE_SYSTEM_PROMPT,
        )
        text = str(getattr(response, "completion_text", "") or "")
        routed = extract_json_object(text)
        if not routed:
            return fallback
        routed.setdefault("rewrite", question)
        routed.setdefault("keywords", [question])
        routed["keywords"] = [
            str(item) for item in routed["keywords"] if str(item).strip()
        ]
        if not routed["keywords"]:
            routed["keywords"] = [question]
        return routed

    async def answer(
        self,
        question: str,
        umo: str | None = None,
    ) -> tuple[bool, str]:
        provider = await self._provider(umo)

        # Layer 1 + 1.5 + 2: run gate, route and (when relevant) the live
        # fetch in parallel. Gate and route are both pure LLM calls and do
        # not depend on each other; the live fetch only needs to finish
        # before the search, so it can run alongside the LLMs too.
        # Use asyncio.gather (not create_task) so the concurrent calls work
        # both in production and under pytest-asyncio's managed event loop.
        do_live = self._needs_live_fetch(question) and self._fetch_callback is not None

        async def _gate() -> dict[str, Any]:
            return await self.gate_check(question, provider)

        async def _route() -> dict[str, Any]:
            return await self.route_query(question, provider)

        coros: list[Any] = [_gate(), _route()]
        if do_live:
            coros.append(self._run_live_fetch())

        gathered = await asyncio.gather(*coros)
        gate_result = gathered[0]
        routed_result = gathered[1]

        if not gate_result.get("trigger", False):
            return False, ""

        # Inject faction/school names detected in the question so a poor LLM
        # rewrite cannot drop the core subject from the search.
        routed = dict(routed_result or {})
        faction_hits = _extract_faction_keywords(question)
        if faction_hits:
            existing_keywords = list(routed.get("keywords") or [])
            for name in faction_hits:
                if name not in existing_keywords:
                    existing_keywords.append(name)
            routed["keywords"] = existing_keywords
            rewrite = str(routed.get("rewrite") or question)
            for name in faction_hits:
                if name not in rewrite:
                    rewrite = f"{name} {rewrite}".strip()
            routed["rewrite"] = rewrite

        results = await self.search.search(
            str(routed.get("rewrite") or question),
            keywords=routed.get("keywords"),
            date_from=routed.get("date_from"),
            date_to=routed.get("date_to"),
            announcement_type=routed.get("announcement_type"),
            top_k=self.max_context_items,
        )
        if not results:
            if provider is None:
                return True, _NO_FINDING_REPLY
            response = await provider.text_chat(
                prompt=(
                    f"当前时间：{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                    f"用户问题：{question}\n\n公告资料：无\n"
                    f"请按系统规则回答。最大长度：{self.max_answer_length} 字。"
                ),
                system_prompt=ANSWER_SYSTEM_PROMPT,
            )
            return True, clean_markdown(
                str(getattr(response, "completion_text", "") or "")
            )

        context_parts: list[str] = []
        for index, item in enumerate(results, start=1):
            context_parts.append(
                "\n".join(
                    [
                        f"[{index}] 日期：{item['announcement_date']}",
                        f"类型：{item['type']}",
                        f"标题：{item['title']}",
                        f"链接：{item['url']}",
                        "内容：",
                        item["content"],
                    ]
                )
            )
        context_text = "\n\n".join(context_parts)
        if provider is None:
            return True, self._fallback_answer(results)

        response = await provider.text_chat(
            prompt=(
                f"当前时间：{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                f"用户问题：{question}\n\n公告资料：\n{context_text}\n\n"
                f"最大回答长度：{self.max_answer_length} 字。"
            ),
            system_prompt=ANSWER_SYSTEM_PROMPT,
        )
        answer_text = clean_markdown(
            str(getattr(response, "completion_text", "") or "")
        )
        kept = verify_answer(answer_text, results)
        if not kept:
            # LLM produced no usable sentences; fall back to snippet summary
            # rather than returning a bare "not found" so the user still gets
            # the retrieved knowledge base content.
            return True, self._fallback_answer(results)
        return True, "\n".join(kept)

    async def _run_live_fetch(self, limit: int = 5) -> dict[str, Any]:
        """Pull the latest announcements from the API before searching."""
        if self._fetch_callback is None:
            return {}
        try:
            fetch_result = await self._fetch_callback(limit=limit)
            inserted = (
                fetch_result.get("inserted", 0) if isinstance(fetch_result, dict) else 0
            )
            revised = (
                fetch_result.get("revised", 0) if isinstance(fetch_result, dict) else 0
            )
            if inserted or revised:
                logger.info(
                    "Live fetch finished: inserted=%d revised=%d",
                    inserted,
                    revised,
                )
            return fetch_result if isinstance(fetch_result, dict) else {}
        except Exception:  # noqa: BLE001 - must not block the QA flow
            logger.exception("Live fetch before QA failed")
            return {}

    def _fallback_answer(self, results: list[dict[str, Any]]) -> str:
        lines = ["根据现有公告资料，相关内容如下："]
        for item in results[:3]:
            snippet = item["content"].replace("\n", " ")
            lines.append(
                f"- {item['announcement_date']}《{item['title']}》：{snippet[:240]}（{item['url']}）"
            )
        return "\n".join(lines)
