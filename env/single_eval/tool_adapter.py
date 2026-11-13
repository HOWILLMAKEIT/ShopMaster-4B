"""Teacher 工具转接层：JSON Schema、参数校验与环境动作转换。

模型通过原生 tool_calls 表达动作；本模块把三个购物工具转换成环境
``search[...]`` / ``click[...]`` / ``finish[...]`` 字符串动作，并对照公开
observation_state 校验动作在当前页面合法。goal、reward 等隐藏字段不经过
本模块。
"""

from typing import Any, Dict, List, Tuple

FINISH_REASON = "no_suitable_product"


class ToolCallError(Exception):
    """工具名、参数或动作合法性校验失败。"""


def _schema(name: str, description: str, properties=None, required=None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


SHOP_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    _schema(
        "search_products",
        "仅当最新观察的 search_available 为 true 时搜索。query 应简洁，包含品类和最有"
        "区分度的品牌、型号、功能或规格；不要重复相同查询或机械复制整段需求。",
        {"query": {"type": "string", "description": "搜索关键词"}},
        ["query"],
    ),
    _schema(
        "click",
        "点击当前页面可点击项之一。value 必须原样取自最新观察的 actions 列表，"
        "用于打开商品、选择规格、翻页、查看详情或点击 Buy Now。",
        {"value": {"type": "string", "description": "要点击的按钮或商品值"}},
        ["value"],
    ),
    _schema(
        "finish_without_purchase",
        "主动结束且不购买。经过多次有实质差异的搜索和候选核验后仍没有可接受商品时"
        "调用；是否达到终止资格由环境判断。",
        {"reason": {"type": "string", "enum": [FINISH_REASON]}},
        ["reason"],
    ),
]


def tool_call_to_action(
    name: str, arguments: Dict[str, Any], observation_state: Dict[str, Any]
) -> str:
    """校验一个工具调用并转换为环境动作字符串。

    Args:
        name: 工具名。
        arguments: 模型给出的 JSON 参数。
        observation_state: 该轮执行前的公开环境状态，用于合法性校验。

    Returns:
        环境动作字符串，如 ``search[键盘]``。

    Raises:
        ToolCallError: 工具未知、参数缺失或类型不符、动作在当前页面不合法。
    """
    if not isinstance(arguments, dict):
        raise ToolCallError(f"工具参数必须是 JSON 对象，收到: {type(arguments).__name__}")

    if name == "search_products":
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolCallError("search_products 缺少非空字符串参数 query")
        if not observation_state.get("search_available", False):
            raise ToolCallError("当前页面不可搜索（search_available 为 false）")
        return f"search[{query.strip()}]"

    if name == "click":
        value = arguments.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ToolCallError("click 缺少非空字符串参数 value")
        value = value.strip()
        if value not in observation_state.get("actions", []):
            raise ToolCallError(f"点击项不在当前页面可用动作中: {value!r}")
        return f"click[{value}]"

    if name == "finish_without_purchase":
        reason = arguments.get("reason")
        if reason != FINISH_REASON:
            raise ToolCallError(f"finish_without_purchase 的 reason 只允许 {FINISH_REASON!r}")
        return f"finish[{FINISH_REASON}]"

    raise ToolCallError(f"未知工具: {name}")


def parse_tool_calls(message: Dict[str, Any], observation_state: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any], str]]:
    """从 assistant 消息解析并校验工具调用。

    每轮只允许一个调用；没有 tool_calls 视为模型停止调用，返回空列表，
    由调用方按未完成终止记录失败。

    Returns:
        [(tool_call_id, 动作字符串, arguments, 工具名)]，长度 0 或 1。
    """
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) == 0:
        return []
    if len(tool_calls) > 1:
        raise ToolCallError(f"每轮只允许一个工具调用，收到 {len(tool_calls)} 个")

    call = tool_calls[0]
    function = call.get("function") or {}
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, str):
        import json

        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolCallError(f"工具参数不是合法 JSON: {exc}") from exc
    else:
        arguments = raw_arguments

    action = tool_call_to_action(name, arguments, observation_state)
    return [(call.get("id", ""), action, arguments, name)]
