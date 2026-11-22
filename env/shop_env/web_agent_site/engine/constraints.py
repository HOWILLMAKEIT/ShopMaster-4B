"""Explicit budget extraction from natural-language shopping instructions."""

from __future__ import annotations

import re


def explicit_budget_from_instruction(instruction):
    """Extract a clearly stated upper budget; return None when ambiguous."""
    text = str(instruction or "").replace(",", "")

    def scaled(number, unit):
        value = float(number)
        normalized_unit = str(unit or "").casefold()
        if normalized_unit == "万":
            value *= 10000
        elif normalized_unit in {"千", "k"}:
            value *= 1000
        return value

    shorthand = re.search(
        r"预算(?:控制)?在?\s*(\d+)\s*万\s*(\d+)\s*(?:千)?\s*(以内|以下|内|左右)?",
        text,
    )
    if shorthand:
        value = float(shorthand.group(1)) * 10000 + float(shorthand.group(2)) * 1000
        if shorthand.group(3) == "左右":
            value *= 1.1
        return value

    # Ranges are explicit upper bounds. Parse them before single-value
    # patterns so “10-20” is never mistaken for an upper bound of 10.
    price_range = re.search(
        r"(?:预算|价格)(?:控制)?在?\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?\s*"
        r"(?:-|~|～|至|到)\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?"
        r"(?:之间|以内|以下|左右)?",
        text,
    )
    if price_range:
        low = scaled(price_range.group(1), price_range.group(2))
        high = scaled(price_range.group(3), price_range.group(4))
        if low > 0 and high >= low:
            return high

    # “4k+” describes a lower/open-ended price, not a maximum.
    if re.search(
        r"(?:预算|价格)(?:控制)?在?\s*\d+(?:\.\d+)?\s*[kK]\s*\+",
        text,
    ):
        return None

    patterns = (
        r"预算(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"价格(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        # 带万/千/k 时“元”可省（“别超过3千”）；单位后若紧跟计量字
        # （千瓦、万毫安、千兆、kw 等）则说明该数字是功率/容量等规格量，
        # 不是价格，必须跳过，否则会遮蔽同一句中真正的价格上限。
        r"(?:别超过|不要超过|不要高于|不超过|不高于|最高)\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])\s*元?(?![a-zA-Z瓦克兆安伏赫吨帕焦米毫次年位])",
        r"(?:别超过|不要超过|不要高于|不超过|不高于|最高)\s*(\d+(?:\.\d+)?)\s*元",
        r"(\d+(?:\.\d+)?)\s*(万|千)?\s*元(以内|以下)",
    )
    for pattern in patterns:
        match = _first_upper_match(pattern, text)
        if match:
            unit = match.group(2) if match.re.groups >= 2 else None
            value = scaled(match.group(1), unit)
            qualifier = match.group(3) if match.re.groups >= 3 and match.lastindex and match.lastindex >= 3 else None
            # “左右” is not a strict upper bound; apply a fixed 10% tolerance
            # so target items just above a round-number budget remain feasible.
            if qualifier == "左右":
                value *= 1.1
            if value > 0:
                return value
    return None


# 紧跟在数字金额之后的下界表达：“60元起”“60元以上”描述的是下限，
# 不是可用的预算上限。
_LOWER_BOUND_MARKERS = ("起", "起步", "起价", "以上", "往上")


def _first_upper_match(pattern, text):
    """返回第一个不后接下界标记的匹配；全部带下界标记时返回 None。"""
    for match in re.finditer(pattern, text):
        rest = text[match.end():match.end() + 2]
        if any(rest.startswith(marker) for marker in _LOWER_BOUND_MARKERS):
            continue
        return match
    return None
