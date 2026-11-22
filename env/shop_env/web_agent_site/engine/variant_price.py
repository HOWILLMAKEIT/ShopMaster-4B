"""Option axis canonicalization and variant-price resolution."""

from __future__ import annotations

import math
import re
import unicodedata


VARIANT_PRICE_VERSION = "variant-price-v1"
OPTION_AXIS_VERSION = "option-axis-v1"
PASS = "pass"
UNVERIFIABLE = "unverifiable"

_AXIS_ALIASES = {
    "color": {"颜色", "颜色分类"},
    "size": {"尺码", "鞋码"},
    "dimensions": {"尺寸", "大小"},
    "net_content": {"净含量", "总净含量"},
    "flavor": {"口味", "食品口味"},
    "specification": {"规格", "规格描述", "规格类型"},
    "bundle": {"套餐", "套餐类型", "组合套餐"},
    "capacity": {"容量", "规格容量"},
}


def normalize_option_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("/", "|")
    return re.sub(r"\s+", "", text)


def canonicalize_option_axis(value: object) -> str:
    normalized = normalize_option_text(value)
    for canonical, aliases in _AXIS_ALIASES.items():
        if normalized in {
            normalize_option_text(alias) for alias in aliases
        }:
            return canonical
    return normalized


def _finite_price(value):
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def option_axes(product: dict) -> dict[str, dict]:
    """Return canonical axes while preserving ambiguity as explicit evidence."""
    axes = {}
    collisions = {}
    raw = product.get("customization_options") or {}
    if not isinstance(raw, dict):
        raw = {}
    for raw_axis, entries in raw.items():
        canonical = canonicalize_option_axis(raw_axis)
        values = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            normalized = normalize_option_text(entry.get("value"))
            if normalized:
                values[normalized] = {
                    "value": str(entry.get("value")),
                    "price": _finite_price(entry.get("price")),
                }
        record = {
            "canonical_axis": canonical,
            "source_axis": str(raw_axis),
            "values": values,
        }
        if canonical in axes:
            collisions.setdefault(canonical, [axes[canonical]["source_axis"]])
            collisions[canonical].append(str(raw_axis))
        else:
            axes[canonical] = record
    for canonical in collisions:
        axes.pop(canonical, None)
    return {"axes": axes, "collisions": collisions}


def _canonical_selected(selected_options: object) -> tuple[dict, list[dict]]:
    if not isinstance(selected_options, dict):
        return {}, [{"code": "selected_options_not_object"}]
    selected = {}
    errors = []
    for raw_axis, raw_value in selected_options.items():
        canonical = canonicalize_option_axis(raw_axis)
        if canonical in selected:
            errors.append(
                {
                    "code": "selected_axis_collision",
                    "axis": canonical,
                }
            )
            continue
        selected[canonical] = {
            "source_axis": str(raw_axis),
            "value": str(raw_value),
        }
    return selected, errors


def _combination_price(product: dict, selected: dict) -> float | None:
    variants = product.get("variant_combinations")
    if not isinstance(variants, list):
        return None
    normalized_selected = {
        canonicalize_option_axis(key): normalize_option_text(value)
        for key, value in selected.items()
    }
    for variant in variants:
        if not isinstance(variant, dict) or not isinstance(
            variant.get("options"), dict
        ):
            continue
        normalized_variant = {
            canonicalize_option_axis(key): normalize_option_text(value)
            for key, value in variant["options"].items()
        }
        if normalized_variant == normalized_selected:
            return _finite_price(variant.get("price"))
    return None


def resolve_variant_price(product: dict, selected_options: object) -> dict:
    selected, errors = _canonical_selected(selected_options)
    indexed = option_axes(product)
    if errors or indexed["collisions"]:
        return {
            "status": UNVERIFIABLE,
            "price": None,
            "version": VARIANT_PRICE_VERSION,
            "method": "invalid_option_axes",
            "evidence": {
                "selection_errors": errors,
                "axis_collisions": indexed["collisions"],
            },
        }
    raw_selected = {
        value["source_axis"]: value["value"] for value in selected.values()
    }
    combination = _combination_price(product, raw_selected)
    if combination is not None:
        return {
            "status": PASS,
            "price": combination,
            "version": VARIANT_PRICE_VERSION,
            "method": "full_variant_combination",
            "evidence": {},
        }

    effective_axes = []
    all_prices = []
    for canonical_axis, axis in indexed["axes"].items():
        prices = {
            entry["price"]
            for entry in axis["values"].values()
            if entry["price"] is not None
        }
        all_prices.extend(prices)
        if len(prices) > 1:
            effective_axes.append(canonical_axis)
    if len(effective_axes) > 1:
        return {
            "status": UNVERIFIABLE,
            "price": None,
            "version": VARIANT_PRICE_VERSION,
            "method": "multiple_effective_price_axes",
            "evidence": {"effective_price_axes": effective_axes},
        }
    if len(effective_axes) == 1:
        price_axis = effective_axes[0]
        selection = selected.get(price_axis)
        axis = indexed["axes"][price_axis]
        if selection is None:
            return {
                "status": UNVERIFIABLE,
                "price": None,
                "version": VARIANT_PRICE_VERSION,
                "method": "effective_price_axis_unselected",
                "evidence": {"effective_price_axis": price_axis},
            }
        entry = axis["values"].get(normalize_option_text(selection["value"]))
        price = entry["price"] if entry else None
        if price is None:
            return {
                "status": UNVERIFIABLE,
                "price": None,
                "version": VARIANT_PRICE_VERSION,
                "method": "selected_variant_price_missing",
                "evidence": {"effective_price_axis": price_axis},
            }
        return {
            "status": PASS,
            "price": price,
            "version": VARIANT_PRICE_VERSION,
            "method": "unique_effective_price_axis",
            "evidence": {"effective_price_axis": price_axis},
        }

    unique_prices = {price for price in all_prices if price is not None}
    if len(unique_prices) == 1:
        return {
            "status": PASS,
            "price": next(iter(unique_prices)),
            "version": VARIANT_PRICE_VERSION,
            "method": "constant_option_price",
            "evidence": {},
        }
    pricing = [
        price
        for price in (_finite_price(value) for value in product.get("pricing") or [])
        if price is not None
    ]
    if not indexed["axes"] and len(set(pricing)) == 1:
        return {
            "status": PASS,
            "price": pricing[0],
            "version": VARIANT_PRICE_VERSION,
            "method": "single_base_price",
            "evidence": {},
        }
    return {
        "status": UNVERIFIABLE,
        "price": None,
        "version": VARIANT_PRICE_VERSION,
        "method": "price_not_deterministic",
        "evidence": {"candidate_prices": sorted(unique_prices)},
    }
