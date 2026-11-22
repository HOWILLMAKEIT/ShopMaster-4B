"""Multiplicative purchase reward from ShopSimulator, Eq. (4) and Appendix C.4."""

from functools import lru_cache
from math import isfinite

from web_agent_site.engine.goal import _nlp, get_attribute_reward, get_option_reward

PAPER_STRICT_MODE = "paper_strict"
PAPER_STRICT_VERSION = "shopsimulator-paper-strict-v1"


@lru_cache(maxsize=4096)
def _title_keywords(title):
    return frozenset(
        token.text.casefold() for token in _nlp()(title)
        if token.pos_ in {"PNOUN", "NOUN", "PROPN"}
    )


def _category_reward(product, goal):
    # The public scorer compares catalog query labels. An absent label cannot match.
    query = goal.get("query")
    if query and product.get("query") == query:
        return 1.0
    product_categories = {c.strip() for c in (product.get("category") or "").split("›")}
    goal_categories = {c.strip() for c in (goal.get("category") or "").split("›")}
    if len((product_categories & goal_categories) - {""}) >= 2:
        return 1.0

    target_keywords = _title_keywords(goal["name"])
    observed_keywords = _title_keywords(product["title"])
    # Preserve the public scorer's convention for a target with no noun tokens.
    overlap = (
        len(target_keywords & observed_keywords) / len(target_keywords)
        if target_keywords else 0.2
    )
    if overlap > 0.2:
        return 1.0
    if overlap >= 0.1:
        return 0.5
    return 0.1 if overlap > 0 else 0.0


def score_purchase(product, goal, selected_options, price_resolution):
    matching_fields = {
        "Attributes": product["Attributes"],
        "Title": product["Title"],
        "BulletPoints": [point for point in product["BulletPoints"] if point],
        "Description": product["Description"] or "",
    }
    r_type = _category_reward(product, goal)
    r_att, matched_attributes = get_attribute_reward(matching_fields, goal)
    goal_options = goal["goal_options"]
    if isinstance(goal_options, dict):
        goal_options = list(goal_options.values())
    r_option, matched_options = get_option_reward(
        list(selected_options.values()), goal_options,
    )
    upper = goal["price_upper"]
    if upper is None or upper <= 0:
        r_price = 1.0
    else:
        price = price_resolution.get("price")
        price_is_known = (
            price_resolution.get("status") == "pass"
            and isinstance(price, (float, int))
            and isfinite(price)
            and price >= 0
        )
        r_price = float(price <= upper) if price_is_known else None

    r_strict = r_type * r_att * r_option * r_price if r_price is not None else None
    # Also report Eq. (3) so the existing evaluation command can label both scores.
    r_loose = (
        r_type * (matched_attributes + matched_options + r_price)
        / (len(goal["attributes"]) + len(goal_options) + 1)
        if r_price is not None else None
    )
    return {
        "r_type": r_type,
        "r_att": r_att,
        "r_option": r_option,
        "r_price": r_price,
        "paper_strict_score": r_strict,
        "paper_strict_success": r_strict == 1.0,
        "r_loose": r_loose,
    }


def terminal_reward_detail(termination_reason: str, purchase_scores=None):
    """Build the paper_strict terminal reward payload."""
    if purchase_scores is None:
        purchase_scores = {
            "r_type": 0.0,
            "r_att": 0.0,
            "r_option": 0.0,
            "r_price": 0.0,
            "paper_strict_score": 0.0,
            "paper_strict_success": False,
            "r_loose": 0.0,
        }
        reason = str(termination_reason)
    else:
        reason = "purchase"

    strict_score = purchase_scores["paper_strict_score"]
    return {
        **purchase_scores,
        "reward_version": PAPER_STRICT_VERSION,
        "reward_type": reason if strict_score is not None else "reward_unverifiable",
        "termination_reason": reason,
        "terminal_utility": strict_score if strict_score is not None else 0.0,
        "reward_valid": strict_score is not None,
        "purchase_success": purchase_scores["paper_strict_success"],
        "sampling_invalid": strict_score is None,
    }
