# -*- coding: utf-8 -*-
"""pydantic 模型 -> JSON Schema 的单一出口，供实验与测试保证声明零漂移。"""
from __future__ import annotations

from typing import Any, Type


def schema_of_model(model: Type) -> dict[str, Any]:
    """pydantic 模型的 JSON Schema（去除 title 噪声）。"""
    raw = model.model_json_schema()
    raw.pop("title", None)
    for property_schema in (raw.get("properties") or {}).values():
        property_schema.pop("title", None)
    return raw
