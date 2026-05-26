"""
Agent Package — 通用工具函数

提供跨模块复用的工具函数，避免重复代码。
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_response(resp: str) -> dict[str, Any]:
    """从 LLM 响应中提取 JSON 对象"""
    json_match = re.search(r'\{[\s\S]*\}', resp)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


def parse_json_array(resp: str) -> list[Any]:
    """从 LLM 响应中提取 JSON 数组"""
    json_match = re.search(r'\[[\s\S]*\]', resp)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return []