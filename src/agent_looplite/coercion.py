"""工具参数的类型宽容转换，对应 Pi 在校验前做的转换（pi-schema 的 `coerce`）。

模型常把数字写成字符串、把单个值当成数组、给可选参数传 `null`。校验前按参数的 JSON Schema 把
这些值转成 schema 要求的类型。转不了的值保持原样，由校验报错回传给模型。规则：

- 字符串转整数、数字、布尔值；数字和布尔值转字符串。
- 单个值转成只有这一项的数组。
- 可选属性的 `null` 当作没传，前提是这个属性的 schema 不接受 `null`。
- `const`、`enum` 取转换后恰好相等的值；`anyOf`、`oneOf` 取第一个转换后能通过的分支。
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from typing import Any

from jsonschema import Draft202012Validator

_UNCHANGED: Any = object()
_INTEGER = re.compile(r"[+-]?[0-9]+")
_NUMBER = re.compile(r"[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?")


def coerce_arguments(arguments: Any, validator: Draft202012Validator) -> Any:
    """`arguments` 必须是调用方自己的副本：没有变化的部分原样返回，不再复制。"""
    return _Coercer(validator).coerce(validator.schema, arguments, (), False)


class _Coercer:
    def __init__(self, validator: Draft202012Validator) -> None:
        self._validator = validator

    def coerce(self, schema: Any, value: Any, refs: tuple[str, ...], wrapped: bool) -> Any:
        """`refs` 是同一个值已经展开过的 `$ref`，schema 自引用时据此停止展开。

        `wrapped` 表示这个值刚被包成数组的一项，不再包第二次。
        """
        if not isinstance(schema, dict):
            return value
        ref = schema.get("$ref")
        if isinstance(ref, str):
            target = self._resolve(ref)
            if target is None or ref in refs:
                return value
            return self.coerce(target, value, (*refs, ref), wrapped)
        if "const" in schema:
            converted = _to_literal(schema["const"], value)
            return value if converted is _UNCHANGED else converted
        if isinstance(schema.get("enum"), list):
            return _to_enum(schema["enum"], value)
        for keyword in ("anyOf", "oneOf"):
            if isinstance(schema.get(keyword), list):
                value = self._union(schema[keyword], value, refs, wrapped)
        for member in schema.get("allOf") or ():
            value = self.coerce(member, value, refs, wrapped)

        types = _types(schema)
        if types and not any(_matches(value, type_) for type_ in types):
            if types == ["array"]:
                if wrapped or "prefixItems" in schema:
                    return value
                return [self.coerce(schema.get("items"), value, refs, True)]
            for type_ in types:
                converted = _convert(type_, value)
                if converted is not _UNCHANGED:
                    value = converted
                    break
        if "object" in types and isinstance(value, dict):
            return self._properties(schema, value)
        if "array" in types and isinstance(value, list) and "prefixItems" not in schema:
            return [self.coerce(schema.get("items"), item, (), False) for item in value]
        return value

    def _union(self, branches: list[Any], value: Any, refs: tuple[str, ...], wrapped: bool) -> Any:
        if any(self._passes(branch, value) for branch in branches):
            return value
        for branch in branches:
            candidate = self.coerce(branch, value, refs, wrapped)
            if self._passes(branch, candidate):
                return candidate
        return value

    def _properties(self, schema: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        additional = schema.get("additionalProperties")
        required = set(schema.get("required") or ())
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in properties:
                if item is None and key not in required and not self._passes(properties[key], None):
                    continue
                result[key] = self.coerce(properties[key], item, (), False)
            elif isinstance(additional, dict):
                result[key] = self.coerce(additional, item, (), False)
            else:
                result[key] = item
        return result

    def _passes(self, schema: Any, value: Any) -> bool:
        return self._validator.evolve(schema=schema).is_valid(value)

    def _resolve(self, ref: str) -> Any:
        """只解析指向本 schema 内部的引用，例如 pydantic 生成的 `#/$defs/Item`。"""
        if not ref.startswith("#"):
            return None
        node: Any = self._validator.schema
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                return None
        return node


def _types(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [type_ for type_ in declared if isinstance(type_, str)]
    if "properties" in schema:
        return ["object"]
    if "items" in schema or "prefixItems" in schema:
        return ["array"]
    return []


def _matches(value: Any, type_: str) -> bool:
    if type_ == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if type_ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_ == "boolean":
        return isinstance(value, bool)
    if type_ == "string":
        return isinstance(value, str)
    if type_ == "null":
        return value is None
    if type_ == "array":
        return isinstance(value, list)
    if type_ == "object":
        return isinstance(value, dict)
    return True


def _convert(type_: str, value: Any) -> Any:
    """转成 `type_` 的值，转不了时返回 `_UNCHANGED`。"""
    if type_ in ("integer", "number") and isinstance(value, str):
        text = value.strip()
        if not _NUMBER.fullmatch(text):
            return _UNCHANGED
        number = float(text)
        if not math.isfinite(number):
            return _UNCHANGED
        if type_ == "number" and not _INTEGER.fullmatch(text):
            return number
        # 是不是整数按十进制原文判断：浮点数会丢精度，把 "9007199254740993.0" 变成另一个整数。
        # Decimal 只存有效数字和指数，"1e-9999999" 这样的原文也不用构造很大的数。
        exact = Decimal(text)
        return int(exact) if exact == exact.to_integral_value() else _UNCHANGED
    if type_ == "boolean" and isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().lower(), _UNCHANGED)
    if type_ == "string" and isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return _UNCHANGED
        return json.dumps(value)
    return _UNCHANGED


def _same(a: Any, b: Any) -> bool:
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _to_literal(literal: Any, value: Any) -> Any:
    if _same(literal, value):
        return value
    if isinstance(literal, bool):
        converted = _convert("boolean", value)
    elif isinstance(literal, (int, float)):
        converted = _convert("number", value)
    elif isinstance(literal, str):
        converted = _convert("string", value)
    else:
        return _UNCHANGED
    return literal if converted is not _UNCHANGED and _same(literal, converted) else _UNCHANGED


def _to_enum(members: list[Any], value: Any) -> Any:
    if any(_same(member, value) for member in members):
        return value
    for member in members:
        converted = _to_literal(member, value)
        if converted is not _UNCHANGED:
            return converted
    return value
