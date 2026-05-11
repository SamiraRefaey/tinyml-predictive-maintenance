from __future__ import annotations


def quantize_int8(value: float, min_value: float, max_value: float) -> int:
    if max_value <= min_value:
        return 0
    scaled = round((value - min_value) / (max_value - min_value) * 255 - 128)
    return max(-128, min(127, scaled))


def quantize_row(row: dict[str, float], ranges: dict[str, tuple[float, float]]) -> dict[str, int]:
    return {key: quantize_int8(row[key], low, high) for key, (low, high) in ranges.items() if key in row}
