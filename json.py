#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def is_empty_related_images(val: Any) -> bool:
    """
    related_images 为空的判定：
    - None
    - 不存在该键
    - 不是 str
    - 等于空字符串 ""
    """
    if val is None:
        return True
    if not isinstance(val, str):
        return True
    if val == "":
        return True
    return False


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list, but got {type(data)}")
    return data


def save_json_list(path: str, data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dedup_by_related_images(
    items: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    返回：
    - 去重后的列表
    - 丢掉 related_images 为空的数量
    - 去重丢弃的数量（related_images 重复导致）

    """
    dropped_empty = 0
    dropped_dup = 0

    seen = set()
    kept: List[Dict[str, Any]] = []

    for it in items:
        # 如果键不存在，get 会返回 None
        rel = it.get("related_images", None)

        if is_empty_related_images(rel):
            dropped_empty += 1
            continue

        if rel in seen:
            dropped_dup += 1
            continue

        seen.add(rel)
        kept.append(it)

    return kept, dropped_empty, dropped_dup


def main():
    parser = argparse.ArgumentParser(
        description="Drop empty related_images, then dedup train.json by related_images (no normalization)."
    )
    parser.add_argument("--in_json", required=True, help="Path to input train.json")
    parser.add_argument("--out_json", required=True, help="Path to output deduped json")
    args = parser.parse_args()

    data = load_json_list(args.in_json)
    total = len(data)

    kept, dropped_empty, dropped_dup = dedup_by_related_images(data)
    kept_cnt = len(kept)

    save_json_list(args.out_json, kept)

    print("========== Done ==========")
    print(f"Input:   {args.in_json}")
    print(f"Output:  {args.out_json}")
    print("--------------------------")
    print(f"Total items:                     {total}")
    print(f"Dropped (empty related_images):  {dropped_empty}")
    print(f"Dropped (duplicate related_images): {dropped_dup}")
    print(f"Kept items:                      {kept_cnt}")
    print("===========================")


if __name__ == "__main__":
    main()
