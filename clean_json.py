#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import Any, Dict, List, Optional


def iter_image_paths(item: Dict[str, Any]) -> List[str]:
    """收集条目中所有需要检查的 image 路径。"""
    paths: List[str] = []

    # query_image / pos_image
    for k in ("query_image", "pos_image"):
        v = item.get(k, None)
        if isinstance(v, str) and v:
            paths.append(v)

    # hard_negatives: list of [text, image_path]
    hn = item.get("hard_negatives", None)
    if isinstance(hn, list):
        for pair in hn:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                p2 = pair[1]
                if isinstance(p2, str) and p2:
                    paths.append(p2)

    return paths


def has_bad_path(item: Dict[str, Any], bad_substr: str) -> bool:
    for p in iter_image_paths(item):
        if bad_substr in p:
            return True
    return False


def hard_neg_scores_all_one(item: Dict[str, Any]) -> bool:
    """hard_negatives_scores 全为 1.0 则返回 True；缺失/非list/空list 返回 False。"""
    scores = item.get("hard_negatives_scores", None)
    if not isinstance(scores, list) or len(scores) == 0:
        return False

    # 允许元素是 int/float/str，统一转 float 判断
    try:
        return all(float(s) == 1.0 for s in scores)
    except (ValueError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Filter json list items by bad image paths and hard_negatives_scores."
    )
    ap.add_argument("--input", "-i", default="/.../..." help="输入 JSON 文件路径（顶层为 list）")
    ap.add_argument("--output", "-o", default="/../..", help="输出 JSON 文件路径")
    ap.add_argument(
        "--bad_substr",
        default="train-image/train",
        help='路径包含该子串则删除条目（默认: "train-image/train"）',
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=2,
        help="输出 JSON 缩进（默认 2）",
    )
    args = ap.parse_args()

    # 读取
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层不是 list，请确认格式。")

    before = len(data)
    error_path = 0
    error_case = 0

    kept: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            # 非 dict 的条目：你没提怎么处理，这里选择直接保留
            kept.append(item)
            continue

        # 规则1：bad path
        if has_bad_path(item, args.bad_substr):
            error_path += 1
            continue

        # 规则2：scores 全 1.0
        if hard_neg_scores_all_one(item):
            error_case += 1
            continue

        kept.append(item)

    after = len(kept)
    deleted = before - after

    # 写出
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=args.indent)

    # 打印统计
    print(f"input:  {args.input}")
    print(f"output: {args.output}")
    print(f"bad_substr: {args.bad_substr}")
    print("-" * 60)
    print(f"处理前条目数: {before}")
    print(f"error_path 删除数: {error_path}")
    print(f"error_case 删除数: {error_case}")
    print(f"总计删掉条目数: {deleted}")
    print(f"剩余条目数: {after}")


if __name__ == "__main__":
    main()
