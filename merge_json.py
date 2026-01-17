#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional


REQUIRED_KEYS = [
    "query_text",
    "query_image",
    "pos_text",
    "hard_negatives",
    "hard_negatives_scores",
]


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_item(item: Any, idx: int, file_path: str, strict: bool) -> List[str]:
    """
    返回该 item 的错误列表（空列表表示通过）
    """
    errors: List[str] = []

    if not isinstance(item, dict):
        return [f"[item#{idx}] 不是 dict，而是 {type(item).__name__}"]

    # 必要字段是否存在
    for k in REQUIRED_KEYS:
        if k not in item:
            errors.append(f"[item#{idx}] 缺少必需键: {k}")

    # 如果缺必要字段就没必要继续深挖
    if errors:
        return errors

    # query_text
    if not isinstance(item["query_text"], str):
        errors.append(f"[item#{idx}] query_text 不是 str，而是 {type(item['query_text']).__name__}")

    # query_image
    if not isinstance(item["query_image"], str) or not item["query_image"].strip():
        errors.append(f"[item#{idx}] query_image 不是非空 str")

    # pos_text
    if not isinstance(item["pos_text"], str):
        errors.append(f"[item#{idx}] pos_text 不是 str，而是 {type(item['pos_text']).__name__}")

    # hard_negatives: 期望是 list，且每个元素是长度为2的 list/tuple（[text, image]）
    hn = item["hard_negatives"]
    if not isinstance(hn, list):
        errors.append(f"[item#{idx}] hard_negatives 不是 list，而是 {type(hn).__name__}")
    else:
        for j, pair in enumerate(hn):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                errors.append(f"[item#{idx}] hard_negatives[{j}] 不是长度为2的 list/tuple")
                continue
            if not isinstance(pair[0], str):
                errors.append(f"[item#{idx}] hard_negatives[{j}][0] 不是 str")
            # 第二个一般是图片路径/URL（你示例中是 "..."），通常也应该是 str
            if strict and not isinstance(pair[1], str):
                errors.append(f"[item#{idx}] hard_negatives[{j}][1] 不是 str")

    # hard_negatives_scores: list[float] 且长度与 hard_negatives 一致
    hns = item["hard_negatives_scores"]
    if not isinstance(hns, list):
        errors.append(f"[item#{idx}] hard_negatives_scores 不是 list，而是 {type(hns).__name__}")
    else:
        for j, s in enumerate(hns):
            if not is_number(s):
                errors.append(f"[item#{idx}] hard_negatives_scores[{j}] 不是数字 (int/float)")

    if isinstance(hn, list) and isinstance(hns, list):
        if len(hn) != len(hns):
            errors.append(
                f"[item#{idx}] hard_negatives 长度({len(hn)}) 与 hard_negatives_scores 长度({len(hns)})不一致"
            )

    # 可选字段：query_pos_scores（你示例有），如果存在则校验为数字
    if "query_pos_scores" in item and not is_number(item["query_pos_scores"]):
        errors.append(f"[item#{idx}] query_pos_scores 存在但不是数字")

    # 可选字段：pos_image（你示例有），如果 strict 则校验为 str
    if strict and "pos_image" in item and not isinstance(item["pos_image"], str):
        errors.append(f"[item#{idx}] pos_image 存在但不是 str")

    return errors


def validate_file(file_path: str, strict: bool) -> Tuple[bool, List[str], Optional[List[Dict[str, Any]]]]:
    """
    返回：(是否通过, 错误列表, 数据(list[dict])或None)
    """
    errors: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"JSON 读取/解析失败: {e}"], None

    if not isinstance(data, list):
        return False, [f"顶层不是 list，而是 {type(data).__name__}"], None

    for idx, item in enumerate(data):
        item_errors = validate_item(item, idx, file_path, strict)
        errors.extend(item_errors)

    if errors:
        return False, errors, None

    # 通过：确保类型是 list[dict]
    return True, [], data  # type: ignore


def dedup_by_query_image(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    按 query_image 去重，重复只保留第一个
    返回：(去重后列表, 去掉数量)
    """
    seen = set()
    kept: List[Dict[str, Any]] = []
    removed = 0

    for it in items:
        qi = it.get("query_image", None)
        # 如果 query_image 异常（理论上不会，因为验证过），也给它一个兜底策略：当作唯一
        if not isinstance(qi, str):
            kept.append(it)
            continue

        if qi in seen:
            removed += 1
            continue
        seen.add(qi)
        kept.append(it)

    return kept, removed


def iter_json_files(folder: str) -> List[str]:
    files = []
    for root, _, fnames in os.walk(folder):
        for name in fnames:
            if name.lower().endswith(".json"):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def main():
    ap = argparse.ArgumentParser(
        description="校验并合并若干 JSON 列表文件，按 query_image 去重后保存"
    )
    ap.add_argument("--input_dir", default="/.../... ", help="包含若干 .json 文件的文件夹路径（会递归遍历）")
    ap.add_argument("--output_json", default="/.../train_data_Qwen3VL_8B_scores.json", help="输出合并去重后的 json 路径")
    ap.add_argument("--strict", action="store_true", default=True, help="更严格校验：hard_negatives 的第二项、pos_image 等也必须是字符串")
    args = ap.parse_args()

    input_dir = args.input_dir
    output_json = args.output_json
    strict = args.strict

    if not os.path.isdir(input_dir):
        raise SystemExit(f"输入路径不是文件夹: {input_dir}")

    json_files = iter_json_files(input_dir)
    if not json_files:
        raise SystemExit(f"在该目录下未找到任何 .json 文件: {input_dir}")

    print(f"发现 {len(json_files)} 个 JSON 文件，开始校验... strict={strict}")

    valid_datas: List[List[Dict[str, Any]]] = []
    bad_files: List[Tuple[str, List[str]]] = []

    for fp in json_files:
        ok, errs, data = validate_file(fp, strict=strict)
        if not ok:
            bad_files.append((fp, errs))
            print(f"\n[格式错误] {fp}")
            # 只展示前若干条，避免刷屏
            max_show = 30
            for e in errs[:max_show]:
                print("  -", e)
            if len(errs) > max_show:
                print(f"  ... 还有 {len(errs) - max_show} 条错误未展示")
        else:
            valid_datas.append(data or [])
            print(f"[OK] {fp}  items={len(data or [])}")

    if bad_files:
        print("\n==================== 校验结果：存在不合格文件 ====================")
        print(f"合格文件数: {len(valid_datas)}")
        print(f"不合格文件数: {len(bad_files)}")
        print("不合格文件列表：")
        for fp, _ in bad_files:
            print(" -", fp)
        print("=================================================================")
    else:
        print("\n==================== 校验结果：全部合格 ✅ ====================")

    if not valid_datas:
        raise SystemExit("没有任何合格的 JSON 文件可合并，已退出。")

    # 合并（按文件顺序、列表顺序依次拼接）
    merged: List[Dict[str, Any]] = []
    for d in valid_datas:
        merged.extend(d)

    merged_total = len(merged)
    print(f"\n合并完成：合并后总元素数量 = {merged_total}")

    # 去重（按 query_image）
    deduped, removed = dedup_by_query_image(merged)
    kept = len(deduped)

    print("\n==================== 去重统计（按 query_image） ====================")
    print(f"合并后总元素数量: {merged_total}")
    print(f"去掉的元素数量:   {removed}")
    print(f"保留的元素数量:   {kept}")
    print("===================================================================")

    # 保存
    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到: {output_json}")


if __name__ == "__main__":
    main()
