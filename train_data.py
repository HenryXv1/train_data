import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
parser.add_argument("--raw_start", type=int, default=0)
parser.add_argument("--raw_end", type=int, default=-1)
parser.add_argument("--out_json", default="")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import random
import faiss
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F

from transformers import AutoModel, CLIPImageProcessor

# ============================================
#                路径配置
# ============================================
TRAIN_JSON = "/data5/llm/xhr/project/unimev2-train-data/train.json"

# query/pos 图像前缀（train.json 的 related_images）
TRAIN_IMAGE_PREFIX = "/data5/llm/dataset/"

# FAISS 检索库
FAISS_INDEX_PATH = "/home/notebook/code/group/zhengxianwu/Project-new/FAISS/eva-clip/shared_index/index.faiss"
FAISS_META_PATH = "/home/notebook/code/group/zhengxianwu/Project-new/FAISS/eva-clip/shared_index/index.json"

# 候选 image_url -> 本地相对路径 映射
URL2IMAGE_JSON = "/data2/llm/wzx/Project2/ThinkVQA/data/database_image/url2image.json"
CAND_IMAGE_PREFIX = "/data4/llm/wzx/dataset/"  # 拼接 url2image.json 的 value

# 输出
OUT_JSON_BASE = "/data5/llm/xhr/project/unimev2-train-data/unimev2-train-data.json"

def _with_range_suffix(path: str, start: int, end: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_{start}_{end}{ext}"

# ============================================
#                超参数
# ============================================
TOPN_RETRIEVE = 50
BETA = 0.001  # α = s_pos - β  :contentReference[oaicite:1]{index=1}
CYCLIC_STEP = 5  # 五步间隔采样 :contentReference[oaicite:2]{index=2}
MIN_HARD_NEG = 10  # 不足十个复制补足 :contentReference[oaicite:3]{index=3}
FALLBACK_SCORE = 1.0  # 全空(<1%)时随机取10个并赋1.0 :contentReference[oaicite:4]{index=4}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 训练数据里统一的“文本字段”（你给的目标格式）
PAIR_TEXT = "<|image_1|>\nRepresent the given image.\n"

# ============================================
#                加载 FAISS
# ============================================
print("Loading FAISS index ...")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(FAISS_META_PATH, "r") as f:
    kb_meta = json.load(f)  # 与 index size 对齐；每项含 image_url 等

# ============================================
#       加载 URL -> image 相对路径 映射
# ============================================
print("Loading url2image.json ...")
with open(URL2IMAGE_JSON, "r") as f:
    url2image = json.load(f)

# ============================================
#        初始化 encoder（EVA-CLIP-8B）
# ============================================

def load_eva_clip_image_encoder(device: str = "cuda"):
    model = AutoModel.from_pretrained(
        "BAAI/EVA-CLIP-8B",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if hasattr(model, "text_model"):
        del model.text_model
    if hasattr(model, "text_projection"):
        del model.text_projection

    model.to(device).eval()

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    return model, processor


encoder_model, encoder_processor = load_eva_clip_image_encoder("cuda")


def encode_image(image_path: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        print(f"[Warning] Image open failed: {image_path}")
        return None

    inputs = encoder_processor(images=image, return_tensors="pt").pixel_values.to("cuda").half()

    with torch.no_grad():
        image_embedding = encoder_model.encode_image(inputs)
        image_embedding = F.normalize(image_embedding, dim=-1)
        return image_embedding.detach().float().cpu().numpy()

# ============================================
#        初始化 Judge（Qwen3-VL-8B-Instruct）
# ============================================
# 说明：不同 transformers/模型实现类可能不同，这里做“兼容式加载”
from transformers import AutoProcessor

JUDGE_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"  # 可换成本地路径
JUDGE_DEVICE = "cuda"

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

def load_judge(model_path: str):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,   # 或 torch.float16
        device_map="auto",
        attn_implementation="sdpa",   # 可选
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


judge_model, judge_processor = load_judge(JUDGE_MODEL_PATH)

def _pick_single_token_id(tokenizer, s: str):
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # 尝试带空格（很多 tokenizer 会把 " Yes" 作为单 token）
    ids2 = tokenizer.encode(" " + s, add_special_tokens=False)
    if len(ids2) == 1:
        return ids2[0]
    # 退化：取第一个 token（严格复现角度不完美，但能跑；如需严格单 token，你要换 prompt/分词策略）
    print(f"[Warn] '{s}' is not single-token. ids={ids}, ids2={ids2}. fallback to first.")
    return ids2[0] if len(ids2) > 0 else ids[0]

# 从 processor 拿 tokenizer
judge_tokenizer = judge_processor.tokenizer if hasattr(judge_processor, "tokenizer") else None
assert judge_tokenizer is not None, "judge_processor 没有 tokenizer，确认 transformers/模型是否正确。"

YES_ID = _pick_single_token_id(judge_tokenizer, "Yes")
NO_ID  = _pick_single_token_id(judge_tokenizer, "No")

# 论文用的 judge 指令模板（Yes/No） :contentReference[oaicite:5]{index=5}
JUDGE_PROMPT = (
    "I will provide you with a query and a candidate.\n"
    "Please evaluate whether the candidate meets the requirements of the query. "
    "If it does, respond with 'Yes'; if it doesn't, respond with 'No'. "
    "Query:<|image_1|>, Candidates:<|image_2|>."
)

@torch.no_grad()
def judge_scores_for_candidates(query_img_path: str, cand_img_paths: list, batch_size: int = 4):
    """
    用 Qwen3-VL 的 chat_template 正确插入 image tokens，然后取最后一个位置 logits 做 Yes/No 概率
    s = exp(Yes) / (exp(Yes) + exp(No))
    """
    scores = []

    def _forward_one_batch(q_path, c_paths):
        # 构造一个 batch 的 conversations：每条样本有两张图（query + candidate）
        conversations = []
        for c_path in c_paths:
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I will provide you with a query and a candidate.\nQuery:"},
                        {"type": "image", "image": q_path},
                        {"type": "text", "text": "Candidate:"},
                        {"type": "image", "image": c_path},
                        {"type": "text", "text":
                            "Please evaluate whether the candidate meets the requirements of the query. "
                            "If it does, respond with 'Yes'; if it doesn't, respond with 'No'."
                        },
                    ],
                }
            ])

        inputs = judge_processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,          # 补齐 batch 内长度
            truncation=True,       # 防止极端长输入
        )

        inputs.pop("token_type_ids", None)
        inputs = inputs.to(next(judge_model.parameters()).device)

        out = judge_model(**inputs, use_cache=False)
        logits = out.logits  # [B, L, V]

        attn = inputs.get("attention_mask", None)
        last_pos = (attn.sum(dim=1).long() - 1) if attn is not None else torch.full(
            (logits.size(0),), logits.size(1) - 1, device=logits.device, dtype=torch.long
        )

        last_logits = logits.gather(
            1, last_pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        ).squeeze(1)  # [B, V]

        yes_logit = last_logits[:, YES_ID]
        no_logit  = last_logits[:, NO_ID]

        two = torch.stack([yes_logit, no_logit], dim=-1)   # [B, 2]
        prob_yes = torch.softmax(two, dim=-1)[:, 0]        # [B]
        return prob_yes.float().cpu().tolist()

    # 分 batch 处理
    for i in range(0, len(cand_img_paths), batch_size):
        batch_paths = cand_img_paths[i:i+batch_size]

        # 过滤打不开的图（防止 apply_chat_template 内部 load_image 崩）
        ok_paths = []
        ok_mask = []
        for p in batch_paths:
            try:
                Image.open(p).close()
                ok_paths.append(p)
                ok_mask.append(True)
            except Exception:
                ok_mask.append(False)

        if len(ok_paths) == 0:
            scores.extend([0.0] * len(batch_paths))
            continue

        batch_scores = _forward_one_batch(query_img_path, ok_paths)

        # 把分数按原顺序塞回去
        it = iter(batch_scores)
        for m in ok_mask:
            scores.append(next(it) if m else 0.0)

    return scores

def normalize_image_url(u: str) -> str:
    """
    meta 里的 image_url 可能是 https:\/\/... 或包含多余反斜杠，统一规整成 https://...
    """
    if u is None:
        return ""
    # 常见：把 \"\/\" 还原
    u = u.replace("\\/", "/")
    # 去掉可能残留的反斜杠
    u = u.replace("\\", "")
    return u

# ============================================
#        构造“wikiurl -> 有序去重图片列表”
# ============================================
print("Loading train.json ...")
with open(TRAIN_JSON, "r") as f:
    train_data_raw = json.load(f)

raw_total = len(train_data_raw)
raw_start = max(0, int(args.raw_start))
raw_end = raw_total if int(args.raw_end) < 0 else min(raw_total, int(args.raw_end))
if raw_end < raw_start:
    raw_end = raw_start

if args.out_json:
    OUT_JSON = args.out_json
else:
    OUT_JSON = _with_range_suffix(OUT_JSON_BASE, raw_start, raw_end)

# ===== 按 related_images 去重 =====
# 规则：相同 related_images 只保留第一次出现的条目
seen_rel = set()
train_data = []
dup_cnt = 0

for i, it in enumerate(train_data_raw):
    if i >= raw_end:
        break
    rel = it.get("related_images", "")
    if not rel:
        continue
    if rel in seen_rel:
        dup_cnt += 1
        continue
    seen_rel.add(rel)
    if i < raw_start:
        continue
    train_data.append(it)

print(f"[Dedup] raw_total={raw_total} range=[{raw_start},{raw_end}) unique_by_related_images={len(train_data)} removed={dup_cnt}")

wiki2imgs = {}
for it in train_data:
    wiki = it.get("wikipedia_url", "")
    rel  = it.get("related_images", "")
    if wiki == "" or rel == "":
        continue
    wiki2imgs.setdefault(wiki, set()).add(rel)

# 去重 + 字典序排序
wiki2imgs_sorted = {k: sorted(list(v)) for k, v in wiki2imgs.items()}

def get_pos_image_rel(wiki: str, query_rel: str) -> str:
    """
    同一 wikiurl 下多张图：去重、排序后，pos 是“下一张循环”
    仅一张则自身为正样本
    """
    lst = wiki2imgs_sorted.get(wiki, [])
    if len(lst) == 0:
        return query_rel
    if len(lst) == 1:
        return lst[0]
    try:
        idx = lst.index(query_rel)
    except ValueError:
        # 如果 query_rel 不在去重集合里（极少），则用第一张
        idx = 0
    return lst[(idx + 1) % len(lst)]

# ============================================
#        主循环：构造 UniME-V2 训练样本
# ============================================
print("Start building UniME-V2 training data ...")

# 用“流式写 json 数组”的方式避免爆内存
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
out_f = open(OUT_JSON, "w", encoding="utf-8")
out_f.write("[\n")
first = True

for item in tqdm(train_data):
    wiki = item.get("wikipedia_url", "")
    query_rel = item.get("related_images", "")
    if wiki == "" or query_rel == "":
        continue

    query_abs = os.path.join(TRAIN_IMAGE_PREFIX, query_rel)

    # ---- 正样本（同 wikiurl 下“下一张循环”）
    pos_rel = get_pos_image_rel(wiki, query_rel)
    pos_abs = os.path.join(TRAIN_IMAGE_PREFIX, pos_rel)

    # ---- 先算 query embedding -> FAISS top50
    q_emb = encode_image(query_abs)
    if q_emb is None:
        continue

    D, I = index.search(q_emb.astype(np.float32), TOPN_RETRIEVE)  # I: [1,50]
    idxs = I[0].tolist()

    # ---- 从 meta 拿 image_url -> url2image -> candidate_abs
    cand_abs_list = []
    for idx in idxs:
        meta = kb_meta[idx]
        raw_url = meta.get("image_url", "")
        url = normalize_image_url(raw_url)
        rel_path = url2image.get(url, None)
        if rel_path is None:
            # 找不到映射就跳过（后面不足会补）
            continue
        cand_abs = os.path.join(CAND_IMAGE_PREFIX, rel_path)
        cand_abs_list.append(cand_abs)

    # 如果因为映射缺失导致候选过少，也允许继续（后面兜底）
    if len(cand_abs_list) == 0:
        # 没候选就直接进入“全空兜底”：随机取10个（但此时也没法随机）
        continue

    # ---- Judge 打分：先算 pos 分数，再算 50 个候选分数
    # 论文：用 Yes/No logits 算语义匹配分数 :contentReference[oaicite:7]{index=7}
    pos_score = judge_scores_for_candidates(query_abs, [pos_abs], batch_size=1)[0]

    cand_scores = judge_scores_for_candidates(query_abs, cand_abs_list, batch_size=50)

    # ---- 阈值过滤：α = s_pos - 0.01，分数 > α 的候选认为“太像(可能假负)”删掉 :contentReference[oaicite:8]{index=8}
    alpha = float(pos_score) - float(BETA)

    filtered = []
    for p, s in zip(cand_abs_list, cand_scores):
        if float(s) <= alpha:
            filtered.append((p, float(s)))

    # ---- 五步间隔采样（cyclical sampling with five-step intervals） :contentReference[oaicite:9]{index=9}
    sampled = []
    for j in range(0, len(filtered), CYCLIC_STEP):
        sampled.append(filtered[j])

    # ---- 不足 10：复制补足 :contentReference[oaicite:10]{index=10}
    if len(sampled) > 0 and len(sampled) < MIN_HARD_NEG:
        k = 0
        while len(sampled) < MIN_HARD_NEG:
            sampled.append(sampled[k % len(sampled)])
            k += 1

    # ---- 全空：随机从初始候选池取 10 个，并把分数设为 1.0 :contentReference[oaicite:11]{index=11}
    if len(sampled) == 0:
        pool = list(zip(cand_abs_list, cand_scores))
        # 如果 pool < 10，就允许重复采样
        if len(pool) >= MIN_HARD_NEG:
            sampled = random.sample(pool, MIN_HARD_NEG)
        else:
            sampled = [random.choice(pool) for _ in range(MIN_HARD_NEG)]
        sampled = [(p, FALLBACK_SCORE) for (p, _) in sampled]

    # ---- 截断/保证恰好 10 个
    sampled = sampled[:MIN_HARD_NEG]

    hard_negs = [[PAIR_TEXT, p] for (p, _) in sampled]
    hard_scores = [float(s) for (_, s) in sampled]

    record = {
        "query_text": PAIR_TEXT,
        "query_image": query_abs,
        "pos_text": PAIR_TEXT,
        "pos_image": pos_abs,
        "query_pos_scores": float(pos_score),
        "hard_negatives": hard_negs,
        "hard_negatives_scores": hard_scores
    }

    if not first:
        out_f.write(",\n")
    out_f.write(json.dumps(record, ensure_ascii=False))
    first = False

out_f.write("\n]\n")
out_f.close()

print(f"Done! Saved to: {OUT_JSON}")
