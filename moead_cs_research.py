#!/usr/bin/env python3
"""
MOEA/D × LLM：计算机科学长尾知识多目标进化系统
═══════════════════════════════════════════════════
算法：MOEA/D（基于分解的多目标演化算法）
LLM：负责知识生成、评分、交叉变异
目标（全部最大化）：
  f1 - 价值度   (Value)    : 对 CS 实践/研究的实际价值
  f2 - 影响力   (Impact)   : 对 CS 发展的长远影响深度
  f3 - 长尾度   (Long-tail): 知识的稀缺性/小众程度
"""

import json
import random
import os
import time
from dataclasses import dataclass, field
from itertools import product as iproduct
from typing import List, Tuple

import numpy as np
from openai import OpenAI

# ══════════════════════════════════════════════════════════════
#  超参数配置
# ══════════════════════════════════════════════════════════════
POP_SIZE    = 8      # 种群大小（子问题数量）
N_OBJ       = 3      # 目标数量
N_NEIGHBORS = 3      # 每个子问题的邻居数
N_GENS      = 10     # 演化代数
MODEL       = "deepseek-chat"

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════
@dataclass
class Individual:
    topic: str
    description: str
    scores: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # scores[0]=价值度, scores[1]=影响力, scores[2]=长尾度，范围 [0,1]

    @property
    def value(self):    return self.scores[0]
    @property
    def impact(self):   return self.scores[1]
    @property
    def longtail(self): return self.scores[2]


# ══════════════════════════════════════════════════════════════
#  权重向量生成（Das & Dennis 系统性方法）
# ══════════════════════════════════════════════════════════════
def _enum_weights(H: int, n_obj: int) -> List[List[float]]:
    """枚举所有满足 sum(w_i) = 1 的均匀权重向量（整数格点归一化）"""
    if n_obj == 1:
        return [[1.0]]
    result = []
    for i in range(H + 1):
        for rest in _enum_weights_int(H - i, n_obj - 1):
            result.append([(i + r) / H for i_val, r in [(i, 0)]] +
                          [r / H for r in rest])
    return result


def _enum_weights_int(H: int, n_obj: int) -> List[List[int]]:
    """递归枚举整数格点，返回 int 列表（未归一化）"""
    if n_obj == 1:
        return [[H]]
    result = []
    for i in range(H + 1):
        for rest in _enum_weights_int(H - i, n_obj - 1):
            result.append([i] + rest)
    return result


def generate_weight_vectors(n: int, n_obj: int = 3) -> np.ndarray:
    """生成 >= n 个均匀分布的权重向量，取前 n 个"""
    from math import comb
    H = 1
    while comb(H + n_obj - 1, n_obj - 1) < n:
        H += 1
    # 使用整数格点方式生成，避免除零
    raw = _enum_weights_int(H, n_obj)
    weights = [[v / H for v in row] for row in raw]
    random.seed(42)
    random.shuffle(weights)
    return np.array(weights[:n], dtype=float)


# ══════════════════════════════════════════════════════════════
#  邻居结构
# ══════════════════════════════════════════════════════════════
def compute_neighborhoods(weights: np.ndarray, T: int) -> List[List[int]]:
    """按权重向量欧氏距离计算每个子问题的 T 个最近邻"""
    n = len(weights)
    neighborhoods = []
    for i in range(n):
        dists = np.linalg.norm(weights - weights[i], axis=1)
        nearest = np.argsort(dists)[:T].tolist()
        neighborhoods.append(nearest)
    return neighborhoods


# ══════════════════════════════════════════════════════════════
#  Tchebycheff 标量化（最小化）
# ══════════════════════════════════════════════════════════════
def tchebycheff(scores: List[float], weight: np.ndarray,
                ideal: np.ndarray) -> float:
    s = np.array(scores)
    # 目标是最大化，所以 ideal - s 越小越好
    return float(np.max(weight * np.abs(ideal - s)))


# ══════════════════════════════════════════════════════════════
#  LLM 接口：初始种群生成
# ══════════════════════════════════════════════════════════════
def llm_generate_initial(n: int) -> List[Individual]:
    """批量生成 n 个 CS 长尾知识条目"""
    print(f"  [LLM] 生成初始 {n} 个 CS 长尾知识条目...")
    prompt = f"""你是计算机科学研究专家。请生成 {n} 个关于计算机科学的**长尾研究知识**条目。

"长尾知识"定义：小众、专业化、非主流，但潜在价值高、尚未被充分研究的知识点。
要求覆盖多样化领域：算法设计、系统架构、编程语言理论、AI 子方向、分布式系统、安全、形式化方法等。

请以如下 JSON 格式返回（只返回 JSON，不含其他文字）：
{{
  "items": [
    {{"topic": "简短主题名（≤15字）", "description": "详细描述（50-80字，说明是什么、为何重要、为何小众）"}},
    ...
  ]
}}"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.95,
    )
    data = json.loads(resp.choices[0].message.content)
    items = data.get("items", [])
    # 兼容不同 key 名
    if not items:
        for v in data.values():
            if isinstance(v, list):
                items = v
                break
    return [Individual(topic=it["topic"], description=it["description"])
            for it in items[:n]]


# ══════════════════════════════════════════════════════════════
#  LLM 接口：批量评分
# ══════════════════════════════════════════════════════════════
def llm_evaluate_batch(individuals: List[Individual]) -> List[List[float]]:
    """批量对所有个体的三个目标打分（0-10），返回归一化到 [0,1] 的分数列表"""
    items_text = "\n".join(
        f"{i+1}. 【{ind.topic}】\n   {ind.description}"
        for i, ind in enumerate(individuals)
    )
    prompt = f"""请对以下 {len(individuals)} 个计算机科学长尾知识条目进行客观评分。

{items_text}

评分标准（每项 0-10 分，整数）：
- value（价值度）：该知识对 CS 工程实践/学术研究的实际价值与有用程度
- impact（影响力）：如果被广泛采用，对 CS 领域发展的长远影响和深度
- longtail（长尾度）：知识的小众程度、稀缺性、非主流程度（完全主流=0，极度小众=10）

评分需客观，避免全部高分，应有区分度。

以 JSON 返回（只返回 JSON）：
{{"scores": [{{"value": 整数, "impact": 整数, "longtail": 整数}}, ...]}}
数组长度必须恰好为 {len(individuals)}。"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(resp.choices[0].message.content)
    raw = data["scores"]
    # 归一化到 [0,1]
    return [[s["value"] / 10.0, s["impact"] / 10.0, s["longtail"] / 10.0]
            for s in raw]


# ══════════════════════════════════════════════════════════════
#  LLM 接口：批量交叉变异（核心驱动）
# ══════════════════════════════════════════════════════════════
def llm_batch_crossover(
    pairs: List[Tuple[Individual, Individual, np.ndarray]]
) -> List[Individual]:
    """
    输入：(parent1, parent2, weight_vector) 的列表
    输出：对应的后代 Individual 列表
    LLM 在一次调用中同时完成所有交叉变异
    """
    tasks = []
    for idx, (p1, p2, w) in enumerate(pairs):
        # 根据权重向量确定演化偏向描述
        obj_names = ["价值度", "影响力", "长尾度"]
        dominant = obj_names[int(np.argmax(w))]
        tasks.append(
            f"任务{idx+1}:\n"
            f"  父代A: 【{p1.topic}】{p1.description}\n"
            f"  父代B: 【{p2.topic}】{p2.description}\n"
            f"  演化偏向: 重点提升「{dominant}」"
            f"（权重 价值={w[0]:.2f}, 影响={w[1]:.2f}, 长尾={w[2]:.2f}）"
        )

    tasks_text = "\n\n".join(tasks)
    prompt = f"""你是 CS 研究创新专家，负责知识进化。请对以下 {len(pairs)} 组父代知识执行交叉变异，产生新的长尾 CS 知识后代。

{tasks_text}

规则：
1. 融合两个父代的核心思想，产生**交叉创新**的新知识点
2. 后代必须保持长尾特性（小众、专业、非主流）
3. 根据演化偏向，使后代在对应目标上表现更优
4. 每个后代必须是独立的、不同于父代的新知识点

以 JSON 返回（只返回 JSON）：
{{"offspring": [
  {{"topic": "≤15字", "description": "50-80字"}},
  ...
]}}
数组长度必须恰好为 {len(pairs)}。"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.88,
    )
    data = json.loads(resp.choices[0].message.content)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it["description"])
            for it in items[:len(pairs)]]


# ══════════════════════════════════════════════════════════════
#  MOEA/D 一代演化
# ══════════════════════════════════════════════════════════════
def moead_one_generation(
    population: List[Individual],
    weights: np.ndarray,
    neighborhoods: List[List[int]],
    ideal_point: np.ndarray,
) -> Tuple[List[Individual], np.ndarray]:

    n = len(population)
    # ── 1. 为每个子问题选父代 ──
    pairs = []
    parent_indices = []
    for i in range(n):
        nb = neighborhoods[i]
        # 至少需要 2 个不同邻居
        if len(nb) >= 2:
            idx1, idx2 = random.sample(nb, 2)
        else:
            idx1 = nb[0]
            idx2 = random.randint(0, n - 1)
        pairs.append((population[idx1], population[idx2], weights[i]))
        parent_indices.append(i)

    # ── 2. 批量 LLM 交叉变异 ──
    offspring_list = llm_batch_crossover(pairs)

    # ── 3. 批量 LLM 评分 ──
    scores_batch = llm_evaluate_batch(offspring_list)
    for child, scores in zip(offspring_list, scores_batch):
        child.scores = scores

    # ── 4. 更新理想点 ──
    for child in offspring_list:
        ideal_point = np.maximum(ideal_point, child.scores)

    # ── 5. 邻居替换（Tchebycheff） ──
    new_pop = list(population)
    for i, child in enumerate(offspring_list):
        for j in neighborhoods[parent_indices[i]]:
            curr_val = tchebycheff(new_pop[j].scores, weights[j], ideal_point)
            child_val = tchebycheff(child.scores, weights[j], ideal_point)
            if child_val <= curr_val:
                new_pop[j] = child

    return new_pop, ideal_point


# ══════════════════════════════════════════════════════════════
#  结果展示
# ══════════════════════════════════════════════════════════════
BLOCK = "█"
HALF  = "▌"

def score_bar(val: float, width: int = 10) -> str:
    filled = int(val * width)
    return BLOCK * filled + "░" * (width - filled)

def print_generation(gen: int, population: List[Individual],
                     ideal_point: np.ndarray):
    label = "初始种群" if gen == 0 else f"第 {gen:2d} 代"
    avg_v = sum(x.value    for x in population) / len(population)
    avg_i = sum(x.impact   for x in population) / len(population)
    avg_l = sum(x.longtail for x in population) / len(population)

    print(f"\n{'─'*72}")
    print(f"  ◈ {label}  │  理想点 V={ideal_point[0]*10:.1f} "
          f"I={ideal_point[1]*10:.1f} L={ideal_point[2]*10:.1f}")
    print(f"  均值: 价值={avg_v*10:.2f}  影响={avg_i*10:.2f}  长尾={avg_l*10:.2f}")
    print(f"{'─'*72}")

    ranked = sorted(population, key=lambda x: sum(x.scores), reverse=True)
    for k, ind in enumerate(ranked):
        print(f"  {k+1}. {ind.topic}")
        print(f"     价值 {score_bar(ind.value)}  {ind.value*10:.1f}")
        print(f"     影响 {score_bar(ind.impact)}  {ind.impact*10:.1f}")
        print(f"     长尾 {score_bar(ind.longtail)}  {ind.longtail*10:.1f}")
        desc = ind.description
        print(f"     ↳ {desc[:65]}{'…' if len(desc)>65 else ''}")
        print()


def print_pareto_front(population: List[Individual]):
    """输出最终 Pareto 前沿"""
    pareto = []
    for i, ind in enumerate(population):
        dominated = any(
            all(other.scores[k] >= ind.scores[k] for k in range(N_OBJ)) and
            any(other.scores[k] >  ind.scores[k] for k in range(N_OBJ))
            for j, other in enumerate(population) if j != i
        )
        if not dominated:
            pareto.append(ind)

    print(f"\n{'═'*72}")
    print(f"  ★  最终 Pareto 最优解集（共 {len(pareto)} 个）")
    print(f"{'═'*72}\n")
    for ind in sorted(pareto, key=lambda x: sum(x.scores), reverse=True):
        print(f"  ◆ {ind.topic}")
        print(f"    价值={ind.value*10:.1f}  影响={ind.impact*10:.1f}  长尾={ind.longtail*10:.1f}  "
              f"综合={sum(ind.scores)*10/3:.1f}")
        print(f"    {ind.description}\n")


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════
def main():
    print("═" * 72)
    print("  MOEA/D × LLM ── 计算机科学长尾知识多目标进化")
    print("  目标: [价值度 × 影响力 × 长尾度]  │  演化 10 代")
    print(f"  模型: {MODEL}  │  种群: {POP_SIZE}  │  邻居: {N_NEIGHBORS}")
    print("═" * 72)

    # ── Step 0: 初始化 MOEA/D 结构 ──
    weights       = generate_weight_vectors(POP_SIZE, N_OBJ)
    neighborhoods = compute_neighborhoods(weights, N_NEIGHBORS)
    print(f"\n[初始化] 权重向量数={len(weights)}，邻居数={N_NEIGHBORS}")
    print("  权重向量（价值/影响/长尾）:")
    for i, w in enumerate(weights):
        print(f"    w{i}: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]  "
              f"邻居={neighborhoods[i]}")

    # ── Step 1: 生成初始种群 ──
    print("\n[Step 1/3] 生成初始 CS 长尾知识种群...")
    t0 = time.time()
    population = llm_generate_initial(POP_SIZE)
    print(f"  完成，耗时 {time.time()-t0:.1f}s")

    # ── Step 2: 评估初始种群 ──
    print("\n[Step 2/3] 评估初始种群...")
    t0 = time.time()
    init_scores = llm_evaluate_batch(population)
    for ind, sc in zip(population, init_scores):
        ind.scores = sc
    print(f"  完成，耗时 {time.time()-t0:.1f}s")

    # ── 初始化理想点 ──
    ideal_point = np.array([
        max(ind.scores[j] for ind in population)
        for j in range(N_OBJ)
    ])

    print_generation(0, population, ideal_point)

    # ── Step 3: 演化 10 代 ──
    print("\n[Step 3/3] 开始 MOEA/D 演化...\n")
    for gen in range(1, N_GENS + 1):
        print(f"[第 {gen:2d}/{N_GENS} 代] 交叉变异 + 评估...", end=" ", flush=True)
        t0 = time.time()
        population, ideal_point = moead_one_generation(
            population, weights, neighborhoods, ideal_point
        )
        elapsed = time.time() - t0
        avg = sum(sum(x.scores) for x in population) / (len(population) * N_OBJ)
        print(f"✓  耗时 {elapsed:.1f}s  │  平均综合分 {avg*10:.2f}")
        print_generation(gen, population, ideal_point)

    # ── 最终 Pareto 前沿 ──
    print_pareto_front(population)
    print("  演化完成！")


if __name__ == "__main__":
    main()
