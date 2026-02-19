#!/usr/bin/env python3
"""
MOEA/D × LLM：跨学科长尾科研知识多目标进化系统（扩展版）
════════════════════════════════════════════════════════
种群: 15   目标: 5   邻居: 5   演化: 10 代
领域: 不限学科（物理、生物、化学、数学、神经科学、材料、环境、社会等）

5 个目标（全部最大化，0-10分）:
  f1 - 知识价值  (Knowledge Value)    : 对基础科学理论的贡献深度
  f2 - 社会影响  (Social Impact)      : 对人类社会的长远影响
  f3 - 长尾度    (Long-tail)          : 研究稀缺性/小众程度
  f4 - 跨学科性  (Interdisciplinarity): 跨领域连接与融合潜力
  f5 - 前沿性    (Frontier)           : 新颖程度/突破已知边界
"""

import json, random, os, time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from openai import OpenAI

# ══════════════════════════════════════════════════════
POP_SIZE    = 15
N_OBJ       = 5
N_NEIGHBORS = 5
N_GENS      = 10
MODEL       = "deepseek-chat"
OBJ_NAMES   = ["知识价值", "社会影响", "长尾度", "跨学科性", "前沿性"]
OBJ_KEYS    = ["knowledge", "social", "longtail", "interdiscip", "frontier"]

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)


# ══════════════════════════════════════════════════════
@dataclass
class Individual:
    topic: str
    description: str
    domain: str = ""
    scores: List[float] = field(default_factory=lambda: [0.0] * N_OBJ)


# ══════════════════════════════════════════════════════
#  权重向量：Das & Dennis 格点法
# ══════════════════════════════════════════════════════
def _enum_int(H: int, d: int) -> List[List[int]]:
    if d == 1:
        return [[H]]
    result = []
    for i in range(H + 1):
        for rest in _enum_int(H - i, d - 1):
            result.append([i] + rest)
    return result

def generate_weight_vectors(n: int, d: int) -> np.ndarray:
    from math import comb
    H = 1
    while comb(H + d - 1, d - 1) < n:
        H += 1
    raw = _enum_int(H, d)
    weights = [[v / H for v in row] for row in raw]
    random.seed(42)
    random.shuffle(weights)
    return np.array(weights[:n], dtype=float)

def compute_neighborhoods(W: np.ndarray, T: int) -> List[List[int]]:
    n = len(W)
    return [np.argsort(np.linalg.norm(W - W[i], axis=1))[:T].tolist() for i in range(n)]

def tchebycheff(scores: List[float], w: np.ndarray, ideal: np.ndarray) -> float:
    return float(np.max(w * np.abs(ideal - np.array(scores))))


# ══════════════════════════════════════════════════════
#  LLM 接口
# ══════════════════════════════════════════════════════
def llm_generate_initial(n: int) -> List[Individual]:
    print(f"  [LLM] 生成初始 {n} 个跨学科长尾知识条目...")
    prompt = f"""你是跨学科科研顾问，精通物理、化学、生物、数学、神经科学、材料科学、环境科学、社会科学、经济学、天文学等领域。

请生成 {n} 个**跨学科长尾研究知识**条目。

"长尾科研知识"定义：
- 尚未被主流研究充分关注的小众研究方向
- 具有潜在颠覆性但目前处于早期探索阶段
- 可能是不同学科交叉产生的新兴领域
- 不限于任何单一学科，鼓励跨领域组合

要求：
- {n} 个条目需覆盖多个不同一级学科（物理/化学/生物/数学/材料/环境/神经/社会等）
- 每个条目必须具体、可研究，不能过于宽泛
- 体现知识的长尾特性：冷门、专业、非主流

以 JSON 返回（只返回 JSON）：
{{
  "items": [
    {{
      "topic": "简短主题名（≤18字）",
      "domain": "所属学科领域（如：量子生物学、计算神经科学等）",
      "description": "详细描述（60-100字，说明是什么、研究现状、为何是长尾、潜在价值）"
    }},
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
    if not items:
        for v in data.values():
            if isinstance(v, list): items = v; break
    return [Individual(topic=it["topic"], description=it["description"],
                       domain=it.get("domain", ""))
            for it in items[:n]]


def llm_evaluate_batch(individuals: List[Individual]) -> List[List[float]]:
    items_text = "\n".join(
        f"{i+1}. 【{ind.topic}】（{ind.domain}）\n   {ind.description}"
        for i, ind in enumerate(individuals)
    )
    prompt = f"""请对以下 {len(individuals)} 个跨学科研究知识条目进行五维客观评分。

{items_text}

评分标准（每项 0-10 整数分，需有区分度，不要全部高分）：
- knowledge（知识价值）：对基础科学理论的贡献深度，能否推进人类对自然/社会规律的理解
- social（社会影响）：若该研究取得突破，对人类社会、技术、环境的长远正向影响
- longtail（长尾度）：当前研究的稀缺性和小众程度（极主流=0，极小众=10）
- interdiscip（跨学科性）：与其他学科融合的潜力，跨越单一学科边界的程度
- frontier（前沿性）：知识的新颖度，突破已知边界的程度，与现有研究的差异化

以 JSON 返回（只返回 JSON，数组长度必须恰好为 {len(individuals)}）：
{{"scores": [
  {{"knowledge": 整数, "social": 整数, "longtail": 整数, "interdiscip": 整数, "frontier": 整数}},
  ...
]}}"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(resp.choices[0].message.content)
    raw = data["scores"]
    return [[s["knowledge"]/10, s["social"]/10, s["longtail"]/10,
             s["interdiscip"]/10, s["frontier"]/10] for s in raw]


def llm_batch_crossover(pairs: List[Tuple[Individual, Individual, np.ndarray]]) -> List[Individual]:
    tasks = []
    for idx, (p1, p2, w) in enumerate(pairs):
        dominant_idx = int(np.argmax(w))
        dominant = OBJ_NAMES[dominant_idx]
        w_str = " / ".join(f"{OBJ_NAMES[j]}={w[j]:.2f}" for j in range(N_OBJ))
        tasks.append(
            f"任务{idx+1}:\n"
            f"  父代A: 【{p1.topic}】（{p1.domain}）{p1.description}\n"
            f"  父代B: 【{p2.topic}】（{p2.domain}）{p2.description}\n"
            f"  演化偏向: 重点提升「{dominant}」（权重: {w_str}）"
        )
    tasks_text = "\n\n".join(tasks)
    prompt = f"""你是跨学科科研创新专家，请对以下 {len(pairs)} 组研究知识执行交叉变异，产生全新的长尾科研知识。

{tasks_text}

规则：
1. 融合两个父代的核心思想，产生真正的**跨学科交叉创新**
2. 后代应属于长尾知识（小众、专业、非主流，具有探索价值）
3. 根据演化偏向，使后代在对应目标上表现更优
4. 每个后代必须是独立的新知识点，不得与父代雷同
5. 鼓励跨越父代所在学科边界，探索新兴交叉领域

以 JSON 返回（只返回 JSON，数组长度必须恰好为 {len(pairs)}）：
{{"offspring": [
  {{"topic": "≤18字", "domain": "所属学科/交叉领域", "description": "60-100字"}},
  ...
]}}"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.88,
    )
    data = json.loads(resp.choices[0].message.content)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it["description"],
                       domain=it.get("domain","")) for it in items[:len(pairs)]]


# ══════════════════════════════════════════════════════
#  MOEA/D 一代演化
# ══════════════════════════════════════════════════════
def moead_one_generation(population, weights, neighborhoods, ideal_point):
    n = len(population)
    pairs = []
    parent_indices = []
    for i in range(n):
        nb = neighborhoods[i]
        idx1, idx2 = (random.sample(nb, 2) if len(nb) >= 2
                      else [nb[0], random.randint(0, n-1)])
        if idx1 == idx2:
            idx2 = (idx2 + 1) % n
        pairs.append((population[idx1], population[idx2], weights[i]))
        parent_indices.append(i)

    offspring_list = llm_batch_crossover(pairs)
    scores_batch   = llm_evaluate_batch(offspring_list)
    for child, sc in zip(offspring_list, scores_batch):
        child.scores = sc

    for child in offspring_list:
        ideal_point = np.maximum(ideal_point, child.scores)

    new_pop = list(population)
    for i, child in enumerate(offspring_list):
        for j in neighborhoods[parent_indices[i]]:
            if tchebycheff(child.scores, weights[j], ideal_point) <= \
               tchebycheff(new_pop[j].scores, weights[j], ideal_point):
                new_pop[j] = child

    return new_pop, ideal_point


# ══════════════════════════════════════════════════════
#  显示
# ══════════════════════════════════════════════════════
def score_bar(v, w=8):
    return "█" * int(v * w) + "░" * (w - int(v * w))

def print_generation(gen, population, ideal_point):
    label = "初始种群" if gen == 0 else f"第 {gen:2d} 代"
    avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(N_OBJ)]
    print(f"\n{'─'*80}")
    print(f"  ◈ {label}  │  理想点: " +
          " ".join(f"{OBJ_NAMES[j][:2]}={ideal_point[j]*10:.1f}" for j in range(N_OBJ)))
    print("  均值: " + "  ".join(f"{OBJ_NAMES[j][:3]}={avgs[j]*10:.2f}" for j in range(N_OBJ)))
    print(f"{'─'*80}")
    ranked = sorted(population, key=lambda x: sum(x.scores), reverse=True)
    for k, ind in enumerate(ranked[:8]):  # 只展示前8个
        scores_str = "  ".join(
            f"{OBJ_NAMES[j][:2]}{score_bar(ind.scores[j],6)}{ind.scores[j]*10:.1f}"
            for j in range(N_OBJ)
        )
        print(f"  {k+1:2}. {ind.topic:<28} [{ind.domain}]")
        print(f"      {scores_str}")
    if len(ranked) > 8:
        print(f"  ... 另有 {len(ranked)-8} 个个体")
    print()


def find_pareto(population):
    pareto = []
    for i, ind in enumerate(population):
        dominated = any(
            all(other.scores[k] >= ind.scores[k] for k in range(N_OBJ)) and
            any(other.scores[k] >  ind.scores[k] for k in range(N_OBJ))
            for j, other in enumerate(population) if j != i
        )
        if not dominated:
            pareto.append(ind)
    return pareto


# ══════════════════════════════════════════════════════
#  结果存储（供 PDF 生成器使用）
# ══════════════════════════════════════════════════════
def save_results(init_pop, gen_stats, final_pop, pareto, output_path):
    def ind2dict(ind):
        return {
            "topic": ind.topic,
            "domain": ind.domain,
            "description": ind.description,
            "scores": ind.scores,
            "pareto": ind in pareto,
        }
    data = {
        "config": {
            "pop_size": POP_SIZE, "n_obj": N_OBJ, "n_neighbors": N_NEIGHBORS,
            "n_gens": N_GENS, "model": MODEL, "obj_names": OBJ_NAMES,
        },
        "init_pop":   [ind2dict(ind) for ind in init_pop],
        "gen_stats":  gen_stats,
        "final_pop":  [ind2dict(ind) for ind in final_pop],
        "pareto":     [ind2dict(ind) for ind in pareto],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {output_path}")


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
def main():
    print("═" * 80)
    print("  MOEA/D × LLM ── 跨学科长尾科研知识多目标进化（扩展版）")
    print(f"  目标: {' × '.join(OBJ_NAMES)}  │  {N_GENS} 代")
    print(f"  模型: {MODEL}  │  种群: {POP_SIZE}  │  邻居: {N_NEIGHBORS}")
    print("═" * 80)

    weights       = generate_weight_vectors(POP_SIZE, N_OBJ)
    neighborhoods = compute_neighborhoods(weights, N_NEIGHBORS)
    print(f"\n[初始化] 权重向量 {len(weights)} 个，邻居 {N_NEIGHBORS} 个")

    print("\n[Step 1/3] 生成初始种群...")
    t0 = time.time()
    population = llm_generate_initial(POP_SIZE)
    init_pop_copy = list(population)
    print(f"  完成 {time.time()-t0:.1f}s")

    print("\n[Step 2/3] 评估初始种群...")
    t0 = time.time()
    for ind, sc in zip(population, llm_evaluate_batch(population)):
        ind.scores = sc
    print(f"  完成 {time.time()-t0:.1f}s")

    ideal_point = np.array([max(ind.scores[j] for ind in population)
                             for j in range(N_OBJ)])
    gen_stats = []
    avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(N_OBJ)]
    comp = sum(avgs) / N_OBJ
    gen_stats.append({"gen": 0, "avgs": [round(a*10,2) for a in avgs],
                       "composite": round(comp*10, 2)})
    print_generation(0, population, ideal_point)

    print("\n[Step 3/3] 开始 MOEA/D 演化...\n")
    for gen in range(1, N_GENS + 1):
        print(f"[第 {gen:2d}/{N_GENS} 代] 交叉变异 + 评估...", end=" ", flush=True)
        t0 = time.time()
        population, ideal_point = moead_one_generation(
            population, weights, neighborhoods, ideal_point)
        avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(N_OBJ)]
        comp = sum(avgs) / N_OBJ
        gen_stats.append({"gen": gen, "avgs": [round(a*10,2) for a in avgs],
                           "composite": round(comp*10, 2)})
        print(f"✓  {time.time()-t0:.1f}s  │  综合均分 {comp*10:.2f}")
        print_generation(gen, population, ideal_point)

    pareto = find_pareto(population)
    print(f"\n  Pareto 最优解集: {len(pareto)} 个")
    for ind in sorted(pareto, key=lambda x: sum(x.scores), reverse=True):
        scores_str = " ".join(f"{OBJ_NAMES[j][:2]}={ind.scores[j]*10:.1f}" for j in range(N_OBJ))
        print(f"  ◆ {ind.topic} [{ind.domain}]  {scores_str}")

    json_path = os.path.expanduser("~/moead_science_results.json")
    save_results(init_pop_copy, gen_stats, population, pareto, json_path)
    print("\n  演化完成！")


if __name__ == "__main__":
    main()
