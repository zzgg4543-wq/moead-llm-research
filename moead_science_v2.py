#!/usr/bin/env python3
"""
MOEA/D × Claude：跨学科长尾科研知识多目标进化（7目标版）
════════════════════════════════════════════════════════
种群: 20   目标: 7   邻居: 5   演化: 10 代
领域: 不限学科

7 个目标（全部最大化，0-10分）:
  f1 - 知识价值  (Knowledge)     : 对基础科学理论的贡献深度
  f2 - 社会影响  (Social Impact)  : 对人类社会的长远正向影响
  f3 - 长尾度    (Long-tail)      : 研究稀缺性/小众程度
  f4 - 跨学科性  (Interdiscip)    : 跨领域连接与融合潜力
  f5 - 前沿性    (Frontier)       : 新颖程度/突破已知边界
  f6 - 可行性    (Feasibility)    : 当前技术条件下可开展研究的程度
  f7 - 合理性    (Rigor)          : 科学假设的逻辑严谨性与证伪可能性
"""

import json, random, os, time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import anthropic

# ══════════════════════════════════════════════════════
POP_SIZE    = 20
N_OBJ       = 7
N_NEIGHBORS = 5
N_GENS      = 10
MODEL       = "claude-sonnet-4-5"

OBJ_NAMES = ["知识价值", "社会影响", "长尾度", "跨学科性", "前沿性", "可行性", "合理性"]
OBJ_KEYS  = ["knowledge", "social", "longtail", "interdiscip", "frontier", "feasibility", "rigor"]

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# ══════════════════════════════════════════════════════
@dataclass
class Individual:
    topic: str
    description: str
    domain: str = ""
    scores: List[float] = field(default_factory=lambda: [0.0] * N_OBJ)


# ══════════════════════════════════════════════════════
#  Claude 调用封装（自动去除代码块包裹）
# ══════════════════════════════════════════════════════
def call_claude(prompt: str, max_tokens: int = 4096, temperature: float = 0.7) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system="你是专业科研分析助手。所有回复必须是有效的JSON格式，不包含任何其他文字、代码块标记（```）或解释。直接输出JSON对象。",
        messages=[{"role": "user", "content": prompt}]
    )
    text = resp.content[0].text.strip()
    # 去除可能的 ```json ... ``` 包裹
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


# ══════════════════════════════════════════════════════
#  权重向量：Dirichlet 均匀采样 + 多样性筛选
# ══════════════════════════════════════════════════════
def generate_weight_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """生成 n 个在 d 维单纯形上的多样化权重向量（Dirichlet + 贪心选取）"""
    rng = np.random.default_rng(seed)
    # 过采样 10x 后用贪心最大最小距离选取最多样的 n 个
    candidates = rng.dirichlet(np.ones(d), size=n * 10)
    # 贪心选取
    selected = [0]
    remaining = list(range(1, len(candidates)))
    while len(selected) < n:
        dists = np.min(
            np.linalg.norm(candidates[remaining][:, None] - candidates[selected][None, :], axis=2),
            axis=1
        )
        best = remaining[int(np.argmax(dists))]
        selected.append(best)
        remaining.remove(best)
    return candidates[selected]


def compute_neighborhoods(W: np.ndarray, T: int) -> List[List[int]]:
    n = len(W)
    return [np.argsort(np.linalg.norm(W - W[i], axis=1))[:T].tolist() for i in range(n)]

def tchebycheff(scores: List[float], w: np.ndarray, ideal: np.ndarray) -> float:
    return float(np.max(w * np.abs(ideal - np.array(scores))))


# ══════════════════════════════════════════════════════
#  LLM 接口
# ══════════════════════════════════════════════════════
def llm_generate_initial(n: int) -> List[Individual]:
    print(f"  [Claude] 生成初始 {n} 个跨学科长尾知识条目...")
    prompt = f"""请生成 {n} 个跨学科长尾研究知识条目。

"长尾科研知识"定义：
- 尚未被主流研究充分关注，处于早期探索阶段
- 具有潜在颠覆性但目前研究群体极小
- 不同学科交叉产生的新兴方向
- 不限任何单一学科

要求：
- {n} 个条目须覆盖多样化的一级学科（物理/化学/生物/数学/材料/环境/神经/社会/天文/经济等）
- 每个条目必须具体可研究，而非过于宏观
- 体现长尾特性：冷门、专业、非主流，但逻辑上合理且原则上可验证
- 要求有合理的科学假设基础，不能是纯粹的科幻臆想

JSON格式：
{{
  "items": [
    {{
      "topic": "简短主题名（≤18字）",
      "domain": "所属交叉学科（具体，如：量子生物信息学）",
      "description": "详细描述（70-100字：是什么、科学依据、为何是长尾、潜在价值、当前研究现状）"
    }}
  ]
}}"""
    data = call_claude(prompt, max_tokens=6000, temperature=0.9)
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
    prompt = f"""请对以下 {len(individuals)} 个跨学科研究知识条目进行七维客观评分。

{items_text}

评分标准（每项 0-10 整数，评分需有显著区分度，不要全部给高分）：
- knowledge（知识价值）：对基础科学理论的贡献深度，能否推进对自然/社会规律的根本理解
- social（社会影响）：若取得突破，对人类社会、技术、医疗、环境的长远正向影响
- longtail（长尾度）：当前研究稀缺性（极主流=1，极小众冷门=10）
- interdiscip（跨学科性）：与其他学科深度融合的潜力与广度
- frontier（前沿性）：知识的新颖度与突破已知边界的程度
- feasibility（可行性）：在当前技术/实验/计算条件下可开展研究的现实程度（完全不可行=1，当前完全可做=10）
- rigor（合理性）：科学假设的逻辑严谨性、是否有理论基础支撑、是否原则上可证伪（纯臆想=1，严谨扎实=10）

注意：可行性和合理性是独立维度——某个研究可以非常前沿但暂时不可行，也可以很合理但已不再新颖。

JSON格式（数组长度必须恰好为 {len(individuals)}）：
{{"scores": [
  {{"knowledge":整数,"social":整数,"longtail":整数,"interdiscip":整数,"frontier":整数,"feasibility":整数,"rigor":整数}},
  ...
]}}"""
    data = call_claude(prompt, max_tokens=3000, temperature=0.1)
    raw = data["scores"]
    return [[s["knowledge"]/10, s["social"]/10, s["longtail"]/10,
             s["interdiscip"]/10, s["frontier"]/10,
             s["feasibility"]/10, s["rigor"]/10] for s in raw]


def llm_batch_crossover(pairs: List[Tuple[Individual, Individual, np.ndarray]]) -> List[Individual]:
    tasks = []
    for idx, (p1, p2, w) in enumerate(pairs):
        dominant_idx = int(np.argmax(w))
        dominant = OBJ_NAMES[dominant_idx]
        w_str = " / ".join(f"{OBJ_NAMES[j][:2]}={w[j]:.2f}" for j in range(N_OBJ))
        tasks.append(
            f"任务{idx+1}:\n"
            f"  父代A: 【{p1.topic}】（{p1.domain}）{p1.description}\n"
            f"  父代B: 【{p2.topic}】（{p2.domain}）{p2.description}\n"
            f"  演化偏向: 重点提升「{dominant}」（权重: {w_str}）"
        )
    prompt = f"""请对以下 {len(pairs)} 组研究知识执行交叉变异，产生全新的长尾跨学科科研知识。

{"".join(t + chr(10)*2 for t in tasks)}

规则：
1. 融合两个父代核心思想，产生真正的跨学科交叉创新
2. 后代必须是长尾知识（小众、专业、非主流）
3. 根据演化偏向优化对应目标：
   - 偏「可行性」→ 后代应基于现有技术/方法可开展，有具体实验路径
   - 偏「合理性」→ 后代应有扎实理论基础，科学假设逻辑严密，可证伪
   - 偏「知识价值」→ 后代应能推进对某一科学规律的根本性理解
   - 偏「社会影响」→ 后代应有明确的应用场景和社会价值
4. 后代必须独立、不得与父代雷同
5. 后代应具备合理性：不能是纯粹科幻，需有理论依据

JSON格式（数组长度必须恰好为 {len(pairs)}）：
{{"offspring": [
  {{"topic":"≤18字","domain":"具体交叉学科","description":"70-100字"}},
  ...
]}}"""
    data = call_claude(prompt, max_tokens=6000, temperature=0.85)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it["description"],
                       domain=it.get("domain", "")) for it in items[:len(pairs)]]


# ══════════════════════════════════════════════════════
#  MOEA/D 一代
# ══════════════════════════════════════════════════════
def moead_one_generation(population, weights, neighborhoods, ideal_point):
    n = len(population)
    pairs, parent_indices = [], []
    for i in range(n):
        nb = neighborhoods[i]
        idx1, idx2 = (random.sample(nb, 2) if len(nb) >= 2
                      else [nb[0], random.randint(0, n - 1)])
        if idx1 == idx2: idx2 = (idx2 + 1) % n
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
def print_generation(gen, population, ideal_point):
    label = "初始种群" if gen == 0 else f"第 {gen:2d} 代"
    avgs = [sum(x.scores[j] for x in population) / len(population) for j in range(N_OBJ)]
    print(f"\n{'─'*88}")
    print(f"  ◈ {label}  │  理想点: " +
          " ".join(f"{OBJ_NAMES[j][:2]}={ideal_point[j]*10:.0f}" for j in range(N_OBJ)))
    print("  均值: " + "  ".join(f"{OBJ_NAMES[j][:3]}={avgs[j]*10:.1f}" for j in range(N_OBJ)))
    print(f"{'─'*88}")
    ranked = sorted(population, key=lambda x: sum(x.scores), reverse=True)
    BAR = 6
    for k, ind in enumerate(ranked[:6]):
        s = ind.scores
        bars = "  ".join(
            f"{OBJ_NAMES[j][:2]}{'█'*int(s[j]*BAR)+'░'*(BAR-int(s[j]*BAR))}{s[j]*10:.0f}"
            for j in range(N_OBJ)
        )
        print(f"  {k+1:2}. {ind.topic:<28} [{ind.domain[:20]}]")
        print(f"      {bars}")
    if len(ranked) > 6:
        print(f"  ... 另有 {len(ranked)-6} 个")
    print()


def find_pareto(population):
    pareto = []
    for i, ind in enumerate(population):
        if not any(
            all(other.scores[k] >= ind.scores[k] for k in range(N_OBJ)) and
            any(other.scores[k] >  ind.scores[k] for k in range(N_OBJ))
            for j, other in enumerate(population) if j != i
        ):
            pareto.append(ind)
    return pareto


def save_results(init_pop, gen_stats, final_pop, pareto, path):
    pareto_keys = {(p.topic, p.domain) for p in pareto}
    def ind2d(ind):
        return {"topic": ind.topic, "domain": ind.domain,
                "description": ind.description, "scores": ind.scores,
                "pareto": (ind.topic, ind.domain) in pareto_keys}
    data = {
        "config": {"pop_size": POP_SIZE, "n_obj": N_OBJ, "n_neighbors": N_NEIGHBORS,
                   "n_gens": N_GENS, "model": MODEL, "obj_names": OBJ_NAMES},
        "init_pop":  [ind2d(i) for i in init_pop],
        "gen_stats": gen_stats,
        "final_pop": [ind2d(i) for i in final_pop],
        "pareto":    [ind2d(i) for i in pareto],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {path}")


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
def main():
    print("═" * 88)
    print("  MOEA/D × Claude ── 跨学科长尾科研知识多目标进化（7目标扩展版）")
    print(f"  目标: {' × '.join(OBJ_NAMES)}")
    print(f"  模型: {MODEL}  │  种群: {POP_SIZE}  │  邻居: {N_NEIGHBORS}  │  演化: {N_GENS} 代")
    print("═" * 88)

    weights       = generate_weight_vectors(POP_SIZE, N_OBJ)
    neighborhoods = compute_neighborhoods(weights, N_NEIGHBORS)
    print(f"\n[初始化] 权重向量 {len(weights)} 个，邻居 {N_NEIGHBORS} 个（Dirichlet 均匀采样）")

    print("\n[Step 1/3] 生成初始种群...")
    t0 = time.time()
    population  = llm_generate_initial(POP_SIZE)
    init_pop_bk = list(population)
    print(f"  完成 {time.time()-t0:.1f}s")

    print("\n[Step 2/3] 评估初始种群（7维）...")
    t0 = time.time()
    for ind, sc in zip(population, llm_evaluate_batch(population)):
        ind.scores = sc
    print(f"  完成 {time.time()-t0:.1f}s")

    ideal = np.array([max(ind.scores[j] for ind in population) for j in range(N_OBJ)])
    gen_stats = []
    avgs = [sum(x.scores[j] for x in population) / len(population) for j in range(N_OBJ)]
    gen_stats.append({"gen": 0, "avgs": [round(a*10, 2) for a in avgs],
                      "composite": round(sum(avgs)/N_OBJ*10, 2)})
    print_generation(0, population, ideal)

    print("\n[Step 3/3] MOEA/D 演化...\n")
    for gen in range(1, N_GENS + 1):
        print(f"[第 {gen:2d}/{N_GENS} 代]", end=" ", flush=True)
        t0 = time.time()
        population, ideal = moead_one_generation(population, weights, neighborhoods, ideal)
        avgs = [sum(x.scores[j] for x in population) / len(population) for j in range(N_OBJ)]
        comp = sum(avgs) / N_OBJ
        gen_stats.append({"gen": gen, "avgs": [round(a*10, 2) for a in avgs],
                          "composite": round(comp*10, 2)})
        print(f"✓ {time.time()-t0:.1f}s  │  综合={comp*10:.2f}")
        print_generation(gen, population, ideal)

    pareto = find_pareto(population)
    print(f"\n  ★ Pareto 最优解集: {len(pareto)} 个")
    for ind in sorted(pareto, key=lambda x: sum(x.scores), reverse=True):
        sc_str = " ".join(f"{OBJ_NAMES[j][:2]}={ind.scores[j]*10:.0f}" for j in range(N_OBJ))
        print(f"  ◆ {ind.topic}  {sc_str}")

    json_path = os.path.expanduser("~/moead_science_v2_results.json")
    save_results(init_pop_bk, gen_stats, population, pareto, json_path)
    print("\n  演化完成！")


if __name__ == "__main__":
    main()
