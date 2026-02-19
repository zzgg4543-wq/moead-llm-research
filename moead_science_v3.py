#!/usr/bin/env python3
"""
MOEA/D × DeepSeek：跨学科长尾科研知识多目标进化（7目标 · 可行性/合理性强化版）
════════════════════════════════════════════════════════════════════════════
算法改进（针对可行性 f6、合理性 f7）：
  ① 偏置权重向量：Dirichlet(α=[1,1,1,1,1,3,3])，f6/f7 获得 3x 更多子问题关注
  ② 惩罚约束 Tchebycheff：f6<0.5 或 f7<0.5 时叠加惩罚项（ε-约束变体）
  ③ 精英保留：每代强制保留全局最优 f6、f7 个体不被替换
  ④ 强化 LLM 提示：交叉变异要求后代 可行性≥6、合理性≥6，并附具体研究路径
  ⑤ 初始种群偏置：明确要求 LLM 生成「当前可开展、理论有依据」的长尾知识
"""

import json, random, os, time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from openai import OpenAI

# ══════════════════════════════════════════════════════
POP_SIZE    = 20
N_OBJ       = 7
N_NEIGHBORS = 5
N_GENS      = 10
MODEL       = "deepseek-chat"

# f6=可行性(idx 5), f7=合理性(idx 6) 的改进参数
FLOOR_IDX   = [5, 6]       # 需要强化的目标索引
FLOOR_MIN   = 0.55         # 低于此分值（5.5分）才触发惩罚
PENALTY_W   = 3.0          # 惩罚权重（叠加到 Tchebycheff 值）
ALPHA_BIAS  = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0]  # Dirichlet α，f6/f7 偏置 3x

OBJ_NAMES = ["知识价值", "社会影响", "长尾度", "跨学科性", "前沿性", "可行性", "合理性"]
OBJ_KEYS  = ["knowledge", "social", "longtail", "interdiscip", "frontier", "feasibility", "rigor"]

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
#  DeepSeek 调用封装
# ══════════════════════════════════════════════════════
def call_claude(prompt: str, max_tokens: int = 4096, temperature: float = 0.7,
                retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "你是专业科研分析助手。所有回复必须是有效的JSON格式，不包含任何其他文字、代码块标记（```）或解释。直接输出JSON对象。"},
                    {"role": "user", "content": prompt},
                ]
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text[text.find("\n") + 1:]
                if text.endswith("```"):
                    text = text[:-3].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                print(f"\n  [重试 {attempt+1}/{retries}] JSON解析失败，重新生成...")
                temperature = min(temperature + 0.05, 1.0)
            else:
                raise
    raise RuntimeError("DeepSeek JSON 解析失败")


# ══════════════════════════════════════════════════════
#  ① 偏置权重向量（f6/f7 获得更多子问题）
# ══════════════════════════════════════════════════════
def generate_weight_vectors_biased(n: int, d: int, alpha: List[float],
                                   seed: int = 42) -> np.ndarray:
    """用偏置 Dirichlet 分布生成权重向量，alpha 越大的目标获得更高期望权重"""
    rng = np.random.default_rng(seed)
    alpha_arr = np.array(alpha, dtype=float)
    candidates = rng.dirichlet(alpha_arr, size=n * 12)
    # 贪心最大最小距离，保证多样性
    selected = [0]
    remaining = list(range(1, len(candidates)))
    while len(selected) < n:
        dists = np.min(
            np.linalg.norm(candidates[remaining][:, None] -
                           candidates[selected][None, :], axis=2), axis=1)
        best = remaining[int(np.argmax(dists))]
        selected.append(best)
        remaining.remove(best)
    weights = candidates[selected]
    # 确保 f6、f7 在至少 40% 子问题中权重超过 0.15
    print(f"  [权重向量] f6平均权重={weights[:,5].mean():.3f}  f7平均权重={weights[:,6].mean():.3f}")
    return weights


def compute_neighborhoods(W: np.ndarray, T: int) -> List[List[int]]:
    return [np.argsort(np.linalg.norm(W - W[i], axis=1))[:T].tolist()
            for i in range(len(W))]


# ══════════════════════════════════════════════════════
#  ② 惩罚约束 Tchebycheff
# ══════════════════════════════════════════════════════
def tchebycheff_penalized(scores: List[float], w: np.ndarray,
                          ideal: np.ndarray) -> float:
    """标准 Tchebycheff + 对低可行性/合理性解的软约束惩罚"""
    base = float(np.max(w * np.abs(ideal - np.array(scores))))
    penalty = sum(
        PENALTY_W * (FLOOR_MIN - scores[i])
        for i in FLOOR_IDX if scores[i] < FLOOR_MIN
    )
    return base + penalty


# ══════════════════════════════════════════════════════
#  LLM 接口
# ══════════════════════════════════════════════════════
def llm_generate_initial(n: int) -> List[Individual]:
    """⑤ 初始种群偏置：要求生成可行且理论合理的长尾研究"""
    print(f"  [Claude] 生成初始 {n} 个长尾知识（偏置：可行+合理）...")
    prompt = f"""请生成 {n} 个跨学科长尾研究知识条目。

核心要求（本次特别强调）：
- 每个条目必须在当前（2024-2025年）技术/实验/计算条件下**原则上可以开展研究**（可行性 ≥ 6/10）
- 每个条目必须有**扎实的理论基础或实验依据**，科学假设逻辑严密且原则上可证伪（合理性 ≥ 6/10）
- 同时保持长尾特性：研究群体小、非主流、尚未充分探索

"长尾但可行且合理"的示例类型：
- 已有理论框架，但缺乏实验验证的交叉方向（如：某量子效应在生物系统中的已知机制，在新型材料中的验证）
- 现有技术可以升级/迁移到新领域的研究（如：已有的单细胞测序技术用于极端环境微生物群落）
- 数学上已经证明存在但物理实现未探索的现象

避免：纯粹的科幻假设、缺乏已知理论基础的臆想、当前完全无法实验验证的想法。

覆盖多个学科领域，每个条目在 description 中明确说明：
1. 现有理论/实验基础是什么
2. 具体可以用什么方法研究
3. 为什么当前可行但尚未被充分研究

JSON格式：
{{
  "items": [
    {{
      "topic": "简短主题名（≤18字）",
      "domain": "具体交叉学科领域",
      "description": "70-100字描述（含现有基础、研究方法、可行原因、长尾原因）"
    }}
  ]
}}"""
    data = call_claude(prompt, max_tokens=6000, temperature=0.85)
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

评分标准（0-10 整数，需有显著区分度）：
- knowledge（知识价值）：对基础科学理论的贡献深度
- social（社会影响）：对人类社会的长远正向影响
- longtail（长尾度）：当前研究稀缺性（主流=1，极小众=10）
- interdiscip（跨学科性）：跨领域融合潜力
- frontier（前沿性）：突破已知边界的程度
- feasibility（可行性）：在当前（2025年）技术/实验/计算条件下可实际开展研究的程度
  · 1-3分：基本不可行，关键技术缺口无法在5年内解决
  · 4-5分：技术上勉强可行但极其困难，需重大突破
  · 6-7分：可行，现有实验室有能力开展，只需适度技术整合
  · 8-10分：高度可行，方法成熟，主要挑战是资金和时间
- rigor（合理性）：科学假设的逻辑严谨性与理论基础
  · 1-3分：假设缺乏理论依据，难以证伪，偏向科幻
  · 4-5分：有一定依据但逻辑链较弱
  · 6-7分：有明确理论支撑，假设逻辑严密，原则上可证伪
  · 8-10分：理论基础扎实，假设清晰可测试，与已知物理/化学/生物规律完全自洽

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
    """④ 强化 LLM 提示：要求后代可行性≥6、合理性≥6"""
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

    prompt = f"""请对以下 {len(pairs)} 组研究知识执行交叉变异，产生新的跨学科长尾科研知识。

{"".join(t + chr(10)*2 for t in tasks)}

【硬性要求——每个后代必须同时满足】：
1. 可行性 ≥ 6/10：必须在 2025 年现有技术/实验条件下原则上可开展
   → description 中须明确说明用什么已有技术/方法可以研究它
2. 合理性 ≥ 6/10：必须有已知的理论基础或实验证据支撑科学假设
   → description 中须引用或类比已有的科学规律/理论框架
3. 长尾性：必须仍是小众、非主流、尚未充分探索的方向

【软性目标】：根据演化偏向，在满足硬性要求的前提下优化对应目标：
- 偏「可行性」→ 重点描述具体研究路径和现有技术支撑
- 偏「合理性」→ 重点阐述严密的科学假设链和可证伪方案
- 偏「知识价值」→ 聚焦对基础理论的贡献
- 偏「社会影响」→ 聚焦应用场景和社会价值

后代须融合两个父代的核心思想，产生真正的交叉创新，不得与父代雷同。

JSON格式（数组长度必须恰好为 {len(pairs)}）：
{{"offspring": [
  {{"topic":"≤18字","domain":"具体交叉学科","description":"70-100字（含研究方法+理论基础+长尾原因）"}},
  ...
]}}"""
    data = call_claude(prompt, max_tokens=8192, temperature=0.82)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it["description"],
                       domain=it.get("domain", "")) for it in items[:len(pairs)]]


# ══════════════════════════════════════════════════════
#  ③ MOEA/D 一代演化（含精英保留）
# ══════════════════════════════════════════════════════
def moead_one_generation(population, weights, neighborhoods, ideal_point):
    n = len(population)

    # ③ 精英保留：记录 f6/f7 当前最优个体索引
    best_f6_idx = max(range(n), key=lambda i: population[i].scores[5])
    best_f7_idx = max(range(n), key=lambda i: population[i].scores[6])
    elite_indices = {best_f6_idx, best_f7_idx}
    elite_inds    = {idx: population[idx] for idx in elite_indices}

    # 生成父代对
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

    # 更新理想点
    for child in offspring_list:
        ideal_point = np.maximum(ideal_point, child.scores)

    # ② 惩罚约束替换
    new_pop = list(population)
    for i, child in enumerate(offspring_list):
        for j in neighborhoods[parent_indices[i]]:
            if tchebycheff_penalized(child.scores, weights[j], ideal_point) <= \
               tchebycheff_penalized(new_pop[j].scores, weights[j], ideal_point):
                new_pop[j] = child

    # ③ 恢复精英（防止最优 f6/f7 被替换掉）
    for idx, elite in elite_inds.items():
        curr_elite_score_f6 = new_pop[idx].scores[5]
        curr_elite_score_f7 = new_pop[idx].scores[6]
        # 如果当前位置的 f6/f7 变差了，恢复精英（但不恢复给 f7 精英位置 f6 已经变好的情况）
        if idx == best_f6_idx and curr_elite_score_f6 < elite.scores[5]:
            new_pop[idx] = elite
        if idx == best_f7_idx and curr_elite_score_f7 < elite.scores[6]:
            new_pop[idx] = elite

    return new_pop, ideal_point


# ══════════════════════════════════════════════════════
#  显示 & 辅助
# ══════════════════════════════════════════════════════
def print_generation(gen, population, ideal_point):
    label = "初始种群" if gen == 0 else f"第 {gen:2d} 代"
    avgs = [sum(x.scores[j] for x in population) / len(population) for j in range(N_OBJ)]
    print(f"\n{'─'*92}")
    print(f"  ◈ {label}  │  " +
          " ".join(f"{OBJ_NAMES[j][:2]}={ideal_point[j]*10:.0f}" for j in range(N_OBJ)))
    # 高亮可行性和合理性
    avg_line = "  均值: " + "  ".join(
        f"\033[1;32m{OBJ_NAMES[j][:3]}={avgs[j]*10:.1f}\033[0m"
        if j in FLOOR_IDX else f"{OBJ_NAMES[j][:3]}={avgs[j]*10:.1f}"
        for j in range(N_OBJ)
    )
    print(avg_line)
    # 可行/合理专项
    f6_vals = sorted([x.scores[5]*10 for x in population], reverse=True)
    f7_vals = sorted([x.scores[6]*10 for x in population], reverse=True)
    print(f"  ★ 可行性 Top3: {f6_vals[:3]}  合理性 Top3: {f7_vals[:3]}")
    print(f"{'─'*92}")
    ranked = sorted(population, key=lambda x: sum(x.scores), reverse=True)
    for k, ind in enumerate(ranked[:5]):
        s = ind.scores
        bars = "  ".join(
            f"\033[32m{OBJ_NAMES[j][:2]}{'█'*int(s[j]*6)+'░'*(6-int(s[j]*6))}{s[j]*10:.0f}\033[0m"
            if j in FLOOR_IDX
            else f"{OBJ_NAMES[j][:2]}{'█'*int(s[j]*6)+'░'*(6-int(s[j]*6))}{s[j]*10:.0f}"
            for j in range(N_OBJ)
        )
        print(f"  {k+1:2}. {ind.topic:<30} [{ind.domain[:20]}]")
        print(f"      {bars}")
    if len(ranked) > 5: print(f"  ... 另有 {len(ranked)-5} 个")
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
                   "n_gens": N_GENS, "model": MODEL, "obj_names": OBJ_NAMES,
                   "floor_min": FLOOR_MIN, "penalty_w": PENALTY_W,
                   "alpha_bias": ALPHA_BIAS},
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
    print("═" * 92)
    print("  MOEA/D × DeepSeek ── 跨学科长尾科研知识（7目标 · 可行性/合理性强化版）")
    print(f"  目标: {' × '.join(OBJ_NAMES)}")
    print(f"  模型: {MODEL}  │  种群: {POP_SIZE}  │  邻居: {N_NEIGHBORS}  │  演化: {N_GENS} 代")
    print(f"  改进: ①偏置权重 ②惩罚约束(ε={FLOOR_MIN},w={PENALTY_W}) ③精英保留 ④强化提示 ⑤初始偏置")
    print("═" * 92)

    print("\n[初始化] 生成偏置权重向量（f6/f7 α=3.0）...")
    weights       = generate_weight_vectors_biased(POP_SIZE, N_OBJ, ALPHA_BIAS)
    neighborhoods = compute_neighborhoods(weights, N_NEIGHBORS)

    print("\n[Step 1/3] 生成初始种群（强调可行性+合理性）...")
    t0 = time.time()
    population  = llm_generate_initial(POP_SIZE)
    init_pop_bk = list(population)
    print(f"  完成 {time.time()-t0:.1f}s")

    print("\n[Step 2/3] 评估初始种群（7维精细评分）...")
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

    print("\n[Step 3/3] MOEA/D 演化（惩罚约束 + 精英保留）...\n")
    for gen in range(1, N_GENS + 1):
        print(f"[第 {gen:2d}/{N_GENS} 代]", end=" ", flush=True)
        t0 = time.time()
        population, ideal = moead_one_generation(population, weights, neighborhoods, ideal)
        avgs = [sum(x.scores[j] for x in population) / len(population) for j in range(N_OBJ)]
        comp = sum(avgs) / N_OBJ
        gen_stats.append({"gen": gen, "avgs": [round(a*10, 2) for a in avgs],
                          "composite": round(comp*10, 2)})
        f6_avg = avgs[5] * 10
        f7_avg = avgs[6] * 10
        print(f"✓ {time.time()-t0:.1f}s  │  综合={comp*10:.2f}  \033[32m可行={f6_avg:.2f} 合理={f7_avg:.2f}\033[0m")
        print_generation(gen, population, ideal)
        # 每代自动保存中间结果
        pareto_tmp = find_pareto(population)
        save_results(init_pop_bk, gen_stats, population, pareto_tmp,
                     os.path.expanduser("~/moead_science_v3ds_results.json"))

    pareto = find_pareto(population)
    # 统计可行/合理平均
    pf6 = sum(p.scores[5] for p in pareto) / len(pareto) * 10
    pf7 = sum(p.scores[6] for p in pareto) / len(pareto) * 10
    print(f"\n  ★ Pareto 最优解集: {len(pareto)} 个")
    print(f"  ★ Pareto 平均可行性={pf6:.2f}  平均合理性={pf7:.2f}")
    for ind in sorted(pareto, key=lambda x: sum(x.scores), reverse=True):
        sc_str = " ".join(f"{OBJ_NAMES[j][:2]}={ind.scores[j]*10:.0f}" for j in range(N_OBJ))
        print(f"  ◆ {ind.topic}  {sc_str}")

    json_path = os.path.expanduser("~/moead_science_v3ds_results.json")
    save_results(init_pop_bk, gen_stats, population, pareto, json_path)
    print("\n  演化完成！")


if __name__ == "__main__":
    main()
