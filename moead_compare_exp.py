#!/usr/bin/env python3
"""
对比实验：单目标（新颖性）vs 四目标 MOEA/D
════════════════════════════════════════════════════════
实验A：单目标进化 — 仅优化「新颖性」（(μ+λ)-ES 贪心选择）
实验B：四目标 MOEA/D — 新颖性 × 知识价值 × 可行性 × 合理性

两个实验：相同种群规模、相同代数、相同 LLM（DeepSeek）
评估时两者都打7分（含全部目标），方便直接对比代价
"""

import json, random, os, time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from openai import OpenAI

# ══════════════════════════════════════════════════════
POP_SIZE    = 15    # 两个实验相同
N_GENS      = 6     # 两个实验相同
N_NEIGHBORS = 5     # 仅 MOEA/D 使用
MODEL       = "deepseek-chat"

# 4 个目标（评估时仍用7维，但优化只用以下4个）
# 实验A：只用 f_novelty（前沿性）
# 实验B：f_novelty + f_knowledge + f_feasibility + f_rigor
OBJ_4 = ["新颖性", "知识价值", "可行性", "合理性"]
OBJ_4_IDX = [4, 0, 5, 6]   # 在7维评分向量中的索引

# 7维完整目标（用于评估和对比分析）
OBJ_ALL = ["知识价值", "社会影响", "长尾度", "跨学科性", "前沿性", "可行性", "合理性"]

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)


@dataclass
class Individual:
    topic: str
    description: str
    domain: str = ""
    scores: List[float] = field(default_factory=lambda: [0.0] * 7)


# ══════════════════════════════════════════════════════
#  LLM 调用（统一入口）
# ══════════════════════════════════════════════════════
def call_llm(prompt: str, max_tokens: int = 4096,
             temperature: float = 0.7, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "你是专业科研分析助手。所有回复必须是有效的JSON格式，不包含任何其他文字或代码块标记。"},
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
                print(f"  [重试 {attempt+1}] JSON解析失败...")
                temperature = min(temperature + 0.05, 1.0)
            else:
                raise
    raise RuntimeError("LLM JSON 解析失败")


# ══════════════════════════════════════════════════════
#  共用：初始种群生成
# ══════════════════════════════════════════════════════
def llm_generate_initial(n: int, mode: str) -> List[Individual]:
    if mode == "novelty":
        extra = "核心要求：每个条目要尽可能新颖、前沿、突破性强，不必考虑当前可行性。"
    else:
        extra = ("核心要求：每个条目需在新颖/前沿的同时，"
                 "具有扎实的理论基础（合理性），并在当前技术条件下原则上可开展（可行性）。")
    prompt = f"""请生成 {n} 个跨学科长尾研究知识条目。

{extra}

覆盖多个学科领域，每个条目在 description 中说明核心研究内容。

JSON格式：
{{"items": [
  {{"topic": "简短主题名（≤18字）", "domain": "交叉学科领域",
   "description": "60-90字描述"}},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=5000, temperature=0.9)
    items = data.get("items", [])
    if not items:
        for v in data.values():
            if isinstance(v, list): items = v; break
    return [Individual(topic=it["topic"], description=it["description"],
                       domain=it.get("domain", ""))
            for it in items[:n]]


# ══════════════════════════════════════════════════════
#  共用：批量7维评分
# ══════════════════════════════════════════════════════
def llm_evaluate_batch(individuals: List[Individual]) -> List[List[float]]:
    items_text = "\n".join(
        f"{i+1}. 【{ind.topic}】（{ind.domain}）\n   {ind.description}"
        for i, ind in enumerate(individuals)
    )
    prompt = f"""请对以下 {len(individuals)} 个跨学科研究知识条目进行七维客观评分（0-10整数，需有显著区分度）。

{items_text}

七个维度：
- knowledge（知识价值）：对基础科学理论的贡献深度
- social（社会影响）：对人类社会的长远正向影响
- longtail（长尾度）：研究的稀缺性/小众程度
- interdiscip（跨学科性）：跨领域融合潜力
- frontier（新颖性/前沿性）：突破已知边界的程度，是否新颖
- feasibility（可行性）：在2025年现有技术条件下可实际开展的程度
  · 1-3分：基本不可行  · 4-5分：极其困难  · 6-7分：可行  · 8-10分：高度可行
- rigor（合理性）：科学假设的逻辑严谨性与理论基础
  · 1-3分：偏向科幻，缺乏依据  · 4-5分：依据较弱  · 6-7分：有明确支撑  · 8-10分：理论扎实

JSON格式（数组长度必须恰好为 {len(individuals)}）：
{{"scores": [
  {{"knowledge":整数,"social":整数,"longtail":整数,"interdiscip":整数,"frontier":整数,"feasibility":整数,"rigor":整数}},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=3000, temperature=0.1)
    raw = data["scores"]
    return [[s["knowledge"]/10, s["social"]/10, s["longtail"]/10,
             s["interdiscip"]/10, s["frontier"]/10,
             s["feasibility"]/10, s["rigor"]/10] for s in raw]


# ══════════════════════════════════════════════════════
#  实验A：单目标进化（仅新颖性）
# ══════════════════════════════════════════════════════
def llm_crossover_novelty(parents: List[Individual]) -> List[Individual]:
    """纯新颖性导向的交叉变异"""
    tasks = [
        f"任务{i+1}:\n  父代A: 【{p[0].topic}】{p[0].description}\n"
        f"  父代B: 【{p[1].topic}】{p[1].description}"
        for i, p in enumerate(parents)
    ]
    n = len(parents)
    prompt = f"""请对以下 {n} 组研究知识执行交叉变异，产生更新颖、更前沿的后代。

{"".join(t + chr(10)*2 for t in tasks)}

核心目标：后代必须比两个父代都更新颖、更有突破性，追求最高的前沿性和创新程度。
不必考虑当前可行性或理论基础的完备性。后代须融合两父代核心思想并大胆创新。

JSON格式（数组长度必须恰好为 {n}）：
{{"offspring": [
  {{"topic":"≤18字","domain":"交叉学科","description":"60-90字"}},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=5000, temperature=0.9)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it["description"],
                       domain=it.get("domain", "")) for it in items[:n]]


def run_exp_a(init_pop: List[Individual]) -> Tuple[List[Individual], list]:
    """实验A：(μ+λ)-ES，仅按新颖性(f5)贪心选择"""
    print("\n" + "="*80)
    print("  实验A：单目标进化（仅新颖性）")
    print("="*80)
    population = list(init_pop)
    gen_stats = []

    # 评估初始种群
    for ind, sc in zip(population, llm_evaluate_batch(population)):
        ind.scores = sc
    avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(7)]
    gen_stats.append({"gen": 0, "avgs": [round(a*10, 2) for a in avgs]})
    _print_gen("A", 0, population)

    for gen in range(1, N_GENS + 1):
        print(f"  [A] 第{gen:2d}/{N_GENS}代", end=" ", flush=True)
        t0 = time.time()
        # 生成 λ=POP_SIZE 个后代
        pairs = [(random.choice(population), random.choice(population))
                 for _ in range(POP_SIZE)]
        offspring = llm_crossover_novelty(pairs)
        # 评估后代
        for ind, sc in zip(offspring, llm_evaluate_batch(offspring)):
            ind.scores = sc
        # (μ+λ) 选择：合并后按新颖性降序选 POP_SIZE
        combined = population + offspring
        combined.sort(key=lambda x: x.scores[4], reverse=True)  # idx4=前沿性
        population = combined[:POP_SIZE]
        avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(7)]
        gen_stats.append({"gen": gen, "avgs": [round(a*10, 2) for a in avgs]})
        print(f"✓ {time.time()-t0:.1f}s  新颖均值={avgs[4]*10:.2f}  可行均值={avgs[5]*10:.2f}  合理均值={avgs[6]*10:.2f}")
        _print_gen("A", gen, population)

    return population, gen_stats


# ══════════════════════════════════════════════════════
#  实验B：四目标 MOEA/D
# ══════════════════════════════════════════════════════
def generate_weights_4(n: int, seed: int = 42) -> np.ndarray:
    """均匀 Dirichlet 权重向量，4个目标"""
    rng = np.random.default_rng(seed)
    candidates = rng.dirichlet(np.ones(4), size=n * 10)
    selected = [0]
    remaining = list(range(1, len(candidates)))
    while len(selected) < n:
        dists = np.min(
            np.linalg.norm(candidates[remaining][:, None] -
                           candidates[selected][None, :], axis=2), axis=1)
        best = remaining[int(np.argmax(dists))]
        selected.append(best)
        remaining.remove(best)
    return candidates[selected]


def tchebycheff_4(scores7: List[float], w: np.ndarray, ideal: np.ndarray) -> float:
    """4目标 Tchebycheff（取4个目标的分量）"""
    s4 = np.array([scores7[i] for i in OBJ_4_IDX])
    return float(np.max(w * np.abs(ideal - s4)))


def llm_crossover_4obj(pairs: List[Tuple[Individual, Individual, np.ndarray]]) -> List[Individual]:
    """四目标均衡导向的交叉变异"""
    tasks = []
    for idx, (p1, p2, w) in enumerate(pairs):
        dom_i = int(np.argmax(w))
        dom = OBJ_4[dom_i]
        w_str = " / ".join(f"{OBJ_4[j][:2]}={w[j]:.2f}" for j in range(4))
        tasks.append(
            f"任务{idx+1}:\n"
            f"  父代A: 【{p1.topic}】（{p1.domain}）{p1.description}\n"
            f"  父代B: 【{p2.topic}】（{p2.domain}）{p2.description}\n"
            f"  演化偏向: 重点提升「{dom}」（权重: {w_str}）"
        )
    n = len(pairs)
    prompt = f"""请对以下 {n} 组研究知识执行交叉变异，产生跨学科长尾科研知识。

{"".join(t + chr(10)*2 for t in tasks)}

【硬性要求】：
1. 新颖性 ≥ 6/10：必须具有突破性和前沿性
2. 可行性 ≥ 6/10：必须在2025年现有技术条件下原则上可开展
3. 合理性 ≥ 6/10：必须有已知理论基础支撑
4. 知识价值 ≥ 6/10：对基础科学有实质贡献

后代须融合两父代核心思想，根据演化偏向重点优化对应目标。

JSON格式（数组长度必须恰好为 {n}）：
{{"offspring": [
  {{"topic":"≤18字","domain":"具体交叉学科","description":"60-90字（含研究方法+理论基础）"}},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=5000, temperature=0.82)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it["description"],
                       domain=it.get("domain", "")) for it in items[:n]]


def run_exp_b(init_pop: List[Individual]) -> Tuple[List[Individual], list]:
    """实验B：四目标 MOEA/D"""
    print("\n" + "="*80)
    print("  实验B：四目标 MOEA/D（新颖性 × 知识价值 × 可行性 × 合理性）")
    print("="*80)
    population = list(init_pop)
    n = len(population)
    weights = generate_weights_4(n)
    neighborhoods = [
        np.argsort(np.linalg.norm(weights - weights[i], axis=1))[:N_NEIGHBORS].tolist()
        for i in range(n)
    ]

    gen_stats = []
    # 评估初始种群
    for ind, sc in zip(population, llm_evaluate_batch(population)):
        ind.scores = sc

    ideal = np.array([max(ind.scores[i] for ind in population) for i in OBJ_4_IDX])
    avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(7)]
    gen_stats.append({"gen": 0, "avgs": [round(a*10, 2) for a in avgs]})
    _print_gen("B", 0, population)

    for gen in range(1, N_GENS + 1):
        print(f"  [B] 第{gen:2d}/{N_GENS}代", end=" ", flush=True)
        t0 = time.time()
        pairs, parent_idx = [], []
        for i in range(n):
            nb = neighborhoods[i]
            i1, i2 = (random.sample(nb, 2) if len(nb) >= 2 else [nb[0], random.randint(0, n-1)])
            if i1 == i2: i2 = (i2 + 1) % n
            pairs.append((population[i1], population[i2], weights[i]))
            parent_idx.append(i)

        offspring = llm_crossover_4obj(pairs)
        for ind, sc in zip(offspring, llm_evaluate_batch(offspring)):
            ind.scores = sc
        for s in offspring:
            ideal = np.maximum(ideal, np.array([s.scores[i] for i in OBJ_4_IDX]))

        new_pop = list(population)
        for i, child in enumerate(offspring):
            for j in neighborhoods[parent_idx[i]]:
                if tchebycheff_4(child.scores, weights[j], ideal) <= \
                   tchebycheff_4(new_pop[j].scores, weights[j], ideal):
                    new_pop[j] = child
        population = new_pop
        avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(7)]
        gen_stats.append({"gen": gen, "avgs": [round(a*10, 2) for a in avgs]})
        print(f"✓ {time.time()-t0:.1f}s  新颖均值={avgs[4]*10:.2f}  可行均值={avgs[5]*10:.2f}  合理均值={avgs[6]*10:.2f}")
        _print_gen("B", gen, population)

    return population, gen_stats


# ══════════════════════════════════════════════════════
#  辅助显示
# ══════════════════════════════════════════════════════
def _print_gen(exp: str, gen: int, population: List[Individual]):
    label = "初始" if gen == 0 else f"第{gen:2d}代"
    avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(7)]
    print(f"\n  ─── 实验{exp} {label} ───")
    ranked = sorted(population, key=lambda x: sum(x.scores), reverse=True)
    for k, ind in enumerate(ranked[:3]):
        s = ind.scores
        bars = "  ".join(
            f"{OBJ_ALL[j][:2]}={'█'*int(s[j]*5)+'░'*(5-int(s[j]*5))}{s[j]*10:.0f}"
            for j in [4, 0, 5, 6]
        )
        print(f"  {k+1}. {ind.topic:<28} [{ind.domain[:20]}]")
        print(f"     {bars}")
    print()


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
def main():
    print("═"*80)
    print("  对比实验：单目标（新颖性） vs 四目标 MOEA/D")
    print(f"  模型: {MODEL}  种群: {POP_SIZE}  代数: {N_GENS}")
    print("  实验A：仅优化新颖性（贪心(μ+λ)-ES）")
    print("  实验B：四目标 MOEA/D（新颖性+知识价值+可行性+合理性）")
    print("═"*80)

    # ── 生成同一批初始种群（A/B 分开生成，保证公平）
    print("\n[初始化] 生成实验A初始种群（新颖性偏置）...")
    t0 = time.time()
    init_a = llm_generate_initial(POP_SIZE, "novelty")
    print(f"  完成 {time.time()-t0:.1f}s")

    print("\n[初始化] 生成实验B初始种群（四目标平衡）...")
    t0 = time.time()
    init_b = llm_generate_initial(POP_SIZE, "balanced")
    print(f"  完成 {time.time()-t0:.1f}s")

    # ── 运行实验A
    pop_a, stats_a = run_exp_a(init_a)

    # ── 运行实验B
    pop_b, stats_b = run_exp_b(init_b)

    # ── 汇总输出
    print("\n" + "═"*80)
    print("  最终对比（最后一代均值）")
    print("═"*80)
    sa = stats_a[-1]["avgs"]
    sb = stats_b[-1]["avgs"]
    names_short = ["知识", "社会", "长尾", "跨学", "新颖", "可行", "合理"]
    header = "       " + "".join(f"{n:>7}" for n in names_short)
    print(header)
    print(f"  实验A" + "".join(f"{sa[j]:>7.1f}" for j in range(7)))
    print(f"  实验B" + "".join(f"{sb[j]:>7.1f}" for j in range(7)))
    print(f"  差值 " + "".join(f"{sb[j]-sa[j]:>+7.1f}" for j in range(7)))

    # ── 保存结果
    def ind2d(ind):
        return {"topic": ind.topic, "domain": ind.domain,
                "description": ind.description, "scores": ind.scores}

    result = {
        "config": {"pop_size": POP_SIZE, "n_gens": N_GENS, "model": MODEL,
                   "obj_all": OBJ_ALL, "obj_4": OBJ_4},
        "exp_a": {
            "name": "单目标（仅新颖性）",
            "gen_stats": stats_a,
            "final_pop": [ind2d(x) for x in pop_a],
        },
        "exp_b": {
            "name": "四目标 MOEA/D",
            "gen_stats": stats_b,
            "final_pop": [ind2d(x) for x in pop_b],
        },
    }
    out = os.path.expanduser("~/moead_compare_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out}")
    print("  对比实验完成！")


if __name__ == "__main__":
    main()
