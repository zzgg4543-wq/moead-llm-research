#!/usr/bin/env python3
"""
CS 领域对比实验：单目标（新颖性）vs 四目标 MOEA/D
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
新特性：
  · 主题限定在计算机科学（允许 CS × 其他领域交叉）
  · 每个个体同时包含【研究主题】和【研究方案】
  · 进化过程同时优化主题新颖性和方案可行性/合理性
  · 四目标：新颖性 × 知识价值 × 可行性 × 合理性

研究方案结构：
  background  ── 背景与研究空白
  questions   ── 2-3 个核心研究问题
  methodology ── 具体技术路线（要有算法/实验细节）
  contributions── 预期贡献
"""

import json, random, os, time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from openai import OpenAI

# ══════════════════════════════════════════════════════
POP_SIZE    = 15
N_GENS      = 6
N_NEIGHBORS = 5
MODEL       = "deepseek-chat"

OBJ_4    = ["新颖性", "知识价值", "可行性", "合理性"]
OBJ_4_IDX= [4, 0, 5, 6]   # 在7维评分向量中的位置
OBJ_ALL  = ["知识价值", "社会影响", "长尾度", "跨学科性", "新颖性", "可行性", "合理性"]

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)


@dataclass
class Individual:
    topic: str
    description: str
    domain: str = ""
    plan: dict = field(default_factory=dict)   # 研究方案
    scores: List[float] = field(default_factory=lambda: [0.0] * 7)

    def plan_text(self) -> str:
        p = self.plan
        return (
            f"【背景】{p.get('background','')}\n"
            f"【问题】{'; '.join(p.get('questions',[]))}\n"
            f"【方法】{p.get('methodology','')}\n"
            f"【贡献】{'; '.join(p.get('contributions',[]))}"
        )

    def score_str(self) -> str:
        return "  ".join(f"{OBJ_ALL[j][:2]}={self.scores[j]*10:.0f}" for j in range(7))


# ══════════════════════════════════════════════════════
def call_llm(prompt: str, max_tokens=5000, temperature=0.7, retries=3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "你是计算机科学研究专家。所有回复必须是有效的JSON格式，不含代码块标记。"},
                    {"role": "user", "content": prompt},
                ]
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text[text.find("\n")+1:]
                if text.endswith("```"): text = text[:-3].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                print(f"  [重试{attempt+1}] JSON解析失败")
                temperature = min(temperature + 0.05, 1.0)
            else:
                raise
    raise RuntimeError("LLM 解析失败")


# ══════════════════════════════════════════════════════
#  生成初始种群（含研究方案）
# ══════════════════════════════════════════════════════
def llm_generate_initial(n: int, mode: str) -> List[Individual]:
    """mode: 'novelty'=仅追求新颖  'balanced'=四目标平衡"""
    if mode == "novelty":
        req = ("追求最高新颖性和前沿性，探索颠覆性计算机科学研究方向，"
               "不受当前技术限制，大胆假设。")
    else:
        req = ("在保持新颖前沿的同时，确保研究方案在当前（2025年）技术栈下"
               "原则上可开展（可行性≥6），且有扎实的理论依据（合理性≥6）。"
               "每个方案需给出具体算法/实验步骤。")

    prompt = f"""请生成 {n} 个计算机科学领域的长尾研究课题，每个课题附带完整研究方案。

核心要求：{req}

领域范围：聚焦计算机科学，允许 CS 与其他学科交叉（如 CS×生物、CS×物理、CS×认知科学等）。
长尾特征：非主流方向，现有研究稀少，但具有独特价值。

研究方案要包含具体技术细节，不能只有空洞描述。

JSON格式：
{{"items": [
  {{
    "topic": "研究课题标题（≤20字）",
    "domain": "CS子领域 × 交叉领域（如：图神经网络×蛋白质折叠）",
    "description": "课题概述（50-70字）",
    "plan": {{
      "background": "研究背景与现有空白（60-80字，说明为什么现在还没人做）",
      "questions": ["核心问题1（具体可测量）", "核心问题2", "核心问题3（可选）"],
      "methodology": "技术路线（80-100字，必须包含具体算法/模型/数据集/实验设计）",
      "contributions": ["预期贡献1（具体）", "预期贡献2", "预期贡献3（可选）"]
    }}
  }}
]}}"""
    data = call_llm(prompt, max_tokens=8000, temperature=0.9)
    items = data.get("items", [])
    if not items:
        for v in data.values():
            if isinstance(v, list): items = v; break
    result = []
    for it in items[:n]:
        ind = Individual(
            topic=it["topic"],
            description=it.get("description",""),
            domain=it.get("domain",""),
            plan=it.get("plan",{})
        )
        result.append(ind)
    return result


# ══════════════════════════════════════════════════════
#  批量7维评分（评估主题+方案的综合质量）
# ══════════════════════════════════════════════════════
def llm_evaluate_batch(individuals: List[Individual]) -> List[List[float]]:
    items_text = "\n\n".join(
        f"{i+1}. 【{ind.topic}】\n"
        f"   领域：{ind.domain}\n"
        f"   描述：{ind.description}\n"
        f"   背景：{ind.plan.get('background','')}\n"
        f"   方法：{ind.plan.get('methodology','')}\n"
        f"   贡献：{'; '.join(ind.plan.get('contributions',[]))}"
        for i, ind in enumerate(individuals)
    )
    prompt = f"""请对以下 {len(individuals)} 个计算机科学研究课题（含研究方案）进行七维评分（0-10整数，需有显著区分度）。

{items_text}

评分时同时考虑【研究主题】和【研究方案】的质量：

- knowledge（知识价值）：对CS基础理论/技术体系的贡献深度
- social（社会影响）：对工业界/社会的长远正向影响
- longtail（长尾度）：研究方向的稀缺性（主流热门=1, 极小众=10）
- interdiscip（跨学科性）：CS与其他领域的融合深度
- frontier（新颖性）：研究思路的突破程度，是否超越现有范式
- feasibility（可行性）：研究方案在2025年可实际执行的程度
  · 方案有具体算法+数据集+实验设计 → 6-8分
  · 方案完备且技术成熟 → 8-10分
  · 方案模糊或技术不成熟 → 1-5分
- rigor（合理性）：科学假设的逻辑严密性，方案是否基于已知理论
  · 有明确理论基础+可证伪假设 → 6-8分
  · 完全基于成熟理论 → 8-10分
  · 假设偏向科幻/缺乏依据 → 1-4分

JSON格式（数组长度必须恰好为 {len(individuals)}）：
{{"scores": [
  {{"knowledge":整数,"social":整数,"longtail":整数,"interdiscip":整数,
   "frontier":整数,"feasibility":整数,"rigor":整数}},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=3000, temperature=0.1)
    raw = data["scores"]
    return [[s["knowledge"]/10, s["social"]/10, s["longtail"]/10,
             s["interdiscip"]/10, s["frontier"]/10,
             s["feasibility"]/10, s["rigor"]/10] for s in raw]


# ══════════════════════════════════════════════════════
#  实验A：单目标交叉变异（仅追求新颖性）
# ══════════════════════════════════════════════════════
def llm_crossover_novelty(pairs: List[Tuple[Individual, Individual]]) -> List[Individual]:
    tasks = [
        f"任务{i+1}:\n"
        f"  父代A：【{p[0].topic}】方法：{p[0].plan.get('methodology','')[:80]}\n"
        f"  父代B：【{p[1].topic}】方法：{p[1].plan.get('methodology','')[:80]}"
        for i, p in enumerate(pairs)
    ]
    n = len(pairs)
    prompt = f"""请对以下 {n} 对计算机科学研究课题执行交叉变异，产生更新颖的后代课题+研究方案。

{"".join(t + chr(10)*2 for t in tasks)}

目标：后代必须比双亲都更新颖、更具突破性，追求最高前沿性。
可以大胆假设，不必担心当前可行性。
融合双亲的核心思路并产生创新性新方向。

JSON格式（数组长度必须恰好为 {n}）：
{{"offspring": [
  {{
    "topic": "≤20字",
    "domain": "CS子领域×交叉领域",
    "description": "50-70字",
    "plan": {{
      "background": "50-70字",
      "questions": ["问题1", "问题2"],
      "methodology": "70-90字，含具体技术",
      "contributions": ["贡献1", "贡献2"]
    }}
  }}
]}}"""
    data = call_llm(prompt, max_tokens=7000, temperature=0.92)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it.get("description",""),
                       domain=it.get("domain",""), plan=it.get("plan",{}))
            for it in items[:n]]


# ══════════════════════════════════════════════════════
#  实验B：四目标 MOEA/D 交叉变异
# ══════════════════════════════════════════════════════
def llm_crossover_4obj(pairs: List[Tuple[Individual, Individual, np.ndarray]]) -> List[Individual]:
    tasks = []
    for idx, (p1, p2, w) in enumerate(pairs):
        dom_i = int(np.argmax(w))
        dom = OBJ_4[dom_i]
        w_str = " / ".join(f"{OBJ_4[j][:2]}={w[j]:.2f}" for j in range(4))
        tasks.append(
            f"任务{idx+1}（偏向「{dom}」，权重: {w_str}）:\n"
            f"  父代A：【{p1.topic}】方法：{p1.plan.get('methodology','')[:80]}\n"
            f"  父代B：【{p2.topic}】方法：{p2.plan.get('methodology','')[:80]}"
        )
    n = len(pairs)
    prompt = f"""请对以下 {n} 对计算机科学研究课题执行交叉变异，产生均衡优质的后代课题+研究方案。

{"".join(t + chr(10)*2 for t in tasks)}

【硬性要求】每个后代必须同时满足：
1. 新颖性 ≥ 6：研究方向有突破性，不是已有工作的简单重复
2. 可行性 ≥ 6：方法论中必须包含具体算法/模型名称和可实施的实验步骤
3. 合理性 ≥ 6：必须基于已有CS理论，假设逻辑严密
4. 知识价值 ≥ 6：对CS理论或技术体系有实质贡献

根据每对的演化偏向重点优化对应目标，但不能以牺牲其他硬性要求为代价。
研究方案要有实质技术内容，不能只有口号。

JSON格式（数组长度必须恰好为 {n}）：
{{"offspring": [
  {{
    "topic": "≤20字",
    "domain": "CS子领域×交叉领域",
    "description": "50-70字",
    "plan": {{
      "background": "60-80字（说明现有空白）",
      "questions": ["核心问题1（可测量）", "核心问题2", "核心问题3（可选）"],
      "methodology": "80-100字（必须含：具体算法/架构名+数据集/实验环境+评估指标）",
      "contributions": ["理论贡献1", "技术贡献2", "应用贡献3（可选）"]
    }}
  }}
]}}"""
    data = call_llm(prompt, max_tokens=8000, temperature=0.82)
    items = data["offspring"]
    return [Individual(topic=it["topic"], description=it.get("description",""),
                       domain=it.get("domain",""), plan=it.get("plan",{}))
            for it in items[:n]]


# ══════════════════════════════════════════════════════
#  MOEA/D 工具函数
# ══════════════════════════════════════════════════════
def generate_weights_4(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = rng.dirichlet(np.ones(4), size=n * 10)
    selected = [0]
    remaining = list(range(1, len(candidates)))
    while len(selected) < n:
        dists = np.min(np.linalg.norm(
            candidates[remaining][:, None] - candidates[selected][None, :], axis=2), axis=1)
        best = remaining[int(np.argmax(dists))]
        selected.append(best)
        remaining.remove(best)
    return candidates[selected]


def tchebycheff_4(scores7, w, ideal):
    s4 = np.array([scores7[i] for i in OBJ_4_IDX])
    return float(np.max(w * np.abs(ideal - s4)))


# ══════════════════════════════════════════════════════
#  打印进度
# ══════════════════════════════════════════════════════
def print_gen(exp_label: str, gen: int, population: List[Individual]):
    label = "初始种群" if gen == 0 else f"第 {gen:2d} 代"
    avgs = [sum(x.scores[j] for x in population)/len(population) for j in range(7)]
    print(f"\n  ─── 实验{exp_label} {label} ───  "
          f"新颖={avgs[4]*10:.1f} 可行={avgs[5]*10:.1f} 合理={avgs[6]*10:.1f}")
    ranked = sorted(population, key=lambda x: sum(x.scores[i] for i in OBJ_4_IDX), reverse=True)
    for k, ind in enumerate(ranked[:3]):
        s = ind.scores
        row = "  ".join(f"{OBJ_ALL[j][:2]}={'█'*int(s[j]*5)+'░'*(5-int(s[j]*5))}{s[j]*10:.0f}"
                        for j in [4, 0, 5, 6])
        print(f"  {k+1}. {ind.topic:<30} [{ind.domain[:25]}]")
        print(f"     {row}")


# ══════════════════════════════════════════════════════
#  实验A 运行
# ══════════════════════════════════════════════════════
def run_exp_a(pop: List[Individual]):
    print("\n" + "═"*70)
    print("  实验A：单目标进化（仅新颖性）")
    print("═"*70)
    gen_stats = []
    for ind, sc in zip(pop, llm_evaluate_batch(pop)):
        ind.scores = sc
    avgs = [sum(x.scores[j] for x in pop)/len(pop) for j in range(7)]
    gen_stats.append({"gen": 0, "avgs": [round(a*10, 2) for a in avgs]})
    print_gen("A", 0, pop)

    for gen in range(1, N_GENS + 1):
        print(f"\n  [A] 第{gen:2d}/{N_GENS}代", end=" ", flush=True)
        t0 = time.time()
        pairs = [(random.choice(pop), random.choice(pop)) for _ in range(POP_SIZE)]
        offspring = llm_crossover_novelty(pairs)
        for ind, sc in zip(offspring, llm_evaluate_batch(offspring)):
            ind.scores = sc
        combined = pop + offspring
        combined.sort(key=lambda x: x.scores[4], reverse=True)
        pop = combined[:POP_SIZE]
        avgs = [sum(x.scores[j] for x in pop)/len(pop) for j in range(7)]
        gen_stats.append({"gen": gen, "avgs": [round(a*10, 2) for a in avgs]})
        print(f"✓ {time.time()-t0:.1f}s  新颖={avgs[4]*10:.2f}  可行={avgs[5]*10:.2f}  合理={avgs[6]*10:.2f}")
        print_gen("A", gen, pop)

    return pop, gen_stats


# ══════════════════════════════════════════════════════
#  实验B 运行
# ══════════════════════════════════════════════════════
def run_exp_b(pop: List[Individual]):
    print("\n" + "═"*70)
    print("  实验B：四目标 MOEA/D（新颖性 × 知识价值 × 可行性 × 合理性）")
    print("═"*70)
    n = len(pop)
    weights = generate_weights_4(n)
    neighborhoods = [
        np.argsort(np.linalg.norm(weights - weights[i], axis=1))[:N_NEIGHBORS].tolist()
        for i in range(n)
    ]
    gen_stats = []
    for ind, sc in zip(pop, llm_evaluate_batch(pop)):
        ind.scores = sc
    ideal = np.array([max(ind.scores[i] for ind in pop) for i in OBJ_4_IDX])
    avgs = [sum(x.scores[j] for x in pop)/len(pop) for j in range(7)]
    gen_stats.append({"gen": 0, "avgs": [round(a*10, 2) for a in avgs]})
    print_gen("B", 0, pop)

    for gen in range(1, N_GENS + 1):
        print(f"\n  [B] 第{gen:2d}/{N_GENS}代", end=" ", flush=True)
        t0 = time.time()
        pairs, p_idx = [], []
        for i in range(n):
            nb = neighborhoods[i]
            i1, i2 = (random.sample(nb, 2) if len(nb) >= 2 else [nb[0], random.randint(0, n-1)])
            if i1 == i2: i2 = (i2+1) % n
            pairs.append((pop[i1], pop[i2], weights[i]))
            p_idx.append(i)
        offspring = llm_crossover_4obj(pairs)
        for ind, sc in zip(offspring, llm_evaluate_batch(offspring)):
            ind.scores = sc
        for s in offspring:
            ideal = np.maximum(ideal, np.array([s.scores[i] for i in OBJ_4_IDX]))
        new_pop = list(pop)
        for i, child in enumerate(offspring):
            for j in neighborhoods[p_idx[i]]:
                if tchebycheff_4(child.scores, weights[j], ideal) <= \
                   tchebycheff_4(new_pop[j].scores, weights[j], ideal):
                    new_pop[j] = child
        pop = new_pop
        avgs = [sum(x.scores[j] for x in pop)/len(pop) for j in range(7)]
        gen_stats.append({"gen": gen, "avgs": [round(a*10, 2) for a in avgs]})
        print(f"✓ {time.time()-t0:.1f}s  新颖={avgs[4]*10:.2f}  可行={avgs[5]*10:.2f}  合理={avgs[6]*10:.2f}")
        print_gen("B", gen, pop)

    return pop, gen_stats


# ══════════════════════════════════════════════════════
#  Pareto 计算
# ══════════════════════════════════════════════════════
def find_pareto(pop: List[Individual]) -> List[Individual]:
    n = len(pop)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j: continue
            si, sj = pop[i].scores, pop[j].scores
            if (all(sj[k] >= si[k] for k in OBJ_4_IDX) and
                    any(sj[k] > si[k] for k in OBJ_4_IDX)):
                dominated[i] = True; break
    return [pop[i] for i in range(n) if not dominated[i]]


# ══════════════════════════════════════════════════════
#  保存结果
# ══════════════════════════════════════════════════════
def ind2d(ind: Individual) -> dict:
    return {"topic": ind.topic, "domain": ind.domain,
            "description": ind.description, "plan": ind.plan,
            "scores": ind.scores}


def main():
    print("═"*70)
    print("  CS 领域对比实验：单目标 vs 四目标 MOEA/D")
    print(f"  模型: {MODEL}  种群: {POP_SIZE}  代数: {N_GENS}")
    print("  特性：CS主题（允许交叉）+ 同步进化研究方案")
    print("═"*70)

    print("\n[初始化A] 生成初始种群（新颖性偏置）...")
    t0 = time.time()
    init_a = llm_generate_initial(POP_SIZE, "novelty")
    print(f"  完成 {time.time()-t0:.1f}s  生成 {len(init_a)} 个课题")

    print("\n[初始化B] 生成初始种群（四目标均衡）...")
    t0 = time.time()
    init_b = llm_generate_initial(POP_SIZE, "balanced")
    print(f"  完成 {time.time()-t0:.1f}s  生成 {len(init_b)} 个课题")

    pop_a, stats_a = run_exp_a(init_a)
    pop_b, stats_b = run_exp_b(init_b)
    pareto_b = find_pareto(pop_b)

    # 汇总
    sa, sb = stats_a[-1]["avgs"], stats_b[-1]["avgs"]
    print("\n" + "═"*70)
    print("  最终对比（第6代均值）")
    print("═"*70)
    names = ["知识","社会","长尾","跨学","新颖","可行","合理"]
    print("       " + "".join(f"{n:>7}" for n in names))
    print("  实验A" + "".join(f"{sa[j]:>7.1f}" for j in range(7)))
    print("  实验B" + "".join(f"{sb[j]:>7.1f}" for j in range(7)))
    print("  差值 " + "".join(f"{sb[j]-sa[j]:>+7.1f}" for j in range(7)))

    print(f"\n  实验B Pareto 最优解数: {len(pareto_b)}")
    print("  Pareto 代表（综合最优）:")
    for ind in sorted(pareto_b, key=lambda x: sum(x.scores[i] for i in OBJ_4_IDX), reverse=True)[:3]:
        print(f"    ◆ {ind.topic}  [{ind.domain[:30]}]")
        print(f"      {ind.score_str()}")

    # 保存
    result = {
        "config": {"pop_size": POP_SIZE, "n_gens": N_GENS, "model": MODEL,
                   "obj_all": OBJ_ALL, "obj_4": OBJ_4},
        "exp_a": {"name": "单目标（仅新颖性）",
                  "gen_stats": stats_a, "final_pop": [ind2d(x) for x in pop_a]},
        "exp_b": {"name": "四目标 MOEA/D",
                  "gen_stats": stats_b, "final_pop": [ind2d(x) for x in pop_b],
                  "pareto": [ind2d(x) for x in pareto_b]},
    }
    out = os.path.expanduser("~/moead-research/moead_cs_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out}")
    print("  实验完成！")


if __name__ == "__main__":
    main()
