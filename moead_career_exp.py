#!/usr/bin/env python3
"""
博后/职业选择 MOEA/D：多目标优化可执行职业选项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
· 个体 = 职业选项 + 研究路线 + 职业路径
· 11 维目标（含 Agent 时代、中美格局、启发性、上限）
· 交叉作用于 research_route，变异作用于路线细节
· 新选项含工业界/交叉机构
"""

import json, random, os, time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import numpy as np
from openai import OpenAI

# ══════════════════════════════════════════════════════
POP_SIZE         = 14
N_GENS           = 5
N_NEIGHBORS      = 4
N_NEW_OPTIONS    = 6
N_CROSSOVER_PAIRS= 6
P_MUTATION       = 0.35
N_EXPAND_OPTIONS = 3   # 每代扩展变异生成的新选项数（国内大厂/美国其他学校）
ALPHA_DEEP       = 1.3
FLOOR_MIN        = 0.5
MODEL            = "deepseek-chat"
TEMPERATURE_GEN  = 0.75
TEMPERATURE_EVAL = 0.2

OBJ_NAMES = ["契合","影响力","职业","资源","成长","可行","风险","Agent","格局","启发","上限"]
N_OBJ     = 11

PROFILE = """
Zhe Zhao，USTC+CityU HK 联合培养博士。
· 师从张青富系统学习多目标优化与进化算法。
· 开发 U2E 自动算法进化系统（早于 Alpha Evolve），在组学、PDE 等领域超越人类专家；探索矩阵乘法等底层优化。
· 斯坦福期间主导 Eureka 科研论文撰写系统，将假设与已有研究验证后自动生成可投稿论文。
· 一作：NIPS 2024/2025, ICML 2024/2025, AAAI 2024/2025, TKDE 等；长尾学习、图神经网络、因果图网络。
· 已有合作：NTU 安波(2025.3-7)、Stanford 仇夏婕 Qiu(2025.7-2026.6)。
"""

ANCHORS = [
    {"option_id":"ntu_liuyang", "institution":"NTU", "advisor":"刘洋 Liu Yang", "location":"Singapore",
     "field":"网络安全、软件工程、AI 鲁棒性/公平性"},
    {"option_id":"ntu_boan", "institution":"NTU", "advisor":"安波 Bo An", "location":"Singapore",
     "field":"多智能体、博弈论、强化学习、优化"},
    {"option_id":"stanford_lecong", "institution":"Stanford", "advisor":"丛乐 Le Cong", "location":"Bay Area",
     "field":"基因编辑、单细胞、ML 优化蛋白质/RNA 设计"},
    {"option_id":"stanford_qiu", "institution":"Stanford", "advisor":"仇夏婕 Xiaojie Qiu", "location":"Bay Area",
     "field":"单细胞计算生物学、发育轨迹、可微分深度学习"},
    {"option_id":"stanford_jure", "institution":"Stanford", "advisor":"Jure Leskovec", "location":"Bay Area",
     "field":"图神经网络、PyG、社会网络、药物发现"},
    {"option_id":"stanford_jameszou", "institution":"Stanford", "advisor":"James Zou", "location":"Bay Area",
     "field":"可信AI、公平性、医疗AI、Data Shapley、TextGrad"},
]

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY",""), base_url="https://api.deepseek.com")


@dataclass
class Individual:
    option_id: str
    type: str  # postdoc | industry_researcher | hybrid
    anchor: dict
    research_route: dict
    career_path: dict
    scores: List[float] = field(default_factory=lambda: [0.0] * N_OBJ)

    def to_dict(self) -> dict:
        return {"option_id":self.option_id, "type":self.type, "anchor":self.anchor,
                "research_route":self.research_route, "career_path":self.career_path, "scores":self.scores}


def call_llm(prompt: str, max_tokens=4096, temperature=0.7, retries=3) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role":"system","content":"你是职业发展与学术规划专家。所有回复必须是有效的JSON格式，不含代码块标记。"},
                    {"role":"user","content":prompt},
                ]
            )
            text = resp.choices[0].message.content.strip()
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}")+1
                if start>=0 and end>start: text = text[start:end]
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                print(f"  [重试{attempt+1}]")
                temperature = min(temperature + 0.05, 1.0)
            else:
                raise
    raise RuntimeError("LLM 解析失败")


# ══════════════════════════════════════════════════════
#  初始种群：为锚点生成 route + 新选项
# ══════════════════════════════════════════════════════
def build_initial_population() -> List[Individual]:
    # 1. 为每个锚点生成 research_route + career_path
    items_text = "\n".join(f"- {a['option_id']}: {a['advisor']} @ {a['institution']}, {a['field']}" for a in ANCHORS)
    prompt = f"""请为以下 6 个博后选项分别生成【可执行的研究路线】和【职业路径】，候选人背景：
{PROFILE}

选项列表：
{items_text}

每个选项需给出：
1. research_route: year1_focus, year2_focus, key_outputs(2-4项), continuity_from_background
2. career_path: exit_options, pros(2-3), cons(1-2), risk_level(low/medium/high)

路线必须可在该导师/机构下实际执行，且与候选人 U2E/Eureka/MOEA/图学习 背景有延续性。

JSON格式（数组长度必须为6）：
{{"items": [
  {{
    "option_id": "ntu_liuyang",
    "anchor": {{"institution_or_company":"NTU","advisor_or_team":"刘洋","location":"Singapore"}},
    "research_route": {{
      "year1_focus": "50字内",
      "year2_focus": "50字内",
      "key_outputs": ["产出1","产出2","产出3"],
      "continuity_from_background": "30字内"
    }},
    "career_path": {{
      "exit_options": ["教职","大厂"],
      "pros": ["优势1","优势2"],
      "cons": ["劣势1"],
      "risk_level": "low"
    }}
  }},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=6000, temperature=TEMPERATURE_GEN)
    items = data.get("items", data.get("options", []))
    if not items:
        for v in data.values():
            if isinstance(v, list) and len(v) >= 6: items = v; break

    population = []
    for i, it in enumerate(items[:6]):
        aid = ANCHORS[i]["option_id"] if i < len(ANCHORS) else it.get("option_id", f"anchor_{i}")
        population.append(Individual(
            option_id=aid,
            type="postdoc",
            anchor=it.get("anchor", {"institution_or_company":ANCHORS[i]["institution"], "advisor_or_team":ANCHORS[i]["advisor"], "location":ANCHORS[i]["location"]}),
            research_route=it.get("research_route", {}),
            career_path=it.get("career_path", {})
        ))

    # 2. 生成新选项：工业界 + 国内大厂 + 美国其他学校
    prompt2 = f"""请生成 6 个【真实存在】的职业选项，候选人背景：{PROFILE}

要求（每类至少1个）：
- 2个美国/欧洲工业界AI研究岗（DeepMind、Anthropic、Meta FAIR、Google、OpenAI 等）
- 2个【国内大厂】AI研究岗（阿里达摩院、腾讯AI Lab、字节、华为诺亚、百度、智谱、月之暗面等）
- 2个【美国其他学校】博后或教职（MIT、Berkeley、CMU、Princeton、Yale、Columbia、UCLA 等，具体导师+方向）
每个必须有清晰的 pros、cons、career_path、research_route（可执行的研究主线），option_id 用英文简短标识。

JSON格式：
{{"items": [
  {{
    "option_id": "industry_xxx 或 china_xxx 或 us_xxx",
    "type": "industry_researcher 或 postdoc 或 faculty",
    "anchor": {{"institution_or_company":"","advisor_or_team":"","location":""}},
    "research_route": {{"year1_focus":"","year2_focus":"","key_outputs":[],"continuity_from_background":""}},
    "career_path": {{"exit_options":[],"pros":[],"cons":[],"risk_level":""}}
  }},
  ...
]}}"""
    data2 = call_llm(prompt2, max_tokens=5000, temperature=TEMPERATURE_GEN)
    items2 = data2.get("items", data2.get("options", []))
    for it in items2[:min(N_NEW_OPTIONS, 8)]:
        oid = it.get("option_id", f"new_{len(population)}")
        population.append(Individual(
            option_id=oid,
            type=it.get("type", "industry_researcher"),
            anchor=it.get("anchor", {}),
            research_route=it.get("research_route", {}),
            career_path=it.get("career_path", {})
        ))

    return population


# ══════════════════════════════════════════════════════
#  11 维批量评估
# ══════════════════════════════════════════════════════
def evaluate_batch(individuals: List[Individual]) -> List[List[float]]:
    lines = []
    for i, ind in enumerate(individuals):
        r = ind.research_route
        c = ind.career_path
        lines.append(f"{i+1}. 【{ind.option_id}】{ind.anchor.get('advisor_or_team','')} @ {ind.anchor.get('institution_or_company','')}\n"
                    f"   Y1: {r.get('year1_focus','')} | Y2: {r.get('year2_focus','')}\n"
                    f"   continuity: {r.get('continuity_from_background','')}\n"
                    f"   pros: {c.get('pros',[])} | cons: {c.get('cons',[])}")

    prompt = f"""对以下 {len(individuals)} 个职业选项进行 11 维评分（0-10 整数，需有区分度）。候选人背景：{PROFILE}

{chr(10).join(lines)}

11 维定义：
1.契合-与MOEA/LLM/AI4Science背景匹配
2.影响力-顶会顶刊、领域认可
3.职业-教职/工业界/创业支持
4.资源-经费算力招聘稳定
5.成长-新技能新领域独立性
6.可行-拿到offer概率
7.风险-时间沉没退出成本
8.Agent-与AI agent时代趋势契合
9.格局-中美博弈下定位与弹性
10.启发-对突破性思考的激发
11.上限-职业与认知天花板

JSON格式：
{{"scores": [{{"f1":n,"f2":n,...,"f11":n}}, ...]}}"""
    data = call_llm(prompt, max_tokens=4000, temperature=TEMPERATURE_EVAL)
    raw = data.get("scores", [])
    result = []
    for s in raw:
        if isinstance(s, dict):
            row = [s.get(f"f{i+1}", s.get(OBJ_NAMES[i], 5))/10.0 for i in range(N_OBJ)]
        else:
            row = [float(x)/10.0 for x in s[:N_OBJ]] if isinstance(s,(list,tuple)) else [0.5]*N_OBJ
        result.append(row[:N_OBJ])
    return result


# ══════════════════════════════════════════════════════
#  交叉（作用于 research_route）
# ══════════════════════════════════════════════════════
def crossover_route(parents: List[Tuple[Individual, Individual]]) -> List[Individual]:
    tasks = []
    for i, (A, B) in enumerate(parents):
        rA, rB = A.research_route, B.research_route
        tasks.append(f"对{i+1}: A={A.option_id}(Y1:{rA.get('year1_focus','')[:40]}...)\n"
                    f"     B={B.option_id}(Y1:{rB.get('year1_focus','')[:40]}...)")
    n = len(parents)
    prompt = f"""对以下 {n} 对职业选项做【研究路线交叉】。融合两条路线的方法论与应用域，形成可执行的新路线。
候选人背景：{PROFILE}

{"".join(t + chr(10)*2 for t in tasks)}

规则：后代路线必须能在父代A或B的 anchor 下执行。选更匹配的 anchor 作为后代 anchor。
career_path 从双亲择优或融合。

JSON格式（数组长度{n}）：
{{"offspring": [
  {{"option_id":"继承A或B的id","anchor":{{}},"research_route":{{"year1_focus":"","year2_focus":"","key_outputs":[],"continuity_from_background":""}},"career_path":{{"exit_options":[],"pros":[],"cons":[],"risk_level":""}}}},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=6000, temperature=TEMPERATURE_GEN)
    off = data.get("offspring", [])
    return [Individual(
        option_id=o.get("option_id", f"cross_{i}"),
        type="postdoc",
        anchor=o.get("anchor", parents[i][0].anchor),
        research_route=o.get("research_route", {}),
        career_path=o.get("career_path", {})
    ) for i, o in enumerate(off[:n])]


# ══════════════════════════════════════════════════════
#  变异
# ══════════════════════════════════════════════════════
def mutate_batch(individuals: List[Individual]) -> List[Individual]:
    if not individuals:
        return []
    mut_types = ["year1_year2", "key_outputs", "continuity", "exit_options"]
    lines = "\n".join(f"{i+1}. {ind.option_id}: Y1={ind.research_route.get('year1_focus','')[:50]}" for i, ind in enumerate(individuals))
    prompt = f"""对以下 {len(individuals)} 个选项做【轻度变异】：微调 research_route 或 career_path 的一处。
候选人背景：{PROFILE}

{lines}

变异类型随机选一种应用：时间线微调/产出目标/延续策略/退出路径。
保持选项可执行性，只做小幅度改进。

JSON格式：
{{"mutated": [
  {{"option_id":"...","research_route":{{...}},"career_path":{{...}}}},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=5000, temperature=TEMPERATURE_GEN)
    mut = data.get("mutated", [])
    result = []
    for i, m in enumerate(mut[:len(individuals)]):
        ind = individuals[i]
        result.append(Individual(
            option_id=m.get("option_id", ind.option_id),
            type=ind.type,
            anchor=ind.anchor,
            research_route=m.get("research_route", ind.research_route),
            career_path=m.get("career_path", ind.career_path)
        ))
    return result


# ══════════════════════════════════════════════════════
#  扩展变异：生成国内大厂 / 美国其他学校新选项
# ══════════════════════════════════════════════════════
def expand_population(gen: int, existing_ids: List[str]) -> List[Individual]:
    """每代生成若干新选项，覆盖国内大厂、美国其他学校博后/教职。"""
    prompt = f"""请生成 {N_EXPAND_OPTIONS} 个【真实存在】的职业选项，作为演化扩展。候选人背景：{PROFILE}

要求（覆盖不同类别）：
- 至少1个【国内大厂】AI研究岗：阿里达摩院、腾讯AI Lab、字节火山引擎、华为诺亚、百度、智谱、月之暗面、 minimax 等
- 至少1个【美国其他学校】博后或教职：MIT、Berkeley、CMU、Princeton、Yale、Columbia、UCLA、UCSD、Cornell 等，需给出具体导师或实验室方向
- 可与已有选项重复机构但需不同团队/方向
已有 option_id 前缀参考（避免重复）：{existing_ids[:8]}

每个必须有 research_route、career_path、pros、cons。option_id 用简短英文如 china_alibaba、us_berkeley_xxx。

JSON格式：
{{"items": [
  {{
    "option_id": "xxx",
    "type": "industry_researcher 或 postdoc",
    "anchor": {{"institution_or_company":"","advisor_or_team":"","location":""}},
    "research_route": {{"year1_focus":"","year2_focus":"","key_outputs":[],"continuity_from_background":""}},
    "career_path": {{"exit_options":[],"pros":[],"cons":[],"risk_level":""}}
  }},
  ...
]}}"""
    data = call_llm(prompt, max_tokens=5000, temperature=TEMPERATURE_GEN)
    items = data.get("items", data.get("options", []))
    result = []
    for i, it in enumerate(items[:N_EXPAND_OPTIONS]):
        oid = it.get("option_id", f"expand_gen{gen}_{i}")
        result.append(Individual(
            option_id=oid,
            type=it.get("type", "industry_researcher"),
            anchor=it.get("anchor", {}),
            research_route=it.get("research_route", {}),
            career_path=it.get("career_path", {})
        ))
    return result


# ══════════════════════════════════════════════════════
#  MOEA/D 工具
# ══════════════════════════════════════════════════════
def generate_weights(n: int, alpha_deep: float = ALPHA_DEEP) -> np.ndarray:
    rng = np.random.default_rng(42)
    alpha = [1.0]*7 + [alpha_deep]*4
    w = rng.dirichlet(alpha, size=n*15)
    sel = [0]
    rem = list(range(1, len(w)))
    while len(sel) < n:
        d = np.min(np.linalg.norm(w[rem][:, None] - w[sel][None, :], axis=2), axis=1)
        idx = rem[int(np.argmax(d))]
        sel.append(idx)
        rem.remove(idx)
    return w[sel]


def tchebycheff(scores: List[float], w: np.ndarray, ideal: np.ndarray) -> float:
    s = np.array(scores[:N_OBJ])
    pen = 0.0
    for j in [7, 8, 9, 10]:  # Agent, 格局, 启发, 上限
        if s[j] < FLOOR_MIN:
            pen += (FLOOR_MIN - s[j]) * 2.0
    return float(np.max(w * np.abs(ideal - s))) + pen


def find_pareto(pop: List[Individual]) -> List[int]:
    n = len(pop)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j: continue
            si, sj = pop[i].scores, pop[j].scores
            if all(sj[k] >= si[k] for k in range(N_OBJ)) and any(sj[k] > si[k] for k in range(N_OBJ)):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════
def main():
    print("═"*70)
    print("  博后/职业选择 MOEA/D")
    print(f"  种群={POP_SIZE} 代数={N_GENS} 目标={N_OBJ}维")
    print("═"*70)

    # 0. 初始种群
    print("\n[1/5] 构建初始种群（6锚点+6新选项，含国内大厂/美国其他学校）...")
    t0 = time.time()
    pop = build_initial_population()
    # 若不足 POP_SIZE，用扩展选项补足
    if len(pop) < POP_SIZE:
        exp = expand_population(0, [x.option_id for x in pop])
        for ind in exp[:POP_SIZE - len(pop)]:
            pop.append(ind)
        print(f"  扩展补足至 {len(pop)} 个选项")
    print(f"  完成 {len(pop)} 个选项 {time.time()-t0:.1f}s")

    # 1. 评估
    print("\n[2/5] 11维评估...")
    t0 = time.time()
    scores = evaluate_batch(pop)
    for ind, sc in zip(pop, scores):
        ind.scores = sc[:N_OBJ]
    print(f"  完成 {time.time()-t0:.1f}s")
    avgs = np.mean([ind.scores for ind in pop], axis=0)
    print("  初始均值:", " ".join(f"{OBJ_NAMES[j]}={avgs[j]*10:.1f}" for j in range(N_OBJ)))

    n = len(pop)
    weights = generate_weights(n)
    neighbors = [np.argsort(np.linalg.norm(weights - weights[i], axis=1))[:N_NEIGHBORS].tolist() for i in range(n)]
    ideal = np.array([max(ind.scores[j] for ind in pop) for j in range(N_OBJ)])
    gen_stats = [{"gen": 0, "avgs": [round(avgs[j]*10, 2) for j in range(N_OBJ)]}]

    # 日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "career_logs")
    os.makedirs(log_dir, exist_ok=True)
    def save_gen_log(g: int, population: List[Individual]):
        p = [ind.to_dict() for ind in population]
        for d in p:
            d["scores"] = [round(x*10, 2) for x in d["scores"]]
        with open(os.path.join(log_dir, f"gen_{g:02d}_population.json"), "w", encoding="utf-8") as f:
            json.dump({"gen": g, "population": p, "avgs": gen_stats[g]["avgs"] if g < len(gen_stats) else []},
                      f, ensure_ascii=False, indent=2)

    save_gen_log(0, pop)

    # 2. 演化
    for gen in range(1, N_GENS + 1):
        print(f"\n[3/5] 第 {gen}/{N_GENS} 代...")
        t0 = time.time()
        pareto_idx = find_pareto(pop)
        pareto = [pop[i] for i in pareto_idx]

        # 交叉
        pairs = []
        for _ in range(N_CROSSOVER_PAIRS):
            if len(pareto) >= 2:
                a, b = random.sample(pareto, 2)
                pairs.append((a, b))
        offspring = crossover_route(pairs) if pairs else []

        # 变异
        n_mut = max(1, int(len(pop) * P_MUTATION))
        mut_cand = random.sample(pop, min(n_mut, len(pop)))
        mutated = mutate_batch(mut_cand) if mut_cand else []

        # 扩展：国内大厂 / 美国其他学校 新选项
        expanded = expand_population(gen, [x.option_id for x in pop])
        print(f"  扩展变异: +{len(expanded)} 个新选项（国内大厂/美国其他学校）")

        new_all = offspring + mutated + expanded
        if new_all:
            sc_new = evaluate_batch(new_all)
            for ind, sc in zip(new_all, sc_new):
                ind.scores = sc[:N_OBJ]
            for ind in new_all:
                ideal = np.maximum(ideal, np.array(ind.scores))

        # 更新种群：MOEA/D 邻域替换
        combined = list(pop)
        for child in new_all:
            for i in range(n):
                for j in neighbors[i]:
                    if tchebycheff(child.scores, weights[j], ideal) <= tchebycheff(combined[j].scores, weights[j], ideal):
                        combined[j] = child
                        break
        pop = combined

        avgs = np.mean([ind.scores for ind in pop], axis=0)
        gen_stats.append({"gen": gen, "avgs": [round(avgs[j]*10, 2) for j in range(N_OBJ)]})
        save_gen_log(gen, pop)
        print(f"  完成 {time.time()-t0:.1f}s  契合={avgs[0]*10:.1f} Agent={avgs[7]*10:.1f} 上限={avgs[10]*10:.1f}  已保存 gen_{gen:02d}_population.json")

    # 3. Pareto 排序与推荐
    print("\n[4/5] Pareto 排序与推荐...")
    pareto_idx = find_pareto(pop)
    pareto_list = [pop[i] for i in pareto_idx]
    weighted = [(ind, sum(ind.scores)) for ind in pareto_list]
    weighted.sort(key=lambda x: x[1], reverse=True)
    ranking = [x[0].option_id for x in weighted]

    for ind in pop:
        ind.rank = ranking.index(ind.option_id) + 1 if ind.option_id in ranking else 999
        ind.pareto_front = ind.option_id in ranking

    # 推荐理由
    top5 = weighted[:5]
    rec_prompt = f"""对以下 5 个职业选项各写一句推荐理由、pros、cons、best_for（最适合的路径）。
候选人背景：{PROFILE}

""" + "\n".join(f"{i+1}. {ind.option_id}: {ind.anchor.get('advisor_or_team','')} @ {ind.anchor.get('institution_or_company','')}" for i, (ind,_) in enumerate(top5))

    rec_prompt += """

JSON格式：
{"recommendations": [{"option_id":"","brief":"","pros":[],"cons":[],"best_for":""}, ...]}"""
    rec_data = call_llm(rec_prompt, max_tokens=3000, temperature=0.3)
    recommendations = rec_data.get("recommendations", [])

    # 保存
    out = {
        "profile_summary": PROFILE.strip(),
        "config": {"pop_size": POP_SIZE, "n_gens": N_GENS, "n_expand_per_gen": N_EXPAND_OPTIONS,
                   "objectives": OBJ_NAMES, "log_dir": "career_logs"},
        "evaluated_options": [ind.to_dict() for ind in pop],
        "pareto_ranking": ranking,
        "recommendations": recommendations,
        "gen_stats": gen_stats,
    }
    for i, ind in enumerate(pop):
        if i < len(out["evaluated_options"]):
            out["evaluated_options"][i]["rank"] = getattr(ind, "rank", 0)
            out["evaluated_options"][i]["pareto_front"] = getattr(ind, "pareto_front", False)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moead_career_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {path}")
    print(f"  每代种群日志: {log_dir}/gen_XX_population.json")

    # 打印 Top5
    print("\n" + "═"*70)
    print("  Pareto Top 5")
    print("═"*70)
    for i, (ind, _) in enumerate(top5, 1):
        s = ind.scores
        print(f"\n  {i}. {ind.option_id}")
        print(f"     {ind.anchor.get('advisor_or_team','')} @ {ind.anchor.get('institution_or_company','')}")
        print(f"     契合={s[0]*10:.0f} Agent={s[7]*10:.0f} 格局={s[8]*10:.0f} 启发={s[9]*10:.0f} 上限={s[10]*10:.0f}")
        if i <= len(recommendations):
            r = recommendations[i-1]
            print(f"     推荐: {r.get('brief','')[:80]}...")
    print("\n  实验完成！")


if __name__ == "__main__":
    main()
