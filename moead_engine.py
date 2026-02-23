#!/usr/bin/env python3
"""
通用 MOEA/D 预测引擎
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：目标（可配置）、现有信息（profile）、超参数
输出：Pareto 最优预测/推荐
"""

import json, random, os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI


@dataclass
class Individual:
    option_id: str
    content: Dict[str, Any]
    scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"option_id": self.option_id, "content": self.content, "scores": self.scores}
        if hasattr(self, "rank"):
            d["rank"] = self.rank
        if hasattr(self, "pareto_front"):
            d["pareto_front"] = self.pareto_front
        return d


def call_llm(
    prompt: str,
    client: OpenAI,
    model: str = "deepseek-chat",
    max_tokens: int = 4096,
    config: Optional[dict] = None,
    temperature: float = 0.75,
    retries: int = 3,
) -> dict:
    cfg = config or {}
    model = cfg.get("model", model)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "你是多目标决策与预测专家。所有回复必须是有效JSON，不含代码块。"},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content.strip()
            if "```" in text:
                start, end = text.find("{"), text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                temperature = min(temperature + 0.05, 1.0)
            else:
                raise
    raise RuntimeError("LLM 解析失败")


def _constraints_block(config: dict) -> str:
    cs = config.get("constraints") or []
    if isinstance(cs, str):
        cs = [cs] if cs.strip() else []
    if not cs:
        return ""
    lines = "\n".join(f"· {c}" for c in cs if c and str(c).strip())
    return f"\n【硬约束】（所有选项必须满足，违反者将被剔除或严重降分）\n{lines}\n" if lines else ""


def _is_graduate_domain(domain: str) -> bool:
    kw = ("研究生", "读研", "硕博", "三年", "发展路径", "学术生涯")
    return any(k in domain for k in kw)


def generate_initial(
    profile: str,
    domain: str,
    config: dict,
    client: OpenAI,
) -> List[Individual]:
    n = config.get("pop_size", 12)
    cblock = _constraints_block(config)
    grad_hint = ""
    if _is_graduate_domain(domain):
        grad_hint = """
4. 若为研究生发展路径，每个选项需体现：分年安排（Year1/2/3）、论文目标（CCF-A 等）、资金/收入来源、与导师/学长合作方式。"""
    prompt = f"""请根据以下信息，在「{domain}」领域生成 {n} 个可执行的选项/路径/方案，供多目标优化评估。

【现有信息 / 背景】
{profile}
{cblock}
要求：
1. 每个选项必须具体、可执行，不能空洞
2. 选项之间要有差异（不同类型、方向、路径）
3. 与背景信息有合理关联{grad_hint}

JSON格式：
{{"items": [
  {{
    "option_id": "唯一英文标识",
    "content": {{
      "summary": "一句话概述（50字内）",
      "detail": "详细说明（100-150字）",
      "pros": ["优势1", "优势2"],
      "cons": ["劣势1"]
    }}
  }},
  ...
]}}"""
    data = call_llm(prompt, client, max_tokens=6000, temperature=config.get("temperature_gen", 0.8), config=config)
    items = data.get("items", data.get("options", []))
    if not items:
        for v in data.values():
            if isinstance(v, list) and len(v) > 0:
                items = v
                break
    return [
        Individual(option_id=it.get("option_id", f"opt_{i}"), content=it.get("content", it))
        for i, it in enumerate(items[:n])
    ]


def evaluate_batch(
    individuals: List[Individual],
    profile: str,
    objectives: List[dict],
    client: OpenAI,
    config: dict,
) -> List[List[float]]:
    n_obj = len(objectives)
    obj_text = "\n".join(f"{i+1}. {o['name']}: {o.get('definition', o.get('desc', ''))}" for i, o in enumerate(objectives))
    cblock = _constraints_block(config)
    lines = []
    for i, ind in enumerate(individuals):
        c = ind.content
        s = c.get("summary", c.get("detail", str(c)[:200]))
        lines.append(f"{i+1}. 【{ind.option_id}】{s}\n    pros: {c.get('pros',[])} | cons: {c.get('cons',[])}")

    constraint_note = ""
    if cblock:
        constraint_note = f"{cblock}\n若某选项违反上述硬约束，该选项在相关维度上给 0 分。\n\n"
    prompt = f"""对以下 {len(individuals)} 个选项进行多目标评分。背景：{profile[:300]}
{constraint_note}
【评分维度】（0-10整数，需有区分度）
{obj_text}

【选项】
{chr(10).join(lines)}

JSON格式：
{{"scores": [{{"f1":n,"f2":n,...}}, ...]}}
字段数必须与维度数一致，可用 f1,f2,... 或维度名。"""
    data = call_llm(prompt, client, max_tokens=4000, temperature=config.get("temperature_eval", 0.2), config=config)
    raw = data.get("scores", [])
    result = []
    obj_names = [o["name"] for o in objectives]
    for s in raw:
        if isinstance(s, dict):
            row = []
            for i in range(n_obj):
                v = s.get(f"f{i+1}", s.get(obj_names[i], 5))
                row.append(float(v) / 10.0 if isinstance(v, (int, float)) else 0.5)
            result.append(row[:n_obj])
        else:
            result.append([float(x) / 10.0 for x in (s[:n_obj] if isinstance(s, (list, tuple)) else [0.5] * n_obj)])
    return result


def crossover_route(
    parents: List[tuple],
    profile: str,
    domain: str,
    client: OpenAI,
    config: dict,
) -> List[Individual]:
    if not parents:
        return []
    n = len(parents)
    tasks = "\n\n".join(
        f"对{i+1}: A={p[0].option_id}({str(p[0].content.get('summary',''))[:60]}...)  B={p[1].option_id}({str(p[1].content.get('summary',''))[:60]}...)"
        for i, p in enumerate(parents)
    )
    cblock = _constraints_block(config)
    prompt = f"""在「{domain}」领域，对以下 {n} 对选项做交叉融合，产生新的可行选项。背景：{profile[:200]}
{cblock}
{tasks}

规则：后代需结合双亲优势，且可执行；必须满足硬约束。JSON格式：
{{"offspring": [
  {{"option_id":"cross_1","content":{{"summary":"...","detail":"...","pros":[],"cons":[]}}}},
  ...
]}}"""
    data = call_llm(prompt, client, max_tokens=5000, temperature=config.get("temperature_gen", 0.75), config=config)
    off = data.get("offspring", [])
    return [
        Individual(option_id=o.get("option_id", f"cross_{i}"), content=o.get("content", o))
        for i, o in enumerate(off[:n])
    ]


def mutate_batch(
    individuals: List[Individual],
    profile: str,
    domain: str,
    client: OpenAI,
    config: dict,
) -> List[Individual]:
    if not individuals:
        return []
    cblock = _constraints_block(config)
    lines = "\n".join(f"{i+1}. {ind.option_id}: {ind.content.get('summary','')[:80]}" for i, ind in enumerate(individuals))
    prompt = f"""对以下选项做轻度变异（微调一处），领域：{domain}。背景：{profile[:150]}
{cblock}
{lines}

规则：变异后仍须满足硬约束。JSON格式：{{"mutated": [{{"option_id":"...","content":{{"summary":"...","detail":"...","pros":[],"cons":[]}}}}, ...]}}"""
    data = call_llm(prompt, client, max_tokens=4000, temperature=0.8, config=config)
    mut = data.get("mutated", [])
    return [
        Individual(option_id=m.get("option_id", ind.option_id), content=m.get("content", ind.content))
        for i, (ind, m) in enumerate(zip(individuals, mut[:len(individuals)]))
    ]


def expand_options(
    gen: int,
    existing_ids: List[str],
    profile: str,
    domain: str,
    config: dict,
    client: OpenAI,
) -> List[Individual]:
    n = config.get("n_expand", 3)
    cblock = _constraints_block(config)
    prompt = f"""在「{domain}」领域，生成 {n} 个与已有选项不同的新选项。背景：{profile[:200]}
{cblock}
已有 option_id 参考：{existing_ids[:10]}

规则：新选项须满足硬约束。JSON：{{"items":[{{"option_id":"...","content":{{"summary":"...","detail":"...","pros":[],"cons":[]}}}}, ...]}}"""
    data = call_llm(prompt, client, max_tokens=4000, temperature=0.85, config=config)
    items = data.get("items", data.get("offspring", []))
    return [
        Individual(option_id=it.get("option_id", f"expand_{gen}_{i}"), content=it.get("content", it))
        for i, it in enumerate(items[:n])
    ]


def generate_weights(n: int, n_obj: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    w = rng.dirichlet(np.ones(n_obj), size=n * 20)
    sel, rem = [0], list(range(1, len(w)))
    while len(sel) < n:
        d = np.min(np.linalg.norm(w[rem][:, None] - w[sel][None, :], axis=2), axis=1)
        idx = rem[int(np.argmax(d))]
        sel.append(idx)
        rem.remove(idx)
    return w[sel]


def tchebycheff(scores: List[float], w: np.ndarray, ideal: np.ndarray) -> float:
    s = np.array(scores[:len(w)])
    return float(np.max(w * np.abs(ideal - s)))


def find_pareto(pop: List[Individual], n_obj: int) -> List[int]:
    n = len(pop)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            si = np.array(pop[i].scores[:n_obj])
            sj = np.array(pop[j].scores[:n_obj])
            if np.all(sj >= si) and np.any(sj > si):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def run(
    profile: str,
    domain: str,
    objectives: List[dict],
    config: Optional[dict] = None,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    progress_callback=None,
) -> dict:
    cfg = config or {}
    pop_size = cfg.get("pop_size", 10)
    n_gens = cfg.get("n_gens", 3)
    n_neighbors = cfg.get("n_neighbors", 4)
    n_crossover = cfg.get("n_crossover_pairs", 4)
    p_mut = cfg.get("p_mutation", 0.3)
    model = cfg.get("model", "deepseek-chat")

    client = OpenAI(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=base_url,
    )
    n_obj = len(objectives)
    obj_names = [o["name"] for o in objectives]

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # 1. 初始种群
    log("生成初始种群...")
    pop = generate_initial(profile, domain, cfg, client)
    if len(pop) < pop_size:
        pop = pop + [Individual(option_id=f"pad_{i}", content={"summary": "占位"}) for i in range(pop_size - len(pop))]
    pop = pop[:pop_size]

    scores = evaluate_batch(pop, profile, objectives, client, cfg)
    for ind, sc in zip(pop, scores):
        ind.scores = sc[:n_obj]
    ideal = np.array([max(ind.scores[j] for ind in pop) for j in range(n_obj)])
    weights = generate_weights(pop_size, n_obj)
    neighbors = [
        np.argsort(np.linalg.norm(weights - weights[i], axis=1))[:n_neighbors].tolist()
        for i in range(pop_size)
    ]
    gen_stats = [{"gen": 0, "avgs": [round(np.mean([ind.scores[j] for ind in pop]) * 10, 2) for j in range(n_obj)]}]
    archive = {}  # (option_id, summary) -> dict，保存演化过程中出现过的所有唯一样本

    def add_to_archive(ind: Individual):
        s = ind.content.get("summary", "")
        k = (ind.option_id, s)
        if k not in archive:
            d = ind.to_dict()
            d["scores"] = ind.scores[:n_obj]
            archive[k] = d
    for ind in pop:
        add_to_archive(ind)

    # 2. 演化
    for gen in range(1, n_gens + 1):
        log(f"第 {gen}/{n_gens} 代...")
        pareto_idx = find_pareto(pop, n_obj)
        pareto = [pop[i] for i in pareto_idx]

        # 交叉
        pairs = []
        for _ in range(n_crossover):
            if len(pareto) >= 2:
                a, b = random.sample(pareto, 2)
                pairs.append((a, b))
        offspring = crossover_route(pairs, profile, domain, client, cfg) if pairs else []

        # 变异
        n_mut = max(1, int(pop_size * p_mut))
        mut_cand = random.sample(pop, min(n_mut, len(pop)))
        mutated = mutate_batch(mut_cand, profile, domain, client, cfg)

        # 扩展
        expanded = expand_options(gen, [x.option_id for x in pop], profile, domain, cfg, client)

        new_all = offspring + mutated + expanded
        if new_all:
            sc_new = evaluate_batch(new_all, profile, objectives, client, cfg)
            for ind, sc in zip(new_all, sc_new):
                ind.scores = sc[:n_obj]
            for ind in new_all:
                ideal = np.maximum(ideal, np.array(ind.scores[:n_obj]))
                add_to_archive(ind)

        for child in new_all:
            for i in range(pop_size):
                for j in neighbors[i]:
                    wi = weights[j]
                    if tchebycheff(child.scores, wi, ideal) <= tchebycheff(pop[j].scores, wi, ideal):
                        pop[j] = child
                        break

        avgs = [round(np.mean([ind.scores[j] for ind in pop]) * 10, 2) for j in range(n_obj)]
        gen_stats.append({"gen": gen, "avgs": avgs})
        log(f"  契合={avgs[0]:.1f}" + (f"  {obj_names[-1]}={avgs[-1]:.1f}" if n_obj > 1 else ""))

    # 3. Pareto 排序
    pareto_idx = find_pareto(pop, n_obj)
    pareto_list = [pop[i] for i in pareto_idx]
    weighted = [(ind, sum(ind.scores)) for ind in pareto_list]
    weighted.sort(key=lambda x: x[1], reverse=True)
    ranking = [x[0].option_id for x in weighted]

    for ind in pop:
        ind.rank = ranking.index(ind.option_id) + 1 if ind.option_id in ranking else 999
        ind.pareto_front = ind.option_id in ranking

    pareto_ids = set(ranking)
    all_seen = list(archive.values())
    for d in all_seen:
        oid = d.get("option_id", "")
        d["pareto_front"] = oid in pareto_ids
        d["rank"] = ranking.index(oid) + 1 if oid in ranking else 999

    return {
        "profile": profile,
        "domain": domain,
        "objectives": objectives,
        "config": {**cfg, "pop_size": pop_size, "n_gens": n_gens},
        "evaluated_options": [ind.to_dict() for ind in pop],
        "all_options": all_seen,
        "pareto_ranking": ranking,
        "gen_stats": gen_stats,
    }
