#!/usr/bin/env python3
"""
通用 MOEA/D 预测系统 — Web API
输入：目标、现有信息、超参数 → Pareto 最优预测
"""

import json
import os
import threading
from pathlib import Path

# 尝试加载 .env（可选，需 pip install python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from flask import Flask, jsonify, request, send_from_directory

from moead_engine import run

app = Flask(__name__)
BASE = Path(__file__).resolve().parent
RESULTS_PATH = BASE / "predict_results.json"
RUN_STATUS = {"running": False, "log": [], "error": None, "results": None}


def _get_api_config(config: dict):
    """优先：config > OpenRouter(DeepSeek) > DEEPSEEK > OPENAI。OpenRouter 可解决直接调用 DeepSeek 的 governor 认证问题。"""
    # OpenRouter 调用 DeepSeek：解决直接 DeepSeek API 认证失败
    or_key = (config.get("api_key") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if or_key:
        base = config.get("base_url") or os.environ.get("OPENROUTER_API_BASE") or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        if "openrouter" in base.lower():
            model = config.get("model") or os.environ.get("OPENROUTER_MODEL") or "openai/gpt-4o-mini"
            return or_key, base.rstrip("/"), model

    api_key = (config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    base_url = (
        config.get("base_url")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ("https://api.openai.com/v1" if os.environ.get("OPENAI_API_KEY") else "https://api.deepseek.com")
    )
    model = config.get("model") or ("gpt-4o-mini" if "openai" in base_url else "deepseek-chat")
    return api_key, base_url, model


def run_predict(params: dict):
    global RUN_STATUS
    RUN_STATUS["running"] = True
    RUN_STATUS["log"] = []
    RUN_STATUS["error"] = None
    RUN_STATUS["results"] = None
    try:
        config = params.get("config", {})
        api_key, base_url, model = _get_api_config(config)
        config = {**config, "model": model}
        if not api_key:
            RUN_STATUS["error"] = (
                "未配置 API Key。请在 .env 中设置 OPENROUTER_API_KEY、DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
            )
            return

        def progress(msg):
            RUN_STATUS["log"].append(msg)

        res = run(
            profile=params["profile"],
            domain=params["domain"],
            objectives=params["objectives"],
            config=config,
            api_key=api_key,
            base_url=base_url,
            progress_callback=progress,
        )
        RUN_STATUS["results"] = res
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        err_msg = str(e)
        if "Authentication" in type(e).__name__ or "Authentication" in err_msg or "governor" in err_msg.lower():
            RUN_STATUS["error"] = (
                "API 认证失败。请检查：① API Key 是否正确、未过期；② 账号余额/调用限制。"
                "可尝试改用 OPENAI_API_KEY（在 .env 中设置），使用 OpenAI 作为备选。"
            )
        else:
            RUN_STATUS["error"] = err_msg
        RUN_STATUS["log"].append(traceback.format_exc())
    finally:
        RUN_STATUS["running"] = False


def load_results():
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── 路由 ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BASE / "web"), "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global RUN_STATUS
    if RUN_STATUS["running"]:
        return jsonify({"ok": False, "error": "已有任务运行中"}), 409
    data = request.get_json() or {}
    profile = data.get("profile", "").strip()
    domain = data.get("domain", "").strip()
    objectives = data.get("objectives", [])
    if not profile:
        return jsonify({"ok": False, "error": "缺少 profile（现有信息）"}), 400
    if not domain:
        return jsonify({"ok": False, "error": "缺少 domain（预测领域）"}), 400
    if not objectives:
        return jsonify({"ok": False, "error": "缺少 objectives（目标列表）"}), 400
    for i, o in enumerate(objectives):
        if isinstance(o, str):
            objectives[i] = {"name": o, "definition": ""}
        elif not isinstance(o, dict) or "name" not in o:
            objectives[i] = {"name": o.get("name", f"目标{i+1}"), "definition": o.get("definition", o.get("desc", ""))}

    config = data.get("config", {})
    or_key = (config.get("api_key") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    api_key = or_key or (config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return jsonify({
            "ok": False,
            "error": "未配置 API Key。请在 .env 中设置 OPENROUTER_API_KEY、DEEPSEEK_API_KEY 或 OPENAI_API_KEY"
        }), 400
    constraints = data.get("constraints")
    if constraints is not None:
        if isinstance(constraints, str) and constraints.strip():
            config["constraints"] = [c.strip() for c in constraints.split("\n") if c.strip()]
        elif isinstance(constraints, list):
            config["constraints"] = [str(c).strip() for c in constraints if str(c).strip()]
    params = {"profile": profile, "domain": domain, "objectives": objectives, "config": config}
    threading.Thread(target=run_predict, args=(params,), daemon=True).start()
    return jsonify({"ok": True, "message": "已启动预测"})


@app.route("/api/run/status")
def api_run_status():
    return jsonify({
        "ok": True,
        "running": RUN_STATUS["running"],
        "log": RUN_STATUS["log"][-80:],
        "error": RUN_STATUS["error"],
        "done": RUN_STATUS["results"] is not None,
    })


@app.route("/api/results")
def api_results():
    data = load_results()
    if data is None and RUN_STATUS["results"]:
        data = RUN_STATUS["results"]
    if data is None:
        return jsonify({"ok": False, "error": "无结果，请先运行预测"}), 404
    return jsonify({"ok": True, "data": data})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "接口不存在"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"ok": False, "error": "服务器错误，请查看终端日志"}), 500

if __name__ == "__main__":
    os.chdir(BASE)
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
