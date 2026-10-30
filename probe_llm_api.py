#!/usr/bin/env python3
"""探测一个 OpenAI 兼容 / Anthropic 兼容的 LLM 网关（默认 http://localhost:3000）。

流程：
  1. GET  /v1/models 获取模型列表
  2. 对每个模型，分别用 OpenAI 格式 (/v1/chat/completions) 和
     Anthropic 格式 (/v1/messages) 发送 "hello" 测试
  3. 可选多模态（图片/音频）与长文本（10k/100k token）探测
  4. 全部完成后，终端打印汇总报告，并自动生成 report.html

用法示例：
    export LLM_PROBE_API_KEY=sk-xxxx        # 先设置密钥（否则运行会报错）
    python probe_llm_api.py
    python probe_llm_api.py --base-url http://localhost:3000 --api-key sk-xxx
    python probe_llm_api.py --only openai
    python probe_llm_api.py --pattern "deepseek|glm"
    python probe_llm_api.py --image-path img.png --audio-path audio.mp3
    python probe_llm_api.py --no-multimodal --no-long
    python probe_llm_api.py --json --verbose

多模态默认读取 omni_test_data/ 下的 222.jpg 与 assistant.mp3，可用
--image-path / --audio-path 覆盖；图片探测附提示词「这张图片有什么内容」。
API Key 可通过环境变量 LLM_PROBE_API_KEY 或 --api-key 参数提供，源码不硬编码密钥。
"""

import argparse
import base64
import html
import json
import mimetypes
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# 控制台可能不是 UTF-8（如 Windows GBK），用 sys.stdout 自身的编码安全打印，
# 避免中文 / emoji 触发 UnicodeEncodeError。
_ENC = getattr(sys.stdout, "encoding", None) or "utf-8"


def p(s=""):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode(_ENC, errors="replace").decode(_ENC, errors="replace"))

DEFAULT_BASE_URL = "http://localhost:3000"

# API Key 不从源码取，改从环境变量读取，避免把密钥写进公开仓库。
DEFAULT_API_KEY = os.environ.get("LLM_PROBE_API_KEY", "")

ANTHROPIC_VERSION = "2023-06-01"
TEST_MESSAGE = "hello"
IMAGE_PROMPT = "这张图片有什么内容"
LONG_TOKENS_DEFAULT = "10000,100000"
LONG_MAX_TOKENS_DEFAULT = 1024

# 长文本探测的提示词：要求模型对长文本做多部分详细分析，引导其尽量多输出，
# 便于测量长输入下的吞吐与输出 token 上限。
LONG_PROMPT = (
    "请仔细阅读下面的长文本，并给出尽可能详细的深度分析。请尽量多写，"
    "输出务必覆盖以下五个部分，每部分都展开说明：\n"
    "一、全文摘要（至少 3 段）\n"
    "二、主要内容与关键信息提取\n"
    "三、文本结构与行文特点分析\n"
    "四、词汇、主题与句式统计\n"
    "五、总结与延伸思考\n"
    "长文本内容如下：\n"
)

# 多模态默认测试数据目录（相对脚本位置，避免硬编码绝对路径）。
_OMNI_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omni_test_data")
DEFAULT_IMAGE_PATH = os.path.join(_OMNI_DATA_DIR, "222.jpg")
DEFAULT_AUDIO_PATH = os.path.join(_OMNI_DATA_DIR, "assistant.mp3")

# 常见扩展名 -> MIME，作为 mimetypes 的补充/兜底（确保媒体格式判定稳定）。
_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
}

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def color(s, c, enabled):
    return f"{c}{s}{RESET}" if enabled else s


def brief(obj, limit=80):
    if obj is None:
        return ""
    if isinstance(obj, str):
        text = obj
    else:
        try:
            text = json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(obj)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


# ---------- 媒体 / 长文本 / 内容块 辅助 ----------

def _guess_mime(path):
    """由扩展名推断 MIME，优先走显式表，兜底 mimetypes。"""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def read_media(path):
    """读取媒体文件并 base64 编码。返回 (mime, b64)；文件不可读时返回 (None, None)。"""
    if not path:
        return None, None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        p(f"! 无法读取媒体文件 '{path}': {exc}，该模态被跳过")
        return None, None
    return _guess_mime(path), base64.b64encode(data).decode("ascii")


def make_long_text(n_tokens):
    """按 'token ≈ 4 字符' 启发式生成约 n_tokens 的占位文本。
    用语义完整的重复短语避免网关对单字符重复做去重。"""
    phrase = "The quick brown fox jumps over the lazy dog. "
    per_token = 4
    repeats = max(1, int(n_tokens * per_token / len(phrase)))
    return phrase * repeats


def make_long_content(n_tokens):
    """长文本探测的完整 user content：详细分析提示词 + 占位长文本。"""
    return LONG_PROMPT + make_long_text(n_tokens)


def _ext_to_openai_format(path):
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return ext or "mp3"


def _content_blocks(fmt, modality, media, path):
    """把「图片/音频」渲染成某格式的 content 块列表。media=(mime, b64)。
    图片前附一句提示词；音频不带提示词。"""
    mime, b64 = media
    if modality == "image":
        if fmt == "openai":
            blocks = [{"type": "text", "text": IMAGE_PROMPT},
                      {"type": "image_url",
                       "image_url": {"url": f"data:{mime};base64,{b64}"}}]
        else:
            blocks = [{"type": "text", "text": IMAGE_PROMPT},
                      {"type": "image",
                       "source": {"type": "base64", "media_type": mime, "data": b64}}]
        return blocks
    if modality == "audio":
        if fmt == "openai":
            return [{"type": "input_audio",
                     "input_audio": {"data": b64, "format": _ext_to_openai_format(path)}}]
        return [{"type": "audio",
                 "source": {"type": "base64", "media_type": mime, "data": b64}}]
    raise ValueError(f"unknown modality: {modality}")


# ---------- 结果分组（终端与 HTML 共用） ----------

def _label_kind(probe):
    """把 probe 标签解析为 (format, kind, variant)。
    kind ∈ {text, image, audio, long}; variant 为长文本 token 数等说明。"""
    fmt = "openai" if probe.startswith("openai") else "anthropic"
    base = probe[len(fmt):].strip()
    if "long" in base:
        match = re.search(r"(\d+)", base)
        return fmt, "long", (match.group(1) if match else "")
    if "image" in base:
        return fmt, "image", "image"
    if "audio" in base:
        return fmt, "audio", "audio"
    return fmt, "text", ""


def _group_rows(results, fmt):
    """按 kind 分组，返回 {kind: [rows]}。终端与 HTML 复用同一份数据。"""
    rows = {"text": [], "image": [], "audio": [], "long": []}
    for r in results:
        if r.get("probe") and r["probe"].startswith(fmt):
            f, kind, variant = _label_kind(r["probe"])
            if f == fmt and kind in rows:
                r["_variant"] = variant
                rows[kind].append(r)
    return rows


class Prober:
    def __init__(self, base_url, api_key, timeout, verbose):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verbose = verbose
        self.results = []

    def request(self, method, path, headers, payload=None, stream=False):
        """发送请求。stream=True 时走 SSE 流式并统计首 token 时延(TTFT)。"""
        url = self.base_url + path
        started = time.time()
        try:
            resp = requests.request(
                method, url, headers=headers, json=payload, timeout=self.timeout,
                stream=stream,
            )
        except requests.RequestException as exc:
            return {
                "ok": False,
                "status": None,
                "elapsed": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "url": url,
            }

        status = resp.status_code
        if stream:
            return self._read_stream(resp, started)

        body = None
        try:
            body = resp.json()
        except ValueError:
            body = resp.text[:1000]
        return {
            "ok": 200 <= status < 300,
            "status": status,
            "elapsed": time.time() - started,
            "body": body,
            "url": url,
        }

    @staticmethod
    def _read_stream(resp, started):
        """读取 SSE 流，返回首 token 时延、整体时延、拼出的文本与原始事件行。"""
        first_token_at = None
        last_event_at = started
        raw_lines = []

        try:
            # SSE 常为 text/event-stream 且无 charset；requests 会把 resp.encoding
            # 默认成 ISO-8859-1，把 UTF-8 中文解成乱码（模型回复多为中文）。
            # LLM 的 SSE 一律是 UTF-8，这里无条件强制。
            resp.encoding = "utf-8"
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                raw_lines.append(line)
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if first_token_at is None and data and data != "[DONE]":
                    # 以“出现首个含内容的数据块”作为首 token 时刻
                    first_token_at = time.time()
                last_event_at = time.time()
        except requests.RequestException as exc:
            raw_lines.append(f"<stream error: {exc}>")

        done = time.time()
        elapsed = done - started
        return {
            "ok": 200 <= resp.status_code < 300,
            "status": resp.status_code,
            "elapsed": elapsed,
            "ttft": (first_token_at - started) if first_token_at is not None else None,
            "stream_events": raw_lines,
            "stream_text": "",
            "url": resp.url,
        }

    # ---------- OpenAI 格式 ----------
    def openai_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_models(self):
        res = self.request("GET", "/v1/models", self.openai_headers())
        models = []
        if res["ok"] and isinstance(res.get("body"), dict):
            data = res["body"].get("data")
            if isinstance(data, list):
                models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
        self.results.append({"probe": "GET /v1/models", **res, "models": models})
        return res, models

    def openai_chat(self, model, max_tokens=256, content=TEST_MESSAGE,
                    label="openai /v1/chat/completions", extra=None):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
        res = self.request("POST", "/v1/chat/completions", self.openai_headers(),
                           payload, stream=True)
        text, completion_tokens = self._parse_openai_stream(res)
        res["stream_text"] = text
        res["completion_tokens"] = completion_tokens
        self.results.append(
            {
                "probe": label,
                "model": model,
                **res,
                "reply": text,
                **(extra or {}),
            }
        )
        return res

    def openai_multimodal(self, model, modality, media, path, max_tokens=256):
        blocks = _content_blocks("openai", modality, media, path)
        return self.openai_chat(
            model, max_tokens,
            content=blocks,
            label=f"openai multimodal {modality}",
            extra={"modality": modality, "media_file": path},
        )

    def openai_long(self, model, n_tokens, max_tokens=LONG_MAX_TOKENS_DEFAULT):
        content = make_long_content(n_tokens)
        res = self.openai_chat(
            model, max_tokens,
            content=content,
            label=f"openai long {n_tokens}",
            extra={"input_tokens": n_tokens, "input_chars": len(content)},
        )
        return res

    @staticmethod
    def _parse_openai_stream(res):
        """从 OpenAI SSE 流解析文本与 completion_tokens。返回 (text, tokens)。
        部分推理模型把文本放在 delta.reasoning 而非 delta.content，两者都收集。"""
        text, tokens = "", None
        for line in res.get("stream_events") or []:
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except (ValueError, TypeError):
                continue
            choices = chunk.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                for key in ("content", "reasoning"):
                    if isinstance(delta.get(key), str):
                        text += delta[key]
            if chunk.get("usage") and isinstance(chunk["usage"], dict):
                tokens = chunk["usage"].get("completion_tokens")
        return text, tokens

    # ---------- Anthropic 格式 ----------
    def anthropic_headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def anthropic_messages(self, model, max_tokens=256, content=TEST_MESSAGE,
                           label="anthropic /v1/messages", extra=None):
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }
        res = self.request("POST", "/v1/messages", self.anthropic_headers(),
                           payload, stream=True)
        text, completion_tokens = self._parse_anthropic_stream(res)
        res["stream_text"] = text
        res["completion_tokens"] = completion_tokens
        self.results.append(
            {
                "probe": label,
                "model": model,
                **res,
                "reply": text,
                **(extra or {}),
            }
        )
        return res

    def anthropic_multimodal(self, model, modality, media, path, max_tokens=256):
        blocks = _content_blocks("anthropic", modality, media, path)
        return self.anthropic_messages(
            model, max_tokens,
            content=blocks,
            label=f"anthropic multimodal {modality}",
            extra={"modality": modality, "media_file": path},
        )

    def anthropic_long(self, model, n_tokens, max_tokens=LONG_MAX_TOKENS_DEFAULT):
        content = make_long_content(n_tokens)
        res = self.anthropic_messages(
            model, max_tokens,
            content=content,
            label=f"anthropic long {n_tokens}",
            extra={"input_tokens": n_tokens, "input_chars": len(content)},
        )
        return res

    @staticmethod
    def _parse_anthropic_stream(res):
        """从 Anthropic SSE 流解析文本与 output_tokens。返回 (text, tokens)。
        推理模型在 content_block_delta 里以 delta.thinking 返回思考内容，一并收集。"""
        text, tokens = "", None
        for line in res.get("stream_events") or []:
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            try:
                ev = json.loads(data)
            except (ValueError, TypeError):
                continue
            etype = ev.get("type")
            if etype == "content_block_delta":
                delta = ev.get("delta") or {}
                if isinstance(delta.get("text"), str):
                    text += delta["text"]
                if isinstance(delta.get("thinking"), str):
                    text += delta["thinking"]
            elif etype == "message_delta":
                usage = ev.get("usage") or {}
                tokens = usage.get("output_tokens")
        return text, tokens


def run(base_url, api_key, timeout, verbose, only, pattern, max_tokens, concurrency,
        multimodal=True, image_path=None, audio_path=None, long_=True, long_tokens=None,
        long_max_tokens=LONG_MAX_TOKENS_DEFAULT):
    prober = Prober(base_url, api_key, timeout, verbose)
    p(f"目标: {base_url}")
    p(f"Key : {api_key[:8]}…{api_key[-4:]}")
    p("-" * 72)

    _, models = prober.get_models()
    if not models:
        p("! 无法从 /v1/models 获取模型列表")
        return prober.results

    if pattern:
        rx = re.compile(pattern)
        models = [m for m in models if rx.search(m)]
        p(f"按 --pattern '{pattern}' 过滤: {len(models)} 个模型")

    # 媒体一次性读取缓存，避免每个模型重复读 ~1.9MB base64。
    image_media = read_media(image_path) if (multimodal and image_path) else (None, None)
    audio_media = read_media(audio_path) if (multimodal and audio_path) else (None, None)
    if multimodal:
        if image_path and image_media[0] is None:
            p("! 图片无法读取，图片探测被跳过")
        if audio_path and audio_media[0] is None:
            p("! 音频无法读取，音频探测被跳过")

    if long_tokens is None:
        long_tokens = [int(t) for t in LONG_TOKENS_DEFAULT.split(",") if t.strip().isdigit()]

    p(f"共 {len(models)} 个模型，并发 {concurrency}，逐个发送 '{TEST_MESSAGE}' 测试…\n")

    def test_model(model):
        if only in ("openai", "both"):
            prober.openai_chat(model, max_tokens)
        if only in ("anthropic", "both"):
            prober.anthropic_messages(model, max_tokens)

        if multimodal:
            if image_path and image_media[0]:
                if only in ("openai", "both"):
                    prober.openai_multimodal(model, "image", image_media, image_path, max_tokens)
                if only in ("anthropic", "both"):
                    prober.anthropic_multimodal(model, "image", image_media, image_path, max_tokens)
            if audio_path and audio_media[0]:
                if only in ("openai", "both"):
                    prober.openai_multimodal(model, "audio", audio_media, audio_path, max_tokens)
                if only in ("anthropic", "both"):
                    prober.anthropic_multimodal(model, "audio", audio_media, audio_path, max_tokens)

        if long_:
            for n in long_tokens:
                if only in ("openai", "both"):
                    prober.openai_long(model, n, long_max_tokens)
                if only in ("anthropic", "both"):
                    prober.anthropic_long(model, n, long_max_tokens)
        return model

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(test_model, m): m for m in models}
        for fut in as_completed(futs):
            done += 1
            model = futs[fut]
            try:
                fut.result()
            except Exception as exc:
                p(f"[{done}/{len(models)}] {model} 异常: {type(exc).__name__}: {exc}")
            else:
                p(f"[{done}/{len(models)}] {model} 完成")

    return prober.results


def _fmt_sec(s, color_enabled):
    if s is None:
        return "-"
    return f"{s*1000:7.0f}ms"


def _fmt_tps(tokens, gen_time):
    if not tokens or not gen_time:
        return "-"
    return f"{tokens/gen_time:6.1f}"


def _print_rows(rows, use_color):
    """打印一组探测行的汇总表。rows 为已按某格式分组的 (kind, list) 序列。"""
    for kind, rows_k in rows:
        if not rows_k:
            continue
        for r in rows_k:
            model = r["model"]
            status = (color("OK", GREEN, use_color) if r["ok"]
                      else color(str(r.get("status") or "ERR"), RED, use_color))
            gen_time = None
            if r.get("ttft") is not None and r.get("elapsed") is not None:
                gen_time = r["elapsed"] - r["ttft"]
            tps = _fmt_tps(r.get("completion_tokens"), gen_time)
            variant = r.get("_variant") or ""
            p(f"{model:<30} {variant:<10} {status:<5} {_fmt_sec(r.get('ttft'), use_color):>8} "
              f"{_fmt_sec(r.get('elapsed'), use_color):>9} {tps:>7}  {r.get('completion_tokens') or '-'}")
    p("-" * 72)


def report(results, only, use_color):
    models = sorted({r["model"] for r in results if r.get("model")})

    p("\n" + "=" * 72)
    p("汇总报告  (TTFT=首 token 时延 | 整体=总时延 | 速度=tokens/s)")
    p("=" * 72)

    oai_groups = [(kind, _group_rows(results, "openai")[kind])
                  for kind in ("text", "image", "audio", "long")]
    ant_groups = [(kind, _group_rows(results, "anthropic")[kind])
                  for kind in ("text", "image", "audio", "long")]

    if only in ("openai", "both"):
        p("\n[OpenAI 格式] /v1/chat/completions")
        p(f"{'模型':<30} {'类型':<10} {'状态':<5} {'TTFT':>8} {'整体':>9} {'速度':>7}  token")
        p("-" * 72)
        _print_rows(oai_groups, use_color)
        text_ok = sum(1 for r in _group_rows(results, "openai")["text"] if r["ok"])
        p(f"OpenAI 格式: {text_ok}/{len(models)} 通过")

    if only in ("anthropic", "both"):
        p("\n[Anthropic 格式] /v1/messages")
        p(f"{'模型':<30} {'类型':<10} {'状态':<5} {'TTFT':>8} {'整体':>9} {'速度':>7}  token")
        p("-" * 72)
        _print_rows(ant_groups, use_color)
        text_ok = sum(1 for r in _group_rows(results, "anthropic")["text"] if r["ok"])
        p(f"Anthropic 格式: {text_ok}/{len(models)} 通过")

    # ---- 完整响应 ----
    p("\n" + "=" * 72)
    p("完整响应")
    p("=" * 72)
    for r in results:
        if not r.get("probe") or not r.get("model"):
            continue
        if r["probe"] in ("GET /v1/models",):
            continue
        model = r["model"]
        fmt = "OpenAI" if r["probe"].startswith("openai") else "Anthropic"
        label = r["probe"]
        p(f"\n--- {fmt} / {model} [{label}] ---")
        if not r["ok"]:
            if r.get("status") is None:
                p(f"  错误: {r.get('error') or _stream_error(r)}")
            else:
                detail = r.get("body")
                if detail is None:
                    detail = _stream_error(r) or ""
                p(f"  HTTP {r['status']}: {brief(detail)}")
            continue
        reply = r.get("reply") or ""
        p(f"  TTFB/首token: {_fmt_sec(r.get('ttft'), use_color)}  整体: {_fmt_sec(r.get('elapsed'), use_color)}"
          f"  完成 {r.get('completion_tokens')} token, "
          f"速度: {_fmt_tps(r.get('completion_tokens'), (r.get('elapsed')-r.get('ttft')) if r.get('ttft') is not None and r.get('elapsed') is not None else None)} tok/s")
        p(f"  回复: {brief(reply, 500)}")


_HTML_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, "PingFang SC",
       "Microsoft YaHei", sans-serif; margin: 0; padding: 24px;
       color: #1f2933; background: #f6f8fa; }
.container { max-width: 1280px; margin: 0 auto; }
h1 { font-size: 1.5em; margin: 0 0 4px; }
h2 { font-size: 1.2em; margin: 32px 0 12px; padding-bottom: 8px;
     border-bottom: 1px solid #d0d7de; }
.meta { color: #57606a; font-size: .9em; margin: 0 0 8px; }

/* ---- 工具条 ---- */
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
           margin: 12px 0 10px; }
.toolbar input[type=search] { flex: 1 1 200px; padding: 6px 10px;
           border: 1px solid #c9d1d9; border-radius: 6px; background: #fff; }
.toolbar select { padding: 6px 8px; border: 1px solid #c9d1d9; border-radius: 6px;
           background: #fff; }
.toolbar label { font-size: .85em; color: #57606a; }
.count { font-size: .85em; color: #57606a; margin-left: auto; }

/* ---- 表格 ---- */
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: #fff;
        border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden;
        font-size: .9em; }
thead th { position: sticky; top: 0; background: #f0f3f6; font-weight: 600;
           text-align: left; padding: 8px 10px; border-bottom: 2px solid #d0d7de;
           cursor: pointer; user-select: none; white-space: nowrap; }
thead th.sorted-asc::after { content: " ▲"; font-size: .75em; }
thead th.sorted-desc::after { content: " ▼"; font-size: .75em; }
thead th.num, td.num { text-align: right; }
td { padding: 7px 10px; border-bottom: 1px solid #e4e7eb;
     font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr:hover td { background: #f5f9fd; }
tr.hidden { display: none; }

/* ---- 状态 ---- */
.ok { color: #1a7f37; font-weight: 600; }
.err { color: #cf222e; font-weight: 600; }
.warn { color: #9a6700; font-weight: 600; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 10px;
         font-size: .82em; }
.badge-ok { background: #dafbe1; color: #1a7f37; }
.badge-err { background: #ffebe9; color: #cf222e; }

/* ---- 展开行 ---- */
.expand { color: #0969da; cursor: pointer; font-size: .85em; white-space: nowrap; }
.expand:hover { text-decoration: underline; }
tr.detail { display: none; }
tr.detail.open { display: table-row; }
tr.detail td { background: #fafbfc; padding: 12px; white-space: normal; }
pre { background: #0d1117; color: #c9d1d9; padding: 12px; border-radius: 6px;
      overflow-x: auto; font-size: .82em; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word; margin: 0; }
pre .k { color: #7ee787; } pre .e { color: #ff7b72; }
.summary-line { margin: 0 0 6px; font-size: .85em; color: #57606a; }

details { margin: 24px 0; }
summary { cursor: pointer; color: #0969da; }
"""


# 内联 JS：排序、筛选、行展开。作为常量嵌入页面。
_HTML_JS = r"""
(function () {
  function buildData(){
    document.querySelectorAll('.probe-table').forEach(function(tbl){
      var rows = tbl.querySelectorAll('tbody > tr.row');
      rows.forEach(function(tr){
        tr._model = tr.getAttribute('data-model');
        tr._type  = tr.getAttribute('data-type');
        tr._kind  = tr.getAttribute('data-kind');
        tr._tsort = parseFloat(tr.getAttribute('data-tsort'))||null;
        tr._status= tr.getAttribute('data-status');
        tr._ttft  = parseFloat(tr.getAttribute('data-ttft'))||null;
        tr._elapsed=parseFloat(tr.getAttribute('data-elapsed'))||null;
        tr._tps   = parseFloat(tr.getAttribute('data-tps'))||null;
        tr._tok   = parseInt(tr.getAttribute('data-tok'))||null;
      });
    });
  }

  function nextDetail(tr){
    return tr.nextElementSibling && tr.nextElementSibling.classList.contains('detail')
      ? tr.nextElementSibling : null;
  }

  function sortTable(tbl, colIdx, dir){
    var tbody = tbl.querySelector('tbody');
    var rows  = Array.prototype.slice.call(tbody.querySelectorAll('tr.row'));
    var meta  = [['model','_model'],['type','_tsort'],['status','_status'],
                 ['ttft','_ttft'],['elapsed','_elapsed'],['tps','_tps'],['tok','_tok']][colIdx];
    var key = meta[1];
    rows.sort(function(a,b){
      var av=a[key], bv=b[key];
      if (av==null && bv==null) return 0;
      if (av==null) return 1;
      if (bv==null) return -1;
      var c;
      if (typeof av === 'number'){ c = av-bv; }
      else { c = String(av).localeCompare(String(bv)); }
      return dir * c;
    });
    // 行与紧随其后的展开行成对移动，保持展开关联
    rows.forEach(function(r){
      var d = nextDetail(r);
      tbody.appendChild(r);
      if (d) tbody.appendChild(d);
    });
  }

  document.querySelectorAll('.probe-table').forEach(function(tbl){
    var ths = tbl.querySelectorAll('thead th');
    ths.forEach(function(th, i){
      if (th.classList.contains('no-sort')) return;
      th.addEventListener('click', function(){
        var cur = th.classList.contains('sorted-asc') ? 1
                : th.classList.contains('sorted-desc') ? -1 : 0;
        ths.forEach(function(t){ t.classList.remove('sorted-asc','sorted-desc'); });
        var dir = cur === 1 ? -1 : 1;
        th.classList.add(dir === 1 ? 'sorted-asc' : 'sorted-desc');
        sortTable(tbl, i, dir);
      });
    });

    // 筛选
    var wrap = tbl.closest('.table-block');
    if (!wrap) return;
    var q   = wrap.querySelector('.q');
    var fst = wrap.querySelector('.f-status');
    var fty = wrap.querySelector('.f-type');
    var cnt = wrap.querySelector('.count');
    function apply(){
      var n = 0;
      tbl.querySelectorAll('tbody > tr.row').forEach(function(tr){
        var okM = !q || q.value==='' ||
                  (tr._model||'').toLowerCase().indexOf(q.value.toLowerCase())>=0;
        var okS = !fst || fst.value==='' || tr._status===fst.value;
        var okT = !fty || fty.value==='' || tr._kind===fty.value;
        var show = okM && okS && okT;
        tr.classList.toggle('hidden', !show);
        // 隐藏行时也收起其详情
        if (!show && tr.nextElementSibling && tr.nextElementSibling.classList.contains('detail')){
          tr.nextElementSibling.classList.remove('open');
          tr.nextElementSibling.classList.add('hidden');
        }
        if (show) n++;
      });
      if (cnt) cnt.textContent = n + ' 行';
    }
    if (q) q.addEventListener('input', apply);
    if (fst) fst.addEventListener('change', apply);
    if (fty) fty.addEventListener('change', apply);
  });

  // 行展开
  document.querySelectorAll('.probe-table').forEach(function(tbl){
    tbl.querySelectorAll('tbody > tr.row').forEach(function(tr){
      var toggle = tr.querySelector('.expand');
      if (!toggle) return;
      toggle.addEventListener('click', function(){
        var det = tr.nextElementSibling;
        if (!det || !det.classList.contains('detail')) return;
        var open = det.classList.contains('open');
        det.classList.toggle('open', !open);
        det.classList.toggle('hidden', open);
        toggle.textContent = open ? '展开' : '收起';
      });
    });
  });

  // 脚本位于 <body> 末尾，DOM 已就绪，立即构建行数据
  buildData();
})();
"""


def _cell(value, suffix="ms"):
    """HTML 单元格：None 显示 '-'; 数字带后缀。"""
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.0f}{suffix}"
    return html.escape(str(value))


def _html_status_badge(r):
    cls = "ok" if r["ok"] else "err"
    label = "OK" if r["ok"] else str(r.get("status") or "ERR")
    return f'<span class="badge badge-{cls}">{html.escape(label)}</span>'


def _stream_error(r):
    """从 SSE stream_events 里提取网关错误消息（body 为空时兜底用）。
    网关对 4xx/5xx 常以流式 error 块返回，如 {"error":{"message":...}}；
    有的行带 data: 前缀，有的不带，统一处理后解析。"""
    for line in r.get("stream_events") or []:
        s = line.strip()
        if s.startswith("data:"):
            s = s[len("data:"):].strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            continue
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict) and err.get("message"):
            return err["message"]
    return None


def _row_detail(r):
    """单行探测的完整回复/错误详情（嵌入表格的展开行）。"""
    if not r["ok"]:
        if r.get("status") is None:
            msg = r.get("error") or _stream_error(r)
            head = f'<span class="summary-line"><span class="e">错误</span>: {html.escape(str(msg))}</span>'
            payload = html.escape(str(msg))
        else:
            body = r.get("body")
            if body is None:
                msg = _stream_error(r) or ""
                head = (f'<span class="summary-line"><span class="k">HTTP</span> '
                        f'{html.escape(str(r["status"]))}: {html.escape(str(msg))}</span>')
                payload = html.escape(str(msg))
            else:
                head = (f'<span class="summary-line"><span class="k">HTTP</span> '
                        f'{html.escape(str(r["status"]))}: '
                        f'{html.escape(brief(body))}</span>')
                try:
                    payload = html.escape(json.dumps(body, ensure_ascii=False))
                except (TypeError, ValueError):
                    payload = html.escape(str(body))
    else:
        reply = r.get("reply") or ""
        head = f'<span class="summary-line"><span class="k">回复</span>: {html.escape(brief(reply, 120))}</span>'
        payload = html.escape(reply)
    return f"<pre>{payload}</pre>"


def _html_table(results, fmt):
    """渲染某一格式的整张表：文本/图片/音频/长文本合并，支持排序/筛选/展开。"""
    groups = _group_rows(results, fmt)
    order = ("text", "image", "audio", "long")
    flat = [(kind, r) for kind in order for r in groups[kind]]
    if not flat:
        return "<p><em>（无该格式探测）</em></p>"

    head = ("<thead><tr>"
            "<th data-k='model'>模型</th>"
            "<th data-k='type'>类型</th>"
            "<th data-k='status'>状态</th>"
            "<th class='num' data-k='ttft'>TTFT</th>"
            "<th class='num' data-k='elapsed'>整体</th>"
            "<th class='num' data-k='tps'>速度</th>"
            "<th class='num' data-k='tok'>token</th>"
            "<th class='no-sort'></th></tr></thead>")

    body = []
    for kind, r in flat:
        gen_time = None
        if r.get("ttft") is not None and r.get("elapsed") is not None:
            gen_time = r["elapsed"] - r["ttft"]
        tps = f"{r.get('completion_tokens')/gen_time:.1f}" if gen_time and r.get("completion_tokens") else ""
        type_label = {"text": "文本", "image": "图片", "audio": "音频", "long": "长文本"}.get(kind, kind)
        variant = r.get("_variant") or ""
        type_cell = type_label if not variant else f"{type_label} {variant}"
        # 排序用数值秩：文本=0，图片=1，音频=2，长文本按其 token 数，便于按类型排序时分组并区分大小。
        tsort = {"text": 0, "image": 1, "audio": 2}.get(kind, 3)
        if kind == "long":
            try:
                tsort = 3 + float(variant)
            except (TypeError, ValueError):
                tsort = 3
        body.append(
            f"<tr class='row' data-model='{html.escape(str(r.get('model')))}' "
            f"data-type='{html.escape(type_label)}' data-kind='{html.escape(type_label)}' "
            f"data-tsort='{tsort}' data-status='{ 'ok' if r['ok'] else 'err' }' "
            f"data-ttft='{r.get('ttft') or ''}' data-elapsed='{r.get('elapsed') or ''}' "
            f"data-tps='{tps or ''}' data-tok='{r.get('completion_tokens') or ''}'>"
            f"<td>{html.escape(str(r.get('model')))}</td>"
            f"<td>{html.escape(type_cell)}</td>"
            f"<td>{_html_status_badge(r)}</td>"
            f"<td class='num'>{_cell(r.get('ttft'))}</td>"
            f"<td class='num'>{_cell(r.get('elapsed'))}</td>"
            f"<td class='num'>{tps or '-'}</td>"
            f"<td class='num'>{_cell(r.get('completion_tokens'), '')}</td>"
            f"<td><span class='expand'>展开</span></td></tr>"
            f"<tr class='detail hidden'><td colspan='8'>{_row_detail(r)}</td></tr>")

    toolbar = (
        f"<div class='toolbar'>"
        f"<input class='q' type='search' placeholder='搜索模型…' aria-label='搜索模型'>"
        f"<label>状态 <select class='f-status'>"
        f"<option value=''>全部</option><option value='ok'>OK</option>"
        f"<option value='err'>失败</option></select></label>"
        f"<label>类型 <select class='f-type'>"
        f"<option value=''>全部</option><option value='文本'>文本</option>"
        f"<option value='图片'>图片</option><option value='音频'>音频</option>"
        f"<option value='长文本'>长文本</option></select></label>"
        f"<span class='count'></span></div>")

    return (f"<div class='table-block'>{toolbar}"
            f"<div class='table-wrap'><table class='probe-table'>{head}"
            f"<tbody>{''.join(body)}</tbody></table></div></div>")


def write_html_report(results, out_path):
    """生成自包含 report.html（内联 CSS/JS），每次运行调用。"""
    models = sorted({r["model"] for r in results if r.get("model")})
    probes = sorted({r.get("probe") for r in results if r.get("probe")})

    parts = []
    parts.append("<h1>LLM 网关探测报告</h1>")
    parts.append(f"<p class='meta'>模型 {len(models)} 个 · 探测 {len(probes)} 项 · "
                 f"生成于 {time.strftime('%Y-%m-%d %H:%M:%S')} · "
                 f"点击表头排序，搜索/下拉筛选，点击“展开”查看回复</p>")

    for fmt in ("openai", "anthropic"):
        title = "OpenAI 格式 /v1/chat/completions" if fmt == "openai" else "Anthropic 格式 /v1/messages"
        parts.append(f"<h2>{title}</h2>")
        parts.append(_html_table(results, fmt))

    # ---- 原始 JSON ----
    raw_json = html.escape(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    parts.append(f"<h2>原始数据</h2><details><summary>查看 JSON</summary>"
                 f"<pre>{raw_json}</pre></details>")

    doc = ("<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
           f"<title>LLM 网关探测报告</title><style>{_HTML_CSS}</style></head>"
           f"<body><div class='container'>{''.join(parts)}</div>"
           f"<script>{_HTML_JS}</script></body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    p(f"\nHTML 报告已生成: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="探测 LLM 网关对 OpenAI / Anthropic 格式的支持情况（逐个模型发 hello 测试）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-u", "--base-url", default=DEFAULT_BASE_URL, help="网关地址")
    parser.add_argument("-k", "--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--only", choices=["openai", "anthropic", "both"], default="both",
                        help="只测试指定格式")
    parser.add_argument("-p", "--pattern", default=None,
                        help="只测试名称匹配该正则的模型，例如 'deepseek|glm'")
    parser.add_argument("-t", "--timeout", type=float, default=180.0, help="单次请求超时（秒）")
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="并发请求数")
    parser.add_argument("--max-tokens", type=int, default=256, help="每次测试允许生成的 token 上限")
    parser.add_argument("--multimodal", action="store_true", default=True,
                        help="探测多模态（图片/音频），需配合 --image-path / --audio-path")
    parser.add_argument("--no-multimodal", dest="multimodal", action="store_false",
                        help="关闭多模态探测")
    parser.add_argument("--long", action="store_true", default=True,
                        help="测试长文本输入（默认 10k/100k token）")
    parser.add_argument("--no-long", dest="long", action="store_false", help="关闭长文本测试")
    parser.add_argument("--long-tokens", default=LONG_TOKENS_DEFAULT,
                        help="长文本 token 数，逗号分隔，例如 '10000,100000'")
    parser.add_argument("--long-max-tokens", type=int, default=LONG_MAX_TOKENS_DEFAULT,
                        help="长文本探测允许生成的 token 上限（默认 1024，便于测量长输出）")
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH,
                        help="图片文件路径（默认 omni_test_data/222.jpg，可覆盖）")
    parser.add_argument("--audio-path", default=DEFAULT_AUDIO_PATH,
                        help="音频文件路径（默认 omni_test_data/assistant.mp3，可覆盖）")
    parser.add_argument("--html-out", default="report.html", help="HTML 报告输出路径")
    parser.add_argument("-j", "--json", action="store_true", help="以 JSON 输出原始结果")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出完整响应体")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("错误: 未设置 API Key。请通过环境变量 LLM_PROBE_API_KEY 或 --api-key 提供。")

    use_color = sys.stdout.isatty()
    long_tokens = [int(t) for t in args.long_tokens.split(",") if t.strip().isdigit()]
    results = run(args.base_url, args.api_key, args.timeout,
                  args.verbose, args.only, args.pattern,
                  args.max_tokens, args.concurrency,
                  args.multimodal, args.image_path, args.audio_path,
                  args.long, long_tokens, args.long_max_tokens)

    write_html_report(results, args.html_out)

    if args.json:
        p(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return

    report(results, args.only, use_color)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
