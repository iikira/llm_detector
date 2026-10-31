# llm_detector

探测一个 OpenAI 兼容 / Anthropic 兼容的 LLM 网关，确认它对 OpenAI 与 Anthropic 两种 API 格式的支持情况，并对每个模型测量时延与吞吐。

## 功能

- 通过 `GET /v1/models` 获取网关可用的模型列表
- 对每个模型，分别用 OpenAI 格式（`/v1/chat/completions`）和 Anthropic 格式（`/v1/messages`）发送 `hello` 测试
- 流式请求，统计每模型、每格式的：
  - **TTFT / 首 token 时延**（首 token 前耗时）
  - **整体时延**（请求总耗时）
  - **token 速度**（完成 token 数 ÷ 生成时长，tok/s）
  - **完成 token 数**（来自响应 usage）
- **多模态探测**（默认开启）：向图片 / 音频文件发送探测，验证网关对两种格式多模态内容块的支持
- **长文本探测**（默认开启）：发送约 10k / 100k token 的输入，验证网关处理大输入的能力
- 打印 OpenAI 与 Anthropic 各自的完整回复
- **每次运行自动生成自包含的 `report.html`**（内联样式，浏览器直接打开）
- 支持并发、超时、模型过滤等参数

## 环境要求

- Python 3.8+
- `requests` 库：`pip install requests`

## 使用

先设置 API Key。源码不硬编码密钥，通过环境变量或参数提供：

```bash
export LLM_PROBE_API_KEY=sk-xxxx
python probe_llm_api.py
```

或直接传参：

```bash
python probe_llm_api.py --base-url http://localhost:3000 --api-key sk-xxxx
```

## 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-u, --base-url` | 网关地址 | `http://localhost:3000` |
| `-k, --api-key` | API Key（也可用环境变量 `LLM_PROBE_API_KEY`） | 空 |
| `--only` | 只测试格式：`openai` / `anthropic` / `both` | `both` |
| `-p, --pattern` | 只测试名称匹配该正则的模型，如 `deepseek\|glm` | 全部 |
| `-t, --timeout` | 单次请求超时（秒） | `60` |
| `-c, --concurrency` | 并发请求数 | `4` |
| `--max-tokens` | 每次测试允许生成的 token 上限 | `256` |
| `--multimodal` / `--no-multimodal` | 是否启用多模态（图片/音频）探测 | 开启 |
| `--image-path` | 图片文件路径（默认 `omni_test_data/222.jpg`，可覆盖） | `omni_test_data/222.jpg` |
| `--audio-path` | 音频文件路径（默认 `omni_test_data/assistant.mp3`，可覆盖） | `omni_test_data/assistant.mp3` |
| `--long` / `--no-long` | 是否启用长文本探测 | 开启 |
| `--long-max-tokens` | 长文本探测允许生成的 token 上限 | `1024` |
| `--long-timeout` | 长文本探测单次请求超时（秒） | `600` |
| `--html-out` | HTML 报告输出路径 | `report.html` |
| `-j, --json` | 以 JSON 输出原始结果 | 关闭 |
| `-v, --verbose` | 输出完整响应体 | 关闭 |

> 多模态开关默认开启，默认读取 `omni_test_data/` 目录下的 `222.jpg`（图片）与 `assistant.mp3`（音频）；文件缺失时该模态被跳过（不会报错）。图片探测会附提示词「这张图片有什么内容」。
>
> 长文本探测在开头加一段随机 `nonce`（防网关缓存/去重，确保每次真实走模型），后接详细分析提示词（要求输出摘要/结构/主题/统计/结论五部分）与占位文本，用较大的 `max_tokens`（默认 1024）与超时（默认 600 秒）测量长输入下的吞吐与输出上限。

## 示例

```bash
# 只测 OpenAI 格式
python probe_llm_api.py --only openai

# 只测匹配 deepseek 或 glm 的模型，并发 8，超时 5 分钟
python probe_llm_api.py --pattern "deepseek|glm" --concurrency 8 --timeout 300

# 开启多模态（图片 + 音频）探测
python probe_llm_api.py --image-path "C:/Pictures/shot.png" --audio-path "C:/Downloads/voice.mp3"

# 只跑文本 hello，关闭多模态与长文本
python probe_llm_api.py --no-multimodal --no-long

# 输出 JSON 原始结果（仍会生成 report.html）
python probe_llm_api.py --json -v
```

## 输出说明

- **汇总报告**：分 `[OpenAI 格式]` 与 `[Anthropic 格式]` 两张表，列出每个模型各探测类型（文本 / 图片 / 音频 / 长文本）的 状态 / TTFT / 整体时延 / 速度 / 完成 token 数
- **完整响应**：每个模型各探测的指标与回复全文（终端输出）
- **`report.html`**：每次运行自动生成，交互式报告，可在浏览器中直接查看：
  - 每格式一张表，含文本 / 图片 / 音频 / 长文本全部探测
  - **排序**：点击表头按任意列排序
  - **筛选**：搜索框按模型过滤，状态下拉（OK/失败）、类型下拉（文本/图片/音频/长文本）组合筛选
  - **回复内嵌**：每行可点击「展开」查看该探测的完整回复或错误详情，无需独立的完整响应章节

> 注意：Windows 终端若为 GBK 编码，中文可能显示乱码，可用 `chcp 65001` 切换 UTF-8，或重定向输出到文件后用 UTF-8 查看。`report.html` 始终以 UTF-8 写入，不受终端编码影响。
