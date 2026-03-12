# pachong

用于抓取微博、B站、小红书关于“适老化设备”的舆情，并使用 BERTopic 分类，再用 Ollama 自动命名主题。

## 快速开始

1. 安装依赖：

```bash
pip install pandas requests scikit-learn bertopic sentence-transformers umap-learn hdbscan jieba
```

2. 运行：

```bash
python sentiment_topic_pipeline.py --query 适老化设备 --limit 80 --output-dir output
```

> 微博/B站/小红书常常需要登录态。脚本默认直接读取本地 txt Cookie 文件（可通过 `--*-cookie-file` 指定路径）。

可选：如果需要更完整请求头，可用 `--weibo-headers-file` / `--bilibili-headers-file` 传入 txt（每行 `Header-Name: value`）。


## Cookie 与 Header 文件示例

默认文件名：`weibo_cookie.txt`、`bilibili_cookie.txt`、`xhs_cookie.txt`。

`weibo_cookie.txt` / `bilibili_cookie.txt` / `xhs_cookie.txt` 内容示例：

```txt
SESSDATA=xxx; bili_jct=xxx; ...
```

`weibo_headers.txt` 内容示例（B 站同理）：

```txt
X-Requested-With: XMLHttpRequest
Accept-Language: zh-CN,zh;q=0.9
```

运行示例：

```bash
python sentiment_topic_pipeline.py \
  --query 适老化设备 \
  --limit 80 \
  --weibo-cookie-file weibo_cookie.txt \
  --bilibili-cookie-file bilibili_cookie.txt \
  --xhs-cookie-file xhs_cookie.txt \
  --weibo-headers-file weibo_headers.txt \
  --bilibili-headers-file bilibili_headers.txt
```

## 分词说明

- 主题建模阶段使用 `jieba` 对中文文本分词，不再手动维护固定停用词列表。

## 输出文件

- `output/posts_with_topics.csv`：每条文本及其话题编号
- `output/topic_summary.csv`：每个话题的词与 Ollama 命名
- `output/analysis_report.json`：简要统计报告

## Jupyter 里运行

如果你在 Notebook 里执行脚本，`ipykernel` 会注入类似 `--f=...` 的参数。当前脚本已自动忽略未知参数，可直接运行。

也可以在 Notebook 中显式调用：

```python
from sentiment_topic_pipeline import main
main([])  # 使用默认参数
```


## API_URL 自检与修正

脚本内置了可覆盖的 API 参数：

- `--weibo-api-url`（默认：`https://m.weibo.cn/api/container/getIndex`）
- `--bilibili-api-url`（默认：`https://api.bilibili.com/x/web-interface/search/type`）
- `--xhs-api-url`（默认：`https://edith.xiaohongshu.com/api/sns/web/v1/search/notes`）

你可以先用很小样本做连通性验证：

```bash
python sentiment_topic_pipeline.py --query 适老化设备 --limit 3 --output-dir output_smoke
```

若某平台抓取日志持续报错（401/403/空结果/JSON 结构变化），可按下面步骤定位正确 URL：

1. 浏览器打开平台搜索页（微博/B站/小红书），按 `F12` 打开开发者工具。
2. 进入 **Network**，勾选 `Preserve log`，并过滤 `Fetch/XHR`。
3. 在页面里重新执行一次搜索，观察返回“列表数据”的请求。
4. 点开请求，记录：
   - **Request URL**（候选 API）
   - **Query Params / Request Payload**（关键词、页码参数名）
   - **Request Headers**（如 `Cookie`、`x-s`、`x-t` 等鉴权字段）
5. 用 `curl` 或脚本复现该请求；若返回稳定 JSON 且含有列表字段，再写入命令行参数：

```bash
python sentiment_topic_pipeline.py \
  --query 适老化设备 \
  --limit 80 \
  --bilibili-api-url '你确认过的新URL' \
  --xhs-api-url '你确认过的新URL'
```

> 提示：平台接口变更频繁，`API_URL` 正确并不代表可直接抓取；鉴权 Cookie、签名头、Referer/Origin 通常同样关键。
