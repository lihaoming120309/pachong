# pachong

用于抓取微博、B站、小红书关于“适老化设备”的舆情，并使用 BERTopic 分类，再用 Ollama 自动命名主题。

## 快速开始

1. 安装依赖：

```bash
pip install pandas requests scikit-learn bertopic sentence-transformers umap-learn hdbscan
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
