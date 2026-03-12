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

> 小红书通常需要有效 Cookie（以及可能的签名参数），可通过 `--xhs-cookie` 传入。

## 输出文件

- `output/posts_with_topics.csv`：每条文本及其话题编号
- `output/topic_summary.csv`：每个话题的词与 Ollama 命名
- `output/analysis_report.json`：简要统计报告
