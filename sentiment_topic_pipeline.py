#!/usr/bin/env python3
"""多平台“适老化设备”舆情收集 + BERTopic 聚类 + Ollama 主题命名。

说明：
1. 微博与 B 站接口为公开检索接口，可能存在反爬限制。
2. 小红书接口风控较严格，通常需要你提供有效 Cookie 与签名参数。
3. 请在使用前确认平台条款与当地法律法规。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class Post:
    platform: str
    post_id: str
    title: str
    content: str
    url: str
    published_at: str = ""
    extra: dict | None = None

    @property
    def merged_text(self) -> str:
        return " ".join(x for x in [self.title, self.content] if x).strip()


class BaseCollector:
    def collect(self, query: str, limit: int = 100) -> list[Post]:
        raise NotImplementedError


class WeiboCollector(BaseCollector):
    """微博移动端公开搜索接口（无需登录时返回有限数据）。"""

    API_URL = "https://m.weibo.cn/api/container/getIndex"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
        )

    @staticmethod
    def _clean_html(raw_html: str) -> str:
        return re.sub(r"<[^>]+>", "", raw_html or "").strip()

    def collect(self, query: str, limit: int = 100) -> list[Post]:
        posts: list[Post] = []
        page = 1
        while len(posts) < limit:
            params = {
                "containerid": f"100103type=1&q={query}",
                "page_type": "searchall",
                "page": page,
            }
            try:
                resp = self.session.get(self.API_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("微博抓取中断，第 %s 页失败: %s", page, exc)
                break

            cards = data.get("data", {}).get("cards", [])
            if not cards:
                break

            page_items = 0
            for card in cards:
                mblog = card.get("mblog")
                if not mblog:
                    continue
                text = self._clean_html(mblog.get("text", ""))
                post = Post(
                    platform="weibo",
                    post_id=str(mblog.get("id", "")),
                    title="",
                    content=text,
                    url=f"https://m.weibo.cn/detail/{mblog.get('id', '')}",
                    published_at=mblog.get("created_at", ""),
                    extra={"reposts_count": mblog.get("reposts_count")},
                )
                posts.append(post)
                page_items += 1
                if len(posts) >= limit:
                    break

            if page_items == 0:
                break
            page += 1

        return posts


class BilibiliCollector(BaseCollector):
    """B 站公开视频搜索接口。"""

    API_URL = "https://api.bilibili.com/x/web-interface/search/type"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.bilibili.com",
            }
        )

    @staticmethod
    def _clean_highlight_text(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text or "").strip()

    def collect(self, query: str, limit: int = 100) -> list[Post]:
        posts: list[Post] = []
        page = 1
        while len(posts) < limit:
            params = {
                "search_type": "video",
                "keyword": query,
                "page": page,
            }
            try:
                resp = self.session.get(self.API_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("B站抓取中断，第 %s 页失败: %s", page, exc)
                break

            result = payload.get("data", {}).get("result") or []
            if not result:
                break

            for item in result:
                bvid = item.get("bvid", "")
                posts.append(
                    Post(
                        platform="bilibili",
                        post_id=bvid,
                        title=self._clean_highlight_text(item.get("title", "")),
                        content=self._clean_highlight_text(item.get("description", "")),
                        url=f"https://www.bilibili.com/video/{bvid}",
                        published_at=str(item.get("pubdate", "")),
                        extra={"author": item.get("author")},
                    )
                )
                if len(posts) >= limit:
                    break

            page += 1

        return posts


class XiaohongshuCollector(BaseCollector):
    API_URL = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"

    def __init__(self, cookie: str = "", x_s: str = "", x_t: str = "", timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "...",
                "Content-Type": "application/json;charset=UTF-8",
                "Origin": "https://www.xiaohongshu.com",
                "Referer": "https://www.xiaohongshu.com/",
                "Cookie": cookie,
                "x-s": x_s,
                "x-t": x_t,
            }
        )

    def collect(self, query: str, limit: int = 100) -> list[Post]:
        posts: list[Post] = []
        page = 1
        while len(posts) < limit:
            body = {
                "keyword": query,
                "page": page,
                "page_size": min(20, limit - len(posts)),
                "search_id": "",  # 可留空，服务端可能返回默认结果
                "sort": "general",
                "note_type": 0,
            }
            try:
                resp = self.session.post(self.API_URL, json=body, timeout=self.timeout)
                if resp.status_code in {401, 403}:
                    logger.warning("小红书接口鉴权失败，请配置有效 cookie/x-s/x-t。")
                    break
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("小红书抓取中断，第 %s 页失败: %s", page, exc)
                break

            items = payload.get("data", {}).get("items") or []
            if not items:
                break

            for item in items:
                note = item.get("note_card") or {}
                note_id = note.get("note_id", "")
                posts.append(
                    Post(
                        platform="xiaohongshu",
                        post_id=note_id,
                        title=note.get("display_title", ""),
                        content=note.get("desc", ""),
                        url=f"https://www.xiaohongshu.com/explore/{note_id}",
                        published_at=str(note.get("time", "")),
                        extra={"user": (note.get("user") or {}).get("nickname")},
                    )
                )
                if len(posts) >= limit:
                    break

            page += 1

        return posts


class OllamaTopicNamer:
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def name_topic(self, representative_texts: Iterable[str]) -> str:
        prompt = (
            "你是舆情分析助手。请根据下面若干条文本，为该主题生成一个简短中文名称（<=10字）。"
            "只返回主题名称，不要解释。\n\n"
            + "\n".join(f"- {t[:140]}" for t in representative_texts)
        )
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            return result or "未命名主题"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama 命名失败: %s", exc)
            return "未命名主题"


def run_pipeline(
    query: str,
    per_platform_limit: int,
    output_dir: Path,
    xhs_cookie: str,
    ollama_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    collectors: list[BaseCollector] = [
        WeiboCollector(),
        BilibiliCollector(),
        XiaohongshuCollector(cookie=xhs_cookie),
    ]

    all_posts: list[Post] = []
    for collector in collectors:
        name = collector.__class__.__name__
        logger.info("开始抓取: %s", name)
        posts = collector.collect(query=query, limit=per_platform_limit)
        logger.info("%s 抓取完成: %s 条", name, len(posts))
        all_posts.extend(posts)

    if not all_posts:
        raise RuntimeError("未抓取到任何文本，请检查网络、关键词或平台鉴权信息。")

    df = pd.DataFrame(asdict(p) for p in all_posts)
    df["merged_text"] = df[["title", "content"]].fillna("").agg(" ".join, axis=1).str.strip()
    df = df[df["merged_text"].str.len() > 2].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("抓取文本为空，无法建模。")

    logger.info("开始 BERTopic 聚类，样本数: %s", len(df))
    vectorizer_model = CountVectorizer(stop_words=["的", "了", "和", "是", "就", "都", "而", "及"])
    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer_model,
        min_topic_size=max(5, len(df) // 30),
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(df["merged_text"].tolist())
    df["topic_id"] = topics
    topic_probs = []
    if probs is None:
        topic_probs = [None] * len(df)
    else:
        for p in probs:
            if p is None:
                topic_probs.append(None)
            elif isinstance(p, (float, int)):
                topic_probs.append(float(p))
            else:
                try:
                    topic_probs.append(float(max(p)))
                except Exception:
                    topic_probs.append(None)

    df["topic_prob"] = topic_probs


    info = topic_model.get_topic_info().copy()
    info = info[info["Topic"] != -1].reset_index(drop=True)

    namer = OllamaTopicNamer(model=ollama_model)
    custom_names: list[str] = []
    for topic_id in info["Topic"].tolist():
        reps = topic_model.get_representative_docs(topic_id) or []
        custom_names.append(namer.name_topic(reps[:5]))

    info["custom_name"] = custom_names

    # 输出结果
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "posts_with_topics.csv", index=False, encoding="utf-8-sig")
    info.to_csv(output_dir / "topic_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "query": query,
        "sample_size": int(len(df)),
        "platform_counts": df["platform"].value_counts().to_dict(),
        "topics": info[["Topic", "Count", "Name", "custom_name"]].to_dict(orient="records"),
    }
    (output_dir / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("分析完成，结果输出目录: %s", output_dir)
    return df, info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多平台舆情抓取 + BERTopic + Ollama 命名")
    parser.add_argument("--query", default="适老化设备", help="搜索关键词")
    parser.add_argument("--limit", type=int, default=80, help="每个平台抓取上限")
    parser.add_argument("--output-dir", default="output", help="结果输出目录")
    parser.add_argument("--xhs-cookie", default="", help="小红书 Cookie")
    parser.add_argument("--ollama-model", default="qwen2.5:7b", help="Ollama 模型名")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.info("忽略未知参数: %s", unknown)
    run_pipeline(
        query=args.query,
        per_platform_limit=args.limit,
        output_dir=Path(args.output_dir),
        xhs_cookie=args.xhs_cookie,
        ollama_model=args.ollama_model,
    )


if __name__ == "__main__":
    main()
