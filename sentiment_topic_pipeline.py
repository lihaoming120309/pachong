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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import jieba
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
class CollectorDiagnostic:
    platform: str
    collector: str
    api_url: str
    status: str = "not_started"
    total_posts: int = 0
    pages_succeeded: int = 0
    last_page: int = 0
    last_http_status: int | None = None
    last_error_type: str = ""
    last_error_message: str = ""
    last_content_type: str = ""
    last_response_snippet: str = ""
    hint: str = ""
    notes: list[str] = field(default_factory=list)

    def mark_error(
        self,
        *,
        page: int,
        exc: Exception | None = None,
        response: requests.Response | None = None,
        hint: str = "",
        note: str = "",
    ) -> None:
        self.status = "failed"
        self.last_page = page
        if exc is not None:
            self.last_error_type = type(exc).__name__
            self.last_error_message = str(exc)
        if response is not None:
            self.last_http_status = response.status_code
            self.last_content_type = response.headers.get("Content-Type", "")
            snippet = (response.text or "")[:300]
            self.last_response_snippet = snippet.replace("\n", " ")
        if hint:
            self.hint = hint
        if note:
            self.notes.append(note)

    def mark_warning(self, *, page: int, note: str, hint: str = "") -> None:
        if self.status == "not_started":
            self.status = "warning"
        self.last_page = page
        self.notes.append(note)
        if hint and not self.hint:
            self.hint = hint

    def mark_success_page(self, page: int, added_count: int) -> None:
        self.pages_succeeded += 1
        self.last_page = page
        self.total_posts += added_count
        if self.status in {"not_started", "warning"}:
            self.status = "running"

    def finalize(self) -> None:
        if self.status in {"not_started", "running"}:
            self.status = "ok" if self.total_posts > 0 else "warning"
            if self.total_posts == 0 and not self.hint:
                self.hint = "请求成功但未拿到帖子，可能关键词过窄或返回结构变化。"


def infer_hint_from_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    etype = type(exc).__name__.lower()
    if "jsondecodeerror" in etype or "expecting value" in msg:
        return "响应不是 JSON，可能被风控重定向到登录/验证页，或 API URL 已变更。"
    if "timeout" in msg:
        return "请求超时，建议稍后重试或降低并发/频率。"
    if "connection aborted" in msg or "connection reset" in msg or "10054" in msg:
        return "连接被对端重置，常见于反爬限流或 TLS/WAF 拦截。"
    if "name or service not known" in msg or "nodename nor servname" in msg:
        return "DNS 解析失败，请检查网络或域名是否正确。"
    return "请结合 HTTP 状态码、响应片段和请求头检查鉴权或接口参数。"


def infer_hint_from_status(status_code: int) -> str:
    if status_code in {401, 403}:
        return "鉴权失败：请检查 Cookie、签名头（如 x-s/x-t）及 Referer/Origin。"
    if status_code == 404:
        return "接口不存在：API_URL 可能失效或路径已调整。"
    if status_code == 429:
        return "触发限流：建议降低频率、增加重试间隔并使用稳定登录态。"
    if 500 <= status_code < 600:
        return "服务端异常：可稍后重试，或检查参数是否触发风控。"
    return "HTTP 异常：请检查请求参数与请求头是否与浏览器一致。"


def probe_json(
    response: requests.Response,
) -> tuple[dict[str, Any] | list[Any] | None, Exception | None]:
    try:
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, exc



def jieba_tokenizer(text: str) -> list[str]:
    """使用 jieba 进行中文分词，并过滤空白 token。"""
    return [tok.strip() for tok in jieba.lcut(text or "") if tok.strip()]

def load_text_from_file(file_path: str) -> str:
    """读取本地 txt 内容，找不到文件时返回空字符串。"""
    if not file_path:
        return ""
    path = Path(file_path).expanduser()
    if not path.exists():
        logger.warning("文件不存在: %s", path)
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_headers_from_file(file_path: str) -> dict[str, str]:
    """读取 Header 配置文件，每行格式：Header-Name: value。"""
    content = load_text_from_file(file_path)
    headers: dict[str, str] = {}
    if not content:
        return headers

    for line in content.splitlines():
        row = line.strip()
        if not row or row.startswith("#") or ":" not in row:
            continue
        key, value = row.split(":", 1)
        headers[key.strip()] = value.strip()

    return headers

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
    platform = ""

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.diagnostic = CollectorDiagnostic(
            platform=self.platform or self.__class__.__name__.lower(),
            collector=self.__class__.__name__,
            api_url=api_url,
        )

    def collect(self, query: str, limit: int = 100) -> list[Post]:
        raise NotImplementedError


class WeiboCollector(BaseCollector):
    """微博移动端公开搜索接口（无需登录时返回有限数据）。"""

    platform = "weibo"
    DEFAULT_API_URL = "https://m.weibo.cn/api/container/getIndex"

    def __init__(
        self,
        timeout: int = 10,
        cookie: str = "",
        extra_headers: dict[str, str] | None = None,
        api_url: str | None = None,
    ):
        self.timeout = timeout
        super().__init__((api_url or self.DEFAULT_API_URL).strip())
        self.session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://m.weibo.cn",
        }
        if cookie:
            headers["Cookie"] = cookie
        if extra_headers:
            headers.update(extra_headers)
        self.session.headers.update(headers)

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
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                self.diagnostic.mark_error(page=page, exc=exc, hint=infer_hint_from_exception(exc))
                logger.warning("微博抓取中断，第 %s 页失败: %s | 诊断建议: %s", page, exc, self.diagnostic.hint)
                break

            data, json_exc = probe_json(resp)
            if json_exc is not None:
                self.diagnostic.mark_error(
                    page=page,
                    exc=json_exc,
                    response=resp,
                    hint="返回体不是 JSON，可能命中登录页/验证页或接口路径失效。",
                )
                logger.warning("微博抓取中断，第 %s 页 JSON 解析失败: %s | HTTP=%s", page, json_exc, resp.status_code)
                break

            cards = (data or {}).get("data", {}).get("cards", [])
            if not cards:
                self.diagnostic.mark_warning(page=page, note="返回 cards 为空", hint="请求成功但无结果，可能关键词较窄或返回结构变化。")
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
                self.diagnostic.mark_warning(page=page, note="cards 存在但无 mblog 项", hint="返回结构可能变化（未找到 mblog 字段）。")
                break
            self.diagnostic.mark_success_page(page, page_items)
            page += 1

        self.diagnostic.finalize()
        return posts


class BilibiliCollector(BaseCollector):
    """B 站公开视频搜索接口。"""

    platform = "bilibili"
    DEFAULT_API_URL = "https://api.bilibili.com/x/web-interface/search/type"

    def __init__(
        self,
        timeout: int = 10,
        cookie: str = "",
        extra_headers: dict[str, str] | None = None,
        api_url: str | None = None,
    ):
        self.timeout = timeout
        super().__init__((api_url or self.DEFAULT_API_URL).strip())
        self.session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com",
            "Origin": "https://www.bilibili.com",
        }
        if cookie:
            headers["Cookie"] = cookie
        if extra_headers:
            headers.update(extra_headers)
        self.session.headers.update(headers)

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
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                self.diagnostic.mark_error(page=page, exc=exc, hint=infer_hint_from_exception(exc))
                logger.warning("B站抓取中断，第 %s 页失败: %s | 诊断建议: %s", page, exc, self.diagnostic.hint)
                break

            payload, json_exc = probe_json(resp)
            if json_exc is not None:
                self.diagnostic.mark_error(page=page, exc=json_exc, response=resp, hint="返回体不是 JSON，可能触发风控页。")
                logger.warning("B站抓取中断，第 %s 页 JSON 解析失败: %s | HTTP=%s", page, json_exc, resp.status_code)
                break

            result = (payload or {}).get("data", {}).get("result") or []
            if not result:
                self.diagnostic.mark_warning(page=page, note="data.result 为空", hint="请求成功但结果为空，可能关键词无结果或接口字段变化。")
                break

            page_items = 0
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
                page_items += 1
                if len(posts) >= limit:
                    break

            self.diagnostic.mark_success_page(page, page_items)
            page += 1

        self.diagnostic.finalize()
        return posts


class XiaohongshuCollector(BaseCollector):
    platform = "xiaohongshu"
    DEFAULT_API_URL = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"

    def __init__(self, cookie: str = "", timeout: int = 10, api_url: str | None = None):
        self.timeout = timeout
        super().__init__((api_url or self.DEFAULT_API_URL).strip())
        self.session = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/json;charset=UTF-8",
            "Origin": "https://www.xiaohongshu.com",
            "Referer": "https://www.xiaohongshu.com/",
        }
        if cookie:
            headers["Cookie"] = cookie
        self.session.headers.update(headers)

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
                resp = self.session.post(self.api_url, json=body, timeout=self.timeout)
                if resp.status_code in {401, 403}:
                    self.diagnostic.mark_error(
                        page=page,
                        response=resp,
                        hint=infer_hint_from_status(resp.status_code),
                        note="小红书常需 Cookie + x-s/x-t 等签名头",
                    )
                    logger.warning("小红书接口鉴权失败，HTTP=%s，建议检查 cookie/x-s/x-t。", resp.status_code)
                    break
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                self.diagnostic.mark_error(page=page, exc=exc, hint=infer_hint_from_exception(exc))
                logger.warning("小红书抓取中断，第 %s 页失败: %s | 诊断建议: %s", page, exc, self.diagnostic.hint)
                break

            payload, json_exc = probe_json(resp)
            if json_exc is not None:
                self.diagnostic.mark_error(page=page, exc=json_exc, response=resp, hint="返回体不是 JSON，可能是风控页或网关拦截。")
                logger.warning("小红书抓取中断，第 %s 页 JSON 解析失败: %s | HTTP=%s", page, json_exc, resp.status_code)
                break

            items = (payload or {}).get("data", {}).get("items") or []
            if not items:
                self.diagnostic.mark_warning(page=page, note="data.items 为空", hint="请求成功但无结果，可能参数/签名缺失或字段变化。")
                break

            page_items = 0
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
                page_items += 1
                if len(posts) >= limit:
                    break

            self.diagnostic.mark_success_page(page, page_items)
            page += 1

        self.diagnostic.finalize()
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
    weibo_cookie_file: str,
    bilibili_cookie_file: str,
    xhs_cookie_file: str,
    weibo_extra_headers: dict[str, str] | None,
    bilibili_extra_headers: dict[str, str] | None,
    ollama_model: str,
    weibo_api_url: str,
    bilibili_api_url: str,
    xhs_api_url: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    collectors: list[BaseCollector] = [
        WeiboCollector(
            cookie=load_text_from_file(weibo_cookie_file),
            extra_headers=weibo_extra_headers,
            api_url=weibo_api_url,
        ),
        BilibiliCollector(
            cookie=load_text_from_file(bilibili_cookie_file),
            extra_headers=bilibili_extra_headers,
            api_url=bilibili_api_url,
        ),
        XiaohongshuCollector(
            cookie=load_text_from_file(xhs_cookie_file),
            api_url=xhs_api_url,
        ),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_posts: list[Post] = []
    diagnostics: list[dict[str, Any]] = []
    for collector in collectors:
        name = collector.__class__.__name__
        logger.info("开始抓取: %s", name)
        posts = collector.collect(query=query, limit=per_platform_limit)
        logger.info("%s 抓取完成: %s 条", name, len(posts))
        all_posts.extend(posts)
        diagnostics.append(asdict(collector.diagnostic))

    (output_dir / "collector_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("抓取诊断已输出: %s", output_dir / "collector_diagnostics.json")

    if not all_posts:
        raise RuntimeError("未抓取到任何文本，请检查 output/collector_diagnostics.json 了解平台失败原因。")

    df = pd.DataFrame(asdict(p) for p in all_posts)
    df["merged_text"] = df[["title", "content"]].fillna("").agg(" ".join, axis=1).str.strip()
    df = df[df["merged_text"].str.len() > 2].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("抓取文本为空，无法建模。")

    logger.info("开始 BERTopic 聚类，样本数: %s", len(df))
    vectorizer_model = CountVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)
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
    parser.add_argument("--weibo-cookie-file", default="weibo_cookie.txt", help="微博 Cookie txt 文件")
    parser.add_argument("--weibo-headers-file", default="", help="微博额外 Headers txt 文件")
    parser.add_argument("--bilibili-cookie-file", default="bilibili_cookie.txt", help="B 站 Cookie txt 文件")
    parser.add_argument("--bilibili-headers-file", default="", help="B 站额外 Headers txt 文件")
    parser.add_argument("--xhs-cookie-file", default="xhs_cookie.txt", help="小红书 Cookie txt 文件")
    parser.add_argument("--ollama-model", default="qwen2.5:7b", help="Ollama 模型名")
    parser.add_argument(
        "--weibo-api-url",
        default=WeiboCollector.DEFAULT_API_URL,
        help="微博搜索 API 地址（默认值可用，一般无需修改）",
    )
    parser.add_argument(
        "--bilibili-api-url",
        default=BilibiliCollector.DEFAULT_API_URL,
        help="B站搜索 API 地址（平台变更时可覆盖）",
    )
    parser.add_argument(
        "--xhs-api-url",
        default=XiaohongshuCollector.DEFAULT_API_URL,
        help="小红书搜索 API 地址（平台变更时可覆盖）",
    )
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
        weibo_cookie_file=args.weibo_cookie_file,
        bilibili_cookie_file=args.bilibili_cookie_file,
        xhs_cookie_file=args.xhs_cookie_file,
        weibo_extra_headers=load_headers_from_file(args.weibo_headers_file),
        bilibili_extra_headers=load_headers_from_file(args.bilibili_headers_file),
        ollama_model=args.ollama_model,
        weibo_api_url=args.weibo_api_url,
        bilibili_api_url=args.bilibili_api_url,
        xhs_api_url=args.xhs_api_url,
    )


if __name__ == "__main__":
    main()
