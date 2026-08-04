import asyncio
import json
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler

DATA_DIR = Path("data/landing/news")


def setup_directory():
    """Tạo thư mục data/landing/news nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://admissions.vinuni.edu.vn/vi/vinuni-hoc-bong-nam-hoc-2026-2027/",
    "https://admissions.vinuni.edu.vn/vi/hoc-bong-va-ho-tro-tai-chinh/cu-nhan/ho-tro-tai-chinh/",
    "https://admissions.vinuni.edu.vn/vi/hoc-phi/cu-nhan/",
    "https://families.vinuni.edu.vn/vi/vinuni-community-trao-tang-1-000-voucher-ezmeal-cho-sinh-vien-vinuni/",
    "https://registrar.vinuni.edu.vn/vi/2025/09/05/thong-bao-khai-giang-nam-hoc-2025-2026-hoc-ky-thu-2025/"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về metadata + markdown.
    """

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)

        return {
            "url": url,
            "title": result.metadata.get("title", ""),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, start=1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

        article = await crawl_article(url)

        filepath = DATA_DIR / f"article_{i:02d}.json"

        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        asyncio.run(crawl_all())