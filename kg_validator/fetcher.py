"""
fetcher.py — OpenAlex 数据拉取模块
负责从 OpenAlex API 拉取论文数据，构建原始数据集
"""

import time
import logging
import requests
from typing import Optional

log = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"

# 拉取字段：构建图所需的最小集合
WORK_FIELDS = ",".join([
    "id", "doi", "title", "publication_year",
    "primary_location", "topics",
    "referenced_works", "cited_by_count",
])


def _oa_id(full_id: str) -> str:
    """'https://openalex.org/W123' → 'W123'"""
    return full_id.split("/")[-1] if full_id else ""


def fetch_works_by_doi(dois: list[str], email: str = "") -> list[dict]:
    """批量用 DOI 查询论文（每批 50 个）"""
    results = []
    batch_size = 50
    for i in range(0, len(dois), batch_size):
        batch = dois[i: i + batch_size]
        ids_str = "|".join(f"https://doi.org/{d.strip()}" for d in batch)
        params = {
            "filter": f"doi:{ids_str}",
            "select": WORK_FIELDS,
            "per_page": batch_size,
        }
        if email:
            params["mailto"] = email
        try:
            r = requests.get(f"{OPENALEX_BASE}/works", params=params, timeout=30)
            r.raise_for_status()
            results.extend(r.json().get("results", []))
            log.info(f"  DOI 批次 {i//batch_size+1}: 获得 {len(r.json().get('results',[]))} 篇")
        except Exception as e:
            log.warning(f"  DOI 批次 {i} 请求失败: {e}")
        time.sleep(0.15)
    return results


def fetch_works_cursor(filter_str: str, email: str = "",
                       per_page: int = 100, max_records: Optional[int] = None) -> list[dict]:
    """cursor 分页拉取满足条件的所有论文"""
    results, cursor, total = [], "*", 0
    while True:
        params = {
            "filter":   filter_str,
            "select":   WORK_FIELDS,
            "per_page": per_page,
            "cursor":   cursor,
        }
        if email:
            params["mailto"] = email
        try:
            r = requests.get(f"{OPENALEX_BASE}/works", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.error(f"  cursor 请求失败: {e}")
            time.sleep(5)
            continue

        page = data.get("results", [])
        if not page:
            break
        results.extend(page)
        total += len(page)
        meta = data.get("meta", {})
        log.info(f"  已拉取 {total} / {meta.get('count','?')} 篇")
        cursor = meta.get("next_cursor")
        if not cursor or (max_records and total >= max_records):
            break
        time.sleep(0.12)
    return results


def fetch_works_batch_ids(work_ids: list[str], email: str = "") -> list[dict]:
    """通过 OpenAlex ID 批量拉取（每批 50 个）"""
    results = []
    batch_size = 50
    for i in range(0, len(work_ids), batch_size):
        batch = work_ids[i: i + batch_size]
        ids_str = "|".join(batch)
        params = {
            "filter": f"openalex_id:{ids_str}",
            "select": WORK_FIELDS,
            "per_page": batch_size,
        }
        if email:
            params["mailto"] = email
        try:
            r = requests.get(f"{OPENALEX_BASE}/works", params=params, timeout=30)
            r.raise_for_status()
            results.extend(r.json().get("results", []))
        except Exception as e:
            log.warning(f"  ID 批次 {i} 请求失败: {e}")
        time.sleep(0.12)
    return results


def fetch_citing_works(target_id: str, email: str = "",
                       max_records: int = 2000) -> list[dict]:
    """拉取引用了 target_id 的所有论文（入引，用于 CD 指数计算）"""
    filter_str = f"cites:{target_id}"
    return fetch_works_cursor(filter_str, email=email,
                              per_page=100, max_records=max_records)


def normalize_work(w: dict) -> dict:
    """标准化单篇论文字典，统一字段格式"""
    topics = w.get("topics") or []
    loc    = w.get("primary_location") or {}
    source = loc.get("source") or {}
    return {
        "id":              _oa_id(w.get("id", "")),
        "doi":             (w.get("doi") or "").replace("https://doi.org/", ""),
        "title":           w.get("title", ""),
        "year":            w.get("publication_year"),
        "cited_by_count":  w.get("cited_by_count", 0),
        "journal":         source.get("display_name", ""),
        "domain":          topics[0].get("domain",{}).get("display_name","") if topics else "",
        "field":           topics[0].get("field",{}).get("display_name","") if topics else "",
        "subfield":        topics[0].get("subfield",{}).get("display_name","") if topics else "",
        "topic_names":     [t.get("display_name","") for t in topics[:5]],
        "referenced_works": [_oa_id(r) for r in (w.get("referenced_works") or [])],
    }
