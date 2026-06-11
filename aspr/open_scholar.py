import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from openai import OpenAI
from pypdf import PdfReader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from aspr.lats import evaluate_paper_innovation
    from aspr.pdf_downloader import ACLPDFDownloader
    from aspr.prompts import generation_instance_prompts_summarization, prompts_keywords_extraction
else:
    from .lats import evaluate_paper_innovation
    from .pdf_downloader import ACLPDFDownloader
    from .prompts import generation_instance_prompts_summarization, prompts_keywords_extraction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOWNLOAD_DIR = OUTPUTS_DIR / "downloads"
RETRIEVAL_CACHE_DIR = DATA_DIR / "retrieval_cache"
TEMP_PROMPT_PATH = OUTPUTS_DIR / "temp.json"


def _query_cache_paths(title: str, abstract: str, key_words: List[str]) -> Tuple[Path, Path]:
    payload = {
        "title": title.strip(),
        "abstract": abstract.strip(),
        "key_words": [kw.strip().lower() for kw in key_words],
    }
    cache_key = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return (
        RETRIEVAL_CACHE_DIR / f"{cache_key}_most_related_papers.json",
        RETRIEVAL_CACHE_DIR / f"{cache_key}_total_related_papers.jsonl",
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "and"}
    return bool(value)


def keywords_extract(query: str) -> List[str]:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    llm = ChatOpenAI(
        model="qwen3:8b",  
        base_url="http://localhost:11434/v1",  
        api_key="ollama",  
        temperature=0.1,
        max_tokens=2000,
    )
    keyword_agent = (
        PromptTemplate.from_template(prompts_keywords_extraction) |
        llm |
        StrOutputParser()
    )
    keywords = keyword_agent.invoke({"abstract": query})
    try:
        keywords = keywords.strip()
        if "Keywords:" in keywords or "keywords:" in keywords:
            keywords = keywords.split(":", 1)[1].strip()
        
        keywords_list = [kw.strip() for kw in keywords.split(",")]
        keywords_list = [kw.strip(".,;:\"'[]()") for kw in keywords_list if kw.strip()]
        keywords_list = keywords_list[:5]
        print(f"Extracted keywords: {keywords_list}")
        return keywords_list
        
    except Exception as e:
        print(f"Error parsing keywords: {e}")
        print(f"Raw output: {keywords}")
        return []


@lru_cache(maxsize=1)
def _get_recall_model() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


@lru_cache(maxsize=1)
def _get_reranker() -> FlagReranker:
    return FlagReranker("OpenSciLM/OpenScholar_Reranker", use_fp16=True)


def retrieval_recall(query: str, reference: List[str]) -> Tuple[List[str], List[float]]:
    model = _get_recall_model()
    sentence_pairs = [(query, ref) for ref in reference]
    similarity_scores = model.compute_score(
        sentence_pairs,
        max_passage_length=2048,  # a smaller max length leads to a lower latency
        weights_for_different_modes=[0.4, 0.2, 0.4],
        batch_size=100
    )['colbert+sparse+dense']
    paired = list(zip(reference, similarity_scores))
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
    sorted_references = [p for p, s in paired_sorted]
    sorted_scores = [s for p, s in paired_sorted]
    # print(sorted_scores)
    return sorted_references, sorted_scores
    

def retrieval_rerank(query: str, reference: List[str]) -> Tuple[List[str], List[float]]:
    model = _get_reranker()
    sentence_pairs = [(query, ref) for ref in reference]
    rerank_scores = model.compute_score(
        sentence_pairs,
        batch_size=100
    )
    paired = list(zip(reference, rerank_scores))
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
    sorted_references = [p for p, s in paired_sorted]
    sorted_scores = [s for p, s in paired_sorted]

    return sorted_references, sorted_scores


def extract_text_with_pypdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

class Reviewer:
    def __init__(self, args):
        self.args = args
        self.clinet_small = None
        self.client_large = None
        self.open_scholar = None
        self.pdf_downloader = ACLPDFDownloader(max_retries=2, retry_delay=3.0)
        self.save_path = str(DOWNLOAD_DIR)
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        self.initialize_models()


    def initialize_models(self,):
        self.client_large = OpenAI(
            api_key="",
            base_url=f'http://localhost:{self.args.large_model_port}/v1',
        )
        self.open_scholar = OpenScholar(
            args=self.args
        )

    def __call__(self, title: str, abstract: str, key_words: List[str]):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RETRIEVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not key_words:
            key_words = keywords_extract(abstract)

        most_related_path, total_related_path = _query_cache_paths(title, abstract, key_words)

        if most_related_path.exists():
            print(f"Loading papers from {most_related_path}...")
            with most_related_path.open("r", encoding="utf-8") as file:
                paper_after_retrieval = json.load(file)
            print(f"Loaded {len(paper_after_retrieval)} papers from cache.")
        else:
            if total_related_path.exists():
                with total_related_path.open("r", encoding="utf-8") as file:
                    papers = [json.loads(line.strip()) for line in file if line.strip()]
            else:
                papers = self.open_scholar.search_semantic_scholar(key_words)
                with total_related_path.open("w", encoding="utf-8") as file:
                    for paper in papers:
                        print(json.dumps(paper, ensure_ascii=False), file=file)
            
            item2Id, Id2paper = {}, {}
            paper_formatted = []
            for idx, paper in enumerate(papers):
                item = f'Title:{paper.get("title", "")}. Abstract:{paper.get("abstract", "")}'
                paper_formatted.append(item)
                item2Id[item] = paper["paperId"]
                Id2paper[paper["paperId"]] = paper

            if not paper_formatted:
                paper_after_retrieval = []
                with most_related_path.open("w", encoding="utf-8") as file:
                    json.dump(paper_after_retrieval, file, indent=2, ensure_ascii=False)
                return evaluate_paper_innovation(
                    paper_title=title,
                    paper_abstract=abstract,
                    retrieved_papers=paper_after_retrieval,
                )

            print('Start retrieval recall...')
            paper_recalled, _ = retrieval_recall(title + "\n" + abstract, paper_formatted)
            top_n = max(1, int(getattr(self.args, "top_n", 10)))
            recall_n = min(len(paper_recalled), max(top_n * 5, round(len(paper_recalled) / 10), top_n))
            paper_recalled = paper_recalled[:recall_n]
            print('Recalled papers:', len(paper_recalled))
            print('Start retrieval rerank...')
            paper_reranked, _ = retrieval_rerank(title + "\n" + abstract, paper_recalled)
            paper_reranked = paper_reranked[:min(len(paper_reranked), top_n)]
            print('Reranked papers:', len(paper_reranked))

            # 去重
            seen_ids = set()
            paper_after_retrieval = []
            for item in paper_reranked:
                paper_id = item2Id[item]
                if paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    paper_after_retrieval.append(Id2paper[paper_id])
            
            with most_related_path.open("w", encoding="utf-8") as file:
                json.dump(paper_after_retrieval, file, indent=2, ensure_ascii=False)
            print(f"Saved {len(paper_after_retrieval)} papers to {most_related_path}")

        
        reviews = evaluate_paper_innovation(
            paper_title=title,
            paper_abstract=abstract,
            retrieved_papers=paper_after_retrieval
        )

        print(reviews)
        return reviews
        # success_id, failed_id = self._paper_download(paper_after_retrieval)
        # success_id = ['5bea7828c7a5aeaac8fc86e2012d8fa43ba64242', 'ec1c43ca684732d06716a36271a4cb3066797153', '0b9d0bee85e4ef4261147f35be885010e62ad1fb']
        # reference_rag, reference_scholar = "", ""
        # for idx, item in enumerate(paper_after_retrieval):
        #     if item["paperId"] in success_id:
        #         reference_rag += extract_text_with_pypdf(os.path.join(self.save_path, f'{item["paperId"]}.pdf'))
        #     else:
        #         reference_rag += f'Title:{item["title"]}. Abstract:{item["abstract"]}\n'
        #     reference_scholar += f'[{idx}]. Title:{item["title"]}. Abstract:{item["abstract"]}\n'
        
        # graph_rag.insert(reference_rag)
        # response = graph_rag.query(
        #     query=f'What are the novel contributions of {input} compared to the foundational work?',
        #     mode='global'
        # )
        
        # review = self._generate_review(reference_scholar, input, "")
        # print(review)
        # return review

    def _paper_download(self, paper_after_retrieval):
        success_id = []
        for paper in paper_after_retrieval:
            if paper["isOpenAccess"]:
                try:
                    url = paper["url"]
                    filename = f'{paper["paperId"]}.pdf'
                    saved_file = self.pdf_downloader.download_acl_pdf(
                        url,
                        save_dir=self.save_path,
                        filename=filename
                    )
                    if saved_file and os.path.exists(saved_file):
                        file_size = os.path.getsize(saved_file)
                        print(f"✓ 下载成功: {saved_file} ({file_size:,} bytes)")
                        success_id.append(paper["paperId"])
                    else:
                        print("✗ 下载失败")
                except Exception as e:
                    print(f"✗ 下载失败: {e}")
                    continue
        failed_id = [paper["paperId"] for paper in paper_after_retrieval if paper["paperId"] not in success_id]
        return success_id, failed_id

    def _generate_review(self, reference, abstract, innovation):
        # rank papers
        
        input_query = generation_instance_prompts_summarization.format_map({
            "reference":reference, 
            "abstract":abstract,
            "innovation":innovation
        })
        input_query = self._formate_llama3_prompt(input_query)

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with TEMP_PROMPT_PATH.open("w", encoding="utf-8") as file:
            print(json.dumps(input_query), file=file)

        # response = self.client_large.chat.completions.create(
        #     model=self.args.large_model,
        #     messages=[{"role":"user", "content":input_query}],
        #     temperature=0.7,
        #     max_tokens=self.args.max_tokens,
        #     stream=False,
        #     timeout=300
        # )
        # content = response.choices[0].message['content']

        # return content

    def _formate_llama3_prompt(self, prompt):
        formatted_text = "<|begin_of_text|>"
        formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
        formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted_text

class OpenScholar:
    def __init__(self, args):
        self.s2_api_key = args.s2_api_key
        self.and_search = _as_bool(args.and_search)
        self.url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

    def __call__(self,):
        pass

    def search_semantic_scholar(self, key_words: List[str]) -> List[Dict[str, Any]]:
        papers = []
        for kw in key_words:
            print(f"Searching for papers with keyword: {kw}")
            papers_kw = self._search_paper_via_query(kw)
            print(f"Found {len(papers_kw)} papers for keyword: {kw}")
            papers.extend(papers_kw)
        print(f"Retrieved {len(papers)} papers...")
        formatted_papers = []
        for paper in papers:
            open_access_pdf = paper.get("openAccessPdf") or {}
            external_ids = paper.get("externalIds") or {}
            formatted_papers.append({
                "paperId": paper.get("paperId", ""),
                "year": paper.get("year") or 0,
                "title": paper.get("title") or "",
                "authors": ", ".join([author.get("name", "") for author in paper.get("authors") or []]),
                "venue": paper.get("venue") or "",
                "citationCount": paper.get("citationCount") or 0,
                "abstract": paper.get("abstract") or "",
                "isOpenAccess": bool(paper.get("isOpenAccess")),
                "url": open_access_pdf.get("url") or paper.get("url") or "",
                "externalIds": external_ids,
                "doi": external_ids.get("DOI") or external_ids.get("doi") or "",
                "fieldsOfStudy": paper.get("fieldsOfStudy") or [],
                "s2FieldsOfStudy": paper.get("s2FieldsOfStudy") or [],
            })

        return formatted_papers

    def _search_paper_via_query(self, query: str | List[str]) -> List[Dict[str, Any]]:
        terms = [query] if isinstance(query, str) else query
        terms = [term.strip() for term in terms if str(term).strip()]
        if not terms:
            return []
        separator = " + " if self.and_search else " | "
        query = separator.join([f'"{term}"' for term in terms])
        query_params = {
            'query': query,
            'fields': (
                "paperId,title,year,authors.name,abstract,venue,citationCount,url,"
                "externalIds,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy"
            ),
            "year": "2021-",
            "sort": "citationCount:desc"
        }
        headers = {"x-api-key":self.s2_api_key}
        response = requests.get(
            self.url,
            params=query_params,
            headers=headers
        )
        if response.status_code == 200:
            response_data = response.json().get("data", [])
        else:
            raise RuntimeError(f"Semantic Scholar request failed: {response.status_code} {response.text[:200]}")

        return response_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenScholar API Server')
    parser.add_argument('--s2_api_key', type=str, default=os.getenv("S2_API_KEY", ""),
                        help='Semantic Scholar API key or S2_API_KEY environment variable')
    parser.add_argument('--large_model', type=str, default='OpenSciLM/Llama-3.1_OpenScholar-8B',
                        help='Large model name')
    parser.add_argument('--large_model_port', type=int, default=38011,
                        help='Port for large model server')
    parser.add_argument('--small_model', type=str, default='Qwen/Qwen3-0.6B',
                        help='Small model name')
    parser.add_argument('--small_model_port', type=int, default=38014,
                        help='Port for small model server')
    parser.add_argument('--api_port', type=int, default=38015,
                        help='Port for API server')
    parser.add_argument('--and_search', type=_as_bool, default=False,
                        help='and / or search')
    parser.add_argument('--reranker_path', type=str, default='OpenSciLM/OpenScholar_Reranker',
                        help='Path to reranker model')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Top N papers to retrieve')
    parser.add_argument('--max_tokens', type=int, default=3000,
                        help='Maximum tokens for generation')
    parser.add_argument('--search_batch_size', type=int, default=100,
                        help='Batch size for search generation')
    parser.add_argument('--scholar_batch_size', type=int, default=100,
                        help='Batch size for OpenScholar processing')
    args = parser.parse_args()

    # key_words = ["Human noroviruses", "GII.4", "Nanobody M4", "Neutralization", "Epochal evolution","Raised conformation"]
    title = "PQBP5/NOL10 maintains and anchors the nucleolus under physiological and osmotic stress conditions"
    key_words = []
    abstract = """
Polyglutamine binding protein 5 (PQBP5), also called nucleolar protein 10 (NOL10), binds to polyglutamine tract sequences and is expressed in the nucleolus. Using dynamic imaging of high-speed atomic force microscopy, we show that PQBP5/NOL10 is an intrinsically disordered protein. Superresolution microscopy and correlative light and electron microscopy method show that PQBP5/NOL10 makes up the skeletal structure of the nucleolus, constituting the granule meshwork in the granular component area, which is distinct from other nucleolar substructures, such as the fibrillar center and dense fibrillar component. In contrast to other nucleolar proteins, which disperse to the nucleoplasm under osmotic stress conditions, PQBP5/NOL10 remains in the nucleolus and functions as an anchor for reassembly of other nucleolar proteins. Droplet and thermal shift assays show that the biophysical features of PQBP5/NOL10 remain stable under stress conditions, explaining the spatial role of this protein. PQBP5/NOL10 can be functionally depleted by sequestration with polyglutamine disease proteins in vitro and in vivo, leading to the pathological deformity or disappearance of the nucleolus. Taken together, these findings indicate that PQBP5/NOL10 is an essential protein needed to maintain the structure of the nucleolus.
    """

    server = Reviewer(args)
    server(title, abstract, key_words)
