import argparse
import requests
import json
import os
import graph_rag
from pathlib import Path
from openai import OpenAI
from pypdf import PdfReader
from FlagEmbedding import BGEM3FlagModel,FlagReranker
from prompts import generation_instance_prompts_summarization
from pdf_downloader import ACLPDFDownloader

def retrieval_recall(query, reference):
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
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
    
def retrieval_rerank(query, reference):
    model = FlagReranker("OpenSciLM/OpenScholar_Reranker", use_fp16=True)
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

def extract_text_with_pypdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

class Reviewer:
    def __init__(self, args):
        self.args = args
        self.client_large = None
        self.open_scholar = None
        self.pdf_downloader = ACLPDFDownloader(max_retries=2, retry_delay=3.0)
        self.save_path = "./downloads"
        Path(self.save_path).mkdir(exist_ok=True)

        self.initialize_models()

    def initialize_models(self,):
        self.client_large = OpenAI(
            api_key="",
            base_url=f'http://localhost:{self.args.large_model_port}/v1',
        )
        self.open_scholar = OpenScholar(
            args=self.args
        )

    def __call__(self, key_words, input):
        if os.path.exists("papers.json"):
            with open("papers.json", "r") as file:
                papers = [json.loads(line.strip()) for line in file if line.strip()]
        else:
            papers = self.open_scholar.search_semantic_scholar(key_words)
            with open("papers.json", "w") as file:
                for paper in papers:
                    print(json.dumps(paper), file=file)
        
        paper2Id, Id2paper = {}, {}
        paper_formatted = []
        for idx, paper in enumerate(papers):
            item = f'Title:{paper["title"]}. Abstract:{paper["abstract"]}'
            paper_formatted.append(item)
            paper2Id[item] = paper["paperId"]
            Id2paper[paper["paperId"]] = paper

        paper_recalled, _ = retrieval_recall(input, paper_formatted)
        paper_recalled = paper_recalled[:round(len(paper_recalled)/10)]
        paper_reranked, _ = retrieval_rerank(input, paper_recalled)
        paper_reranked = paper_reranked[:round(len(paper_reranked)/10)]

        paper_after_retrieval = [Id2paper[paper2Id[item]] for item in paper_reranked] 
        # success_id, failed_id = self._paper_download(paper_after_retrieval)
        success_id = ['5bea7828c7a5aeaac8fc86e2012d8fa43ba64242', 'ec1c43ca684732d06716a36271a4cb3066797153', '0b9d0bee85e4ef4261147f35be885010e62ad1fb']
        reference_rag, reference_scholar = "", ""
        for idx, item in enumerate(paper_after_retrieval):
            if item["paperId"] in success_id:
                reference_rag += extract_text_with_pypdf(os.path.join(self.save_path, f'{item["paperId"]}.pdf'))
            else:
                reference_rag += f'Title:{item["title"]}. Abstract:{item["abstract"]}\n'
            reference_scholar += f'[{idx}]. Title:{item["title"]}. Abstract:{item["abstract"]}\n'
        
        # graph_rag.insert(reference_rag)
        response = graph_rag.query(
            query=f'What are the novel contributions of {input} compared to the foundational work?',
            mode='global'
        )
        
        review = self._generate_review(reference_scholar, input, response)
        print(review)
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

        response = self.client_large.chat.completions.create(
            model=self.args.large_model,
            messages=[{"role":"user", "content":input_query}],
            temperature=0.7,
            max_tokens=self.args.max_tokens,
            stream=False,
            timeout=300
        )
        content = response.choices[0].message['content']

        return content

    def _formate_llama3_prompt(self, prompt):
        formatted_text = "<|begin_of_text|>"
        formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
        formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted_text

class OpenScholar:
    def __init__(self, args):
        self.s2_api_key = args.s2_api_key
        self.and_search = args.and_search
        self.url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

    def __call__(self,):
        pass

    def search_semantic_scholar(self, key_words):
        papers = self._search_paper_via_query(key_words)
        print(f"Retrieved {len(papers)} papers...")
        formatted_papers = []
        for paper in papers:
            formatted_papers.append({
                "paperId":paper["paperId"],
                "year":paper["year"],
                "title":paper["title"],
                "authors": (', ').join([author["name"] for author in paper["authors"]]),
                "venue":paper["venue"],
                "citationCount":paper["citationCount"],
                "abstract":paper["abstract"],
                "isOpenAccess":paper["isOpenAccess"],
                "url":paper["openAccessPdf"]["url"]
            })

        return formatted_papers

    def _search_paper_via_query(self, query):
        if self.and_search:
            query = ' + '.join([f'"{kw}"' for kw in query])
        else:
            query = ' | '.join([f'"{kw}"' for kw in query])
        query_params = {
            'query': query,
            'fields': "paperId,title,year,authors.name,abstract,venue,citationCount,url,externalIds,isOpenAccess,openAccessPdf",
            "year": "2023-",
            "sort": "citationCount:desc"
        }
        headers = {"x-api-key":self.s2_api_key}
        response = requests.get(
            self.url,
            params=query_params,
            headers=headers
        )
        if response.status_code == 200:
            response_data = response.json()["data"]
        else:
            raise response.status_code

        return response_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenScholar API Server')
    parser.add_argument('--s2_api_key', type=str, default='RdC229ErL37in7bNEmR7W5MFVrd3pzpv1SWWbrLt',
                        help='Semantic Scholar API key')
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
    parser.add_argument('--and_search', type=bool, default='False',
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
    key_words = ["generative ai"]

    query = "Generative artificial intelligence (AI) has revolutionized AI by enabling high-fidelity content creation across text, images, audio, and structured data. This survey explores the core methodologies, advancements, applications, and ongoing challenges of generative AI, covering key models such as Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models, and Transformer-based architectures. These innovations have driven breakthroughs in healthcare, scientific computing, Natural Language Processing (NLP), computer vision, and autonomous systems. Despite its progress, generative AI faces challenges in bias mitigation, interpretability, computational efficiency, and ethical governance, necessitating research into scalable architectures, explainability, and AI safety mechanisms. Integrating Reinforcement Learning (RL), multi-modal learning, and self-supervised techniques enhances controllability and adaptability in generative models. Additionally, as AI reshapes industrial automation, digital media, and scientific discovery, its societal and economic implications demand robust policy frameworks. This survey provides a comprehensive analysis of generative AI’s current state and future directions, highlighting innovations in efficient generative modelling, AI-driven scientific reasoning, adversarial robustness, and ethical deployment. By consolidating theoretical insights and real-world applications, it offers a structured foundation for researchers, industry professionals, and policymakers to navigate the evolving landscape of generative AI."

    server = Reviewer(args)
    server(key_words, query)