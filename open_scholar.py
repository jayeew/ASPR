import argparse
import requests
import json
import os
from openai import OpenAI

class Reviewer:
    def __init__(self, args):
        self.args = args
        self.client_large = None
        self.open_scholar = None

        self.initialize_models()

    def initialize_models(self,):
        self.client_large = OpenAI(
            api_key="",
            base_url=f'http://localhost:{self.args.large_model_port}/v1',
        )
        self.open_scholar = OpenScholar(
            args=self.args
        )

    def __call__(self, key_words):
        if os.path.exists("papers.json"):
            with open("papers.json", "r") as file:
                papers = [json.loads(line.strip()) for line in file if line.strip()]
        else:
            papers = self.open_scholar.search_semantic_scholar(key_words)
            with open("papers.json", "w") as file:
                for paper in papers:
                    print(json.dumps(paper), file=file)
    
    def _generate_review(self, papers):
        # rank papers
        
        references = ""
        for idx, paper in enumerate(papers[:self.args.top_n]):
            references += f'[{idx}] Title:{paper["title"]} Abstract:{paper["abstract"]}\n'
        
        


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
                "year":paper["year"],
                "title":paper["title"],
                "authors":paper["authors"],
                "venue":paper["venue"],
                "citationCount":paper["citationCount"],
                "url":paper["url"],
                "abstract":paper["abstract"]
            })

        return formatted_papers

    def _search_paper_via_query(self, query):
        if self.and_search:
            query = ' + '.join([f'"{kw}"' for kw in query])
        else:
            query = ' | '.join([f'"{kw}"' for kw in query])
        query_params = {
            'query': query,
            'fields': "title,year,authors.name,abstract,venue,citationCount,url,externalIds",
            "year": "2023-"
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
    parser.add_argument('--large_model_model', type=str, default='OpenSciLM/Llama-3.1_OpenScholar-8B',
                        help='Large model name')
    parser.add_argument('--large_model_port', type=int, default=38011,
                        help='Port for large model server')
    parser.add_argument('--small_model_model', type=str, default='Qwen/Qwen3-0.6B',
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

    server = Reviewer(args)
    server(key_words)