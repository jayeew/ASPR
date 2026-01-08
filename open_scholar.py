import argparse
from openai import OpenAI

class Reviewer:
    def __init__(self, args):
        self.args = args
        self.client_large = None

        self.initialize_models()

    def initialize_models(self,):
        self.client_large = OpenAI(
            base_url=f'http://localhost:{self.args.large_model_port}/v1',
        )
        self.open_scholar = OpenScholar(

        )

    def __call__(self, key_words):
        pass

class OpenScholar:
    def __init__(self, args):
        self.args = args

    def __call__(self,):
        pass

    def search_semantic_scholar(self, key_words):
        pass

    def _search_paper_via_query(self, query, and_serch):
        if and_serch:
            query = ' + '.join([f'"{kw}"' for kw in query])
        else:
            query = ' | '.join([f'"{kw}"' for kw in query])
        query_params = {
            'query': query,
            'fields': "title,year,authors.name,abstract,venue,citationCount,url,externalIds"
        }

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

    key_words = []

    server = Reviewer(args)
    server(key_words)