
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_PROMPT_PATH = PROJECT_ROOT / "outputs" / "temp.json"

if __name__ == '__main__':
    client_large = OpenAI(
            api_key="",
            base_url=f'http://localhost:{38011}/v1',
    )

    with TEMP_PROMPT_PATH.open("r", encoding="utf-8") as file:
        input_query = file.read()
    # print(input_query)
    response = client_large.chat.completions.create(
        model='OpenSciLM/Llama-3.1_OpenScholar-8B',
        messages=[{"role":"user", "content":input_query}],
        temperature=0.7,
        max_tokens=3000,
        stream=False,
        timeout=300
    )
    content = response.choices[0].message.content
    print(content)
