from openai import OpenAI

openai_client = OpenAI(
    base_url='http://127.0.0.1:11434/v1',
    api_key='1234',
)

def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="nomic-embed-text")
        .data[0]
        .embedding
    )


# ollama.embeddings(model='nomic-embed-text', prompt='The sky is blue because of rayleigh scattering')

test_embedding = emb_text("The sky is blue because of rayleigh scattering")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])


from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker

converter = DocumentConverter()
chunker = HierarchicalChunker()

# Convert the input file to Docling Document
source = "https://milvus.io/docs/overview.md"
doc = converter.convert(source).document

# Perform hierarchical chunking
texts = [chunk.text for chunk in chunker.chunk(doc)]

for i, text in enumerate(texts[:5]):
    print(f"Chunk {i+1}:\n{text}\n{'-'*50}")

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
)

from tqdm import tqdm

data = []

for i, chunk in enumerate(tqdm(texts, desc="Processing chunks")):
    embedding = emb_text(chunk)
    data.append({"id": i, "vector": embedding, "text": chunk})

milvus_client.insert(collection_name=collection_name, data=data)

question = (
    "What are the three deployment modes of Milvus, and what are their differences?"
)

search_res = milvus_client.search(
    collection_name=collection_name,
    data=[emb_text(question)],
    limit=3,
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["text"],
)

import json

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

response = openai_client.chat.completions.create(
    model="gemma3:4b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)

