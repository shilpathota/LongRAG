# LongRAG Implementation

## Steps to reproduce
### Running model is local 
1. Created the Docker Compose file to pull the Ollama image to run on CPU
```
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    volumes:
      - ollama:/root/.ollama
    ports:
      - "11434:11434"
    restart: unless-stopped
volumes:
  ollama:

```
This pulled the Ollama Docker image (the runtime: binaries, APIs, dependencies).
Size ≈ 1 GB.
Stored once in Docker’s image cache (not large and reused for any Ollama container).

2. Execute the following commands to pull the image and run on docker
```
docker compose -f docker-compose.ollama.yml up -d
docker exec -it ollama ollama pull llama3.1:8b     # 128k context model
# quick test
curl http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","prompt":"Say hi"}'
```
The exec command is run inside the container. It downloads the actual model weights for ollama model registry  into the mounted volume. These weights are separate files. 
Total would be approx - 6 GB of space is required.

3. Now that the model is runnning in local. 

### Ingestion Pipeline
4. For Preprocessing of the raw text, I'm using Hierarchial chunking strategy where the parent is chunked in 2048 bytes which is static length but I want to use hybrid strategy where Child is chunked based on sematic meaning and not the static length
<table><thead>
<tr><td>Goal</td>	<td>Technique</td></tr></thead><tbody>
<tr><td>Preserve document structure (chapter → section → paragraph)</td>	<td>Hierarchical chunking</td></tr>
<tr><td>Split large sections precisely where topic changes</td>	<td>Semantic splitting</td></tr>
<tr><td>Avoid cutting in the middle of sentences or semantic units</td>	<td>Semantic splitting</td></tr>
<tr><td>Maintain parent–child traceability</td>	<td>Hierarchical chunking</td></tr></tbody>
</table>

api/ingest.py gives the steps for preprocess.
5. Now we go to `build_index.py`  which supports:
- Load parents.jsonl (from data/processed/parents/).
- Generate embeddings using a SentenceTransformer (e.g. intfloat/e5-base-v2 or BAAI/bge-base-en).
- Store them in FAISS for fast vector search.
- Save metadata (mapping parent → children, source file, etc.).

6. That index will then be used by your retrieval + generation stage (serve.py) to:
- Retrieve top-k parents,
- Expand to top children,
- Send the combined context to your Ollama model.

## Running the model
7. Run the `app/ingest.py` so the files are created in the data/processed folder as parent and child and ready to get embedded
8. Now run `app/build_index.py` so the files created in above step are embedded and metadata is created and together stored in FAISS DB
9. Now we can run the command `uvicorn app.serve:app --host 0.0.0.0 --port 9000` to serve the APIs and it is ready for queries from the user

## Testing the model
 We can run the command in Powershell to test the application 

 ```
 Invoke-RestMethod -Uri "http://localhost:9000/query" `
>>   -Method Post `
>>   -ContentType "application/json" `
>>   -Body '{"question": "What are the incident reporting requirements in the policy?"}'
```
 The question refers to the context in the policy.txt that was fed to the LLM

 ## What makes this LongRAG?
 ### ingest.py
- Performs semantic splitting using SemanticSplitterNodeParser (LlamaIndex).
- Produces “children” chunks (~512 tokens) — these are semantically coherent, not uniform fixed-size.
- Then packs them into “parents” (~2048 tokens) preserving document order and hierarchy.
This is hierarchial Chunking which is key component of LongRAG. The hierarchical relationship allows you to retrieve parent blocks first, then drill down into semantically aligned children, enabling more efficient recall over long documents.

### build_index.py
Saves both FAISS index and a JSON meta.json that maps FAISS rows → parent IDs.This ensures efficient ANN retrieval at the parent level, not child level. You’re doing coarse-to-fine retrieval:
Parent-level retrieval gives recall efficiency; child-level refinement gives precision — exactly how LongRAG optimizes retrieval for long contexts.

### serve.py
This is where we can see LongRAG behavior
- Retrieves top-K parents from FAISS.
- Applies cross-encoder reranking (BAAI/bge-reranker-base) for more accurate ordering.
- For each parent, selects top semantic children using embedding cosine similarity.
- Dynamically assembles a multi-level context (parents + best children) within a token budget.
- Sends this full context to Ollama (local LLM).



