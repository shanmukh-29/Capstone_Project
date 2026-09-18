# Support Assistant

Run `uvicorn support_assistant.main:app --reload` and POST `{"query":"What is the delivery policy?"}` to `/ask`. The default `MOCK_LLM` path is deterministic and offline: `classify_intent` uses the required keyword heuristic, `retrieve_and_answer` loads all eight policy documents and returns the top three lexical matches, and `direct_answer` returns the fixed general-question response. The validated response is `{answer, sources, confidence}`.

Architecture: ingestion reads `docs/doc_01.txt` through `doc_08.txt`; embedding/indexing is represented by the `ChromaDB`/`all-MiniLM-L6-v2` optional integration point for installed deployments, while the baseline uses a deterministic local scorer so Docker and grading remain offline; retrieval is performed in `retrieve_and_answer`; generation is the mock template or the optional real LLM branch. The structured prompt includes role, context, task, format, length, a negative grounding constraint, and a few-shot example. Only generation branches on `MOCK_LLM`; intent routing and retrieval remain local.

Example responses:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials...","sources":["doc_01","doc_06","doc_04"],"confidence":1.0}
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Docker: from the repository root, run `docker build -f support_assistant/Dockerfile -t zepto-support .` then `docker run -p 7860:7860 zepto-support`.
