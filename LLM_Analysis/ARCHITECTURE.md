# contract_ai — Architecture & File Guide

For the engineer wiring OCR → RAG → this package into the pipeline.
You should only ever need to import from `contract_ai/pipeline.py` — everything
else is internal to how the engine works.

## Data flow

```
OCR module          RAG module              contract_ai
-----------          -----------              -----------
extract(file)  --->  retrieve(ocr_result) --->  ContractPipeline.analyze_contract(
  {metadata,           str | list[{text}]         metadata, clauses, rag_context
   clauses}                                      ) -> {summary, clauses_analysis}
```

Nothing in this package reads OCR/RAG output from disk. Whatever those two
components produce in memory gets passed straight in as function arguments.

---

## File-by-file

### `pipeline.py` — the only file the rest of the system should import from

Exposes `ContractPipeline`, the facade class with two methods:

- `analyze_contract(metadata, clauses, rag_context)` — async. Runs the
  clause-by-clause risk review. `metadata` is a dict (e.g. contract type,
  parties), `clauses` is a list of dicts (`clause_id`, `clause_label`,
  `clause_text` — this is what OCR is expected to produce). Returns
  `{"summary": OverallSummarySchema, "clauses_analysis": [ClauseRiskAnalysis, ...]}`.
- `draft_contract(user_input, rag_context)` — sync. Takes user-supplied
  contract requirements (dict or `ContractUserInput`) and returns a
  `FullContractDocument`.
- `analyze_contract_sync(...)` — same as `analyze_contract` but wraps it in
  `asyncio.run()`, for callers that aren't already inside an event loop
  (e.g. a plain script or a sync web framework).

Also contains `_rag_context_to_text()`, which normalizes whatever shape the
RAG component returns (a raw string, a list of `{"text": ...}` chunks, or a
dict with `"context_text"`) into the plain text block the prompts expect.
**This is the main integration seam with the RAG teammate** — if their
output shape doesn't match one of these three, extend this function rather
than changing anything downstream.

### `schemas.py` — the data contracts

Pydantic models defining exactly what goes in and comes out. These aren't
just internal types — they're the interface. Two families:

- **Analysis**: `ClauseRiskAnalysis` (per-clause verdict: risk level 🔴🟡🟢,
  score 1–10, legal reasoning, cited law articles, suggested rewrite) and
  `OverallSummarySchema` (whole-contract executive summary).
- **Drafting**: `ContractUserInput` (what the user must supply to generate a
  contract) and `FullContractDocument` (the generated contract: preamble,
  list of `ContractClauseDraft`, closing).

If the pipeline engineer needs to know "what fields will I get back," this
is the file to read — it's the source of truth, not the README.

### `analysis.py` — risk analysis engine

`run_parallel_analysis(metadata, clauses, rag_context_text)` is the entry
point pipeline.py calls. It:
1. Fires off one LLM call per clause concurrently, capped at
   `settings.max_concurrent_clauses` in-flight at once (via
   `asyncio.Semaphore` + `asyncio.to_thread`, since the underlying `ollama`
   client is synchronous).
2. Once all clauses are analyzed, makes one more LLM call to produce the
   overall contract-level summary from the combined results.

`analyze_single_clause()` is the synchronous per-clause worker — useful to
call directly if the engineer ever wants to test/debug one clause without
running the full async fan-out.

### `drafting.py` — contract generation engine

One function, `generate_balanced_contract(user_input, rag_legal_context)`.
Builds the prompt from `ContractUserInput`, calls the LLM once, returns a
validated `FullContractDocument`. No concurrency here — it's a single call.

### `llm_client.py` — the only file that talks to Ollama

`call_structured(...)` is the single choke point for every LLM call in the
package (both analysis.py and drafting.py go through it). It:
- Sends the system + user prompt to `ollama.chat()`
- Forces structured JSON output via the Pydantic schema's JSON schema
- Retries on failure (`retries` param, defaults from `config.py`)
- Raises `LLMCallError` (with `.context` and `.cause`) if every attempt fails
  — this is the exception type to catch upstream if you want to handle LLM
  failures gracefully (e.g. mark a clause "analysis failed" instead of
  crashing the whole request)

`check_model_available(model_name)` is a health-check — call it once at
service startup (not per-request) to confirm Ollama is reachable and the
model is loaded.

### `prompts.py` — the actual instructions sent to the model

`ANALYSIS_SYSTEM_PROMPT` and `DRAFTING_SYSTEM_PROMPT`, in Arabic, matching
the domain (Egyptian law). Kept in their own file so legal/prompt tuning
doesn't require touching any logic code.

### `config.py` — all tunables, env-var driven

A frozen `Settings` dataclass, instantiated once as `settings`. Nothing in
this package hardcodes a model name, temperature, retry count, or
concurrency limit — they all read from environment variables at import time
(see README for the full list: `OLLAMA_HOST`, `CONTRACT_AI_ANALYSIS_MODEL`,
etc.). This is what lets the pipeline engineer point staging vs. production
at different Ollama hosts/models without touching code.

### `io_utils.py` — NOT used by the pipeline itself

File-based loaders (`load_ocr_output`, `load_rag_context`) and a
`save_json` helper, plus mock sample data. These exist purely for local
testing via `scripts/run_analysis_cli.py` — they let you run the engine
against a JSON file on disk instead of wiring up real OCR/RAG. **The
pipeline engineer's integration code should never need this file** — real
OCR/RAG output goes straight into `pipeline.py`'s methods as Python objects.

### `scripts/run_analysis_cli.py` — standalone test runner

Not imported by anything — this is a script you run directly
(`python scripts/run_analysis_cli.py analyze`) to sanity-check the engine
end-to-end against mock or file-based data, without needing the rest of the
pipeline built yet. Useful for the pipeline engineer to confirm their
Ollama setup works before wiring in real OCR/RAG.

### `__init__.py` — public API surface

Re-exports `ContractPipeline` and all the schemas at the package level, so
the pipeline engineer writes `from contract_ai import ContractPipeline` and
never needs to know the internal module layout.

---

## What the pipeline engineer needs to hand this package

1. `metadata: dict` and `clauses: list[dict]` from OCR — each clause dict
   needs `clause_id`, `clause_label`, `clause_text` at minimum (see
   `analysis.py::_build_clause_prompt` for exactly what's read).
2. `rag_context` from the RAG component — any of: a string, a list of
   `{"text": ...}` dicts, or a dict with `"context_text"`.
3. A running Ollama instance reachable at `OLLAMA_HOST`, with
   `qwen2.5:14b` and `qwen2.5:7b` (or whatever's set in
   `CONTRACT_AI_ANALYSIS_MODEL` / `CONTRACT_AI_DRAFTING_MODEL`) pulled.

That's the entire integration surface.
