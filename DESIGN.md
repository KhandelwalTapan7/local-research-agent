# Design

## Agent pattern: Plan-and-Execute with per-step reflection

The agent first asks the model for an ordered plan of 3 to 6 steps, one tool per step. It then executes the steps one at a time. For each step the model chooses the tool arguments given the question, the plan, the sources collected so far and a progress log. After every call a quality check runs, and a poor result triggers a reflection call that diagnoses the failure and proposes a revised query, tool or piece of code. Retries are limited to two per step and stop early when the revised call equals one already tried.

Why not pure ReAct: a small model that re-decides the whole strategy each turn drifts, repeats searches and forgets the goal. A plan fixes the strategy, keeps each prompt short and makes the trace readable. Why not plan-only: retrieval quality and generated code are unpredictable, so each step needs feedback. The hybrid keeps adaptivity where it matters (after an observation) and structure everywhere else.

The plan is validated in code instead of trusted:
- unknown tools are dropped and retrieval is guaranteed;
- if the question asks to compute something, a `code_executor` step is added;
- `memory_store` is kept only when the user asks to remember something, and `memory_recall` only when earlier turns exist;
- if fewer than three distinct tools are planned, `summarise` and `structured_output` steps are appended.

Quality signals are programmatic wherever possible: the cross-encoder score for retrieval, exit code and output for code, schema checks for tables. The model is not asked to grade its own results. It only diagnoses and proposes a fix after the code has flagged a failure.

## Model and quantisation

The default is `qwen2.5:7b-instruct` at Ollama's Q4_K_M. It meets the 7B minimum, is good at structured JSON for its size and is Apache-2.0 licensed. Q4_K_M needs about 4.7 GB, which leaves room for an 8k context on a 12 to 24 GB GPU or in about 8 GB of RAM. Q8 would roughly double the memory for a small quality gain. All agent calls use Ollama's JSON mode with temperature 0.2, and parsing falls back to extracting the first JSON object with one retry.

The development machine has 8 GB of RAM, so the model was killed by the out-of-memory killer when loaded in Docker and the system was tested with `qwen2.5:3b-instruct`. The 7B default is configured but was not run there. The 3B model exposed the weaknesses below, which is useful because a fix that works for a weak controller helps a stronger one too.

## Biggest technical challenge

Making a small local model a dependable controller. In testing with the 3B model I saw four concrete failures:

1. The planner added pointless steps, such as saving a computed value to memory when nobody asked to remember anything.
2. The executor put tool calls inside generated Python (`memory_store(...)` inside `code_executor`), which raised a NameError because those functions do not exist there.
3. The executor ignored the planned tool and sent arguments meant for a different tool, causing a TypeError.
4. Reflection repeated the identical failing code, so the retries were wasted.

The fixes are layered. The planned tool is now authoritative: if the model picks another tool, it is re-asked once with an explicit instruction, and mismatched arguments are discarded. Memory steps are filtered by the user's wording and session state. Tool descriptions say that other tools do not exist inside code. The reflection prompt receives the list of failed calls and is told not to repeat them, and the loop stops if it does. Another failure, forgetting `print()`, is caught by the "no output" check and fixed by the reflection step, which is the self-correction visible in the demo.

Two other problems were deciding when retrieval is poor and keeping citations honest. Retrieval quality is judged by thresholding the cross-encoder score instead of asking the model. Sources are registered only from relevant retrievals, referenced by short ids (S1, S2) and checked after generation: ids that do not exist are removed, the answer is regenerated once if no valid citation remains, and only cited sources are returned.

## Other decisions

- Retrieval: BM25 catches exact terms such as HNSW or pgvector, dense vectors catch paraphrases, RRF fuses the two without score calibration, and the cross-encoder reranks the top ten.
- Persistence: SQLite with runs, events and facts tables. Every streamed event is stored, so `/history` returns the exact trace. Recent turns are passed to the planner and the final answer for follow-up questions.
- Streaming: the agent is a Python generator, so the same events feed SSE, SQLite and the UI with no queue.
- Known inconsistency: generated code may divide by 1024^3 while the documents use 10^9 bytes per GB, so a computed value and the cited figure can differ slightly. Both are correct in their own units.

## What I would improve with more time

- Run the demo and the evaluation on the 7B model on suitable hardware and tune the plan and reflection prompts for it.
- Evaluate retrieval and answers on a labelled question set, and tune the rerank threshold from data instead of a fixed default.
- Run independent plan steps in parallel and let the planner re-plan when a step fails repeatedly.
- Replace the subprocess code runner with a container or gVisor sandbox with memory and CPU limits.
- Check each cited claim against its source with an entailment model, not just id validity.
- Stream tokens of the final answer, add authentication and rate limiting, and move to a real vector store once the corpus outgrows in-memory numpy.