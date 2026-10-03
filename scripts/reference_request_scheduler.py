"""Bounded continuous batching for the vLLM 0.4 LLMEngine API.

One engine is driven by one thread. Each job has its own SamplingParams.
No vLLM import is needed here, so scheduler routing can be tested on CPU.
"""

import inspect


def run_requests(llm, jobs, max_pending, backend="engine"):
    """Yield (job, final_request_output), potentially out of input order."""
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    if backend == "serial":
        for job in jobs:
            outputs = llm.generate(prompt_token_ids=[job["prompt_token_ids"]],
                                   sampling_params=job["params"], use_tqdm=False)
            if len(outputs) != 1:
                raise ValueError("Expected exactly one request output")
            yield job, outputs[0]
        return
    if backend != "engine":
        raise ValueError(f"Unknown backend: {backend}")

    engine = llm.llm_engine
    # Check the pinned legacy API before queuing any jobs. Newer engine APIs
    # may accept a different prompt schema; never silently retokenize.
    inspect.signature(engine.add_request).bind(
        "reference_api_probe", None, None, prompt_token_ids=[1],
    )
    if engine.has_unfinished_requests():
        raise ValueError("Reference scheduler requires an idle, exclusively owned engine")

    iterator = iter(jobs)
    pending = {}
    exhausted = False
    sequence = 0
    try:
        while pending or not exhausted:
            while len(pending) < max_pending and not exhausted:
                try:
                    job = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                request_id = f"reference_{sequence}"
                sequence += 1
                # Use positional request_id/prompt/SamplingParams like the
                # 0.4 offline wrapper; forks differ in the third argument name.
                engine.add_request(request_id, None, job["params"],
                                   prompt_token_ids=job["prompt_token_ids"])
                pending[request_id] = job
            if not pending:
                break
            for output in engine.step():
                if output.request_id not in pending:
                    raise ValueError(f"Unknown or repeated request ID: {output.request_id}")
                if not output.finished:
                    continue
                job = pending.pop(output.request_id)
                yield job, output
            if pending and not engine.has_unfinished_requests():
                raise RuntimeError("Engine lost pending reference requests")
    finally:
        # Also executes when scoring/writing fails and the consumer closes us.
        for request_id in pending:
            engine.abort_request(request_id)
