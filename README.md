# phost-agent

An agent layer for [phost](https://github.com/cdevingo-bd/phost): what an agent is told about a stage, what a proposal must
pass before it earns a timed run, what the search remembers, who may use the device, and what it may spend. **It calls no
model.** A proposer is anything that returns the next candidate: a script, a baseline search, or a program that wraps your own
agent and keeps its own keys.

```
phost-agent card effective.solve                                   # what a stage is, as Markdown (or --json)
phost-agent ceiling --scene chain256:ringing --stage effective.solve   # tier 0: is there anything to gain? (measures, then remembers)
phost-agent brief --scene chain256:ringing --stage effective.solve     # the brief an agent is given
phost-agent check proposal.json --scene chain256:ringing --stage effective.solve   # one candidate through tiers 1-3
phost-agent run --scene chain256:ringing --stage effective.solve \
    --plugin phost_author.expression_impls --proposer coordinate --attempts 8
phost-agent status                                                 # funnel, memo hit rate, spend, whether the record is intact
```

It needs phost, phost-lab and phost-score; phost-author (for the sandbox, and typed expressions as candidates) is optional. None
is on PyPI yet, so install from clones (see the phost README), then `pip install -e phost-agent`.

## The funnel

A timed run is the expensive thing: it needs the device to itself, and a candidate that is wrong or hopeless wastes it. So a
proposal goes through cheaper tiers first and stops at the first failure.

| tier | what it asks | what it costs |
| --- | --- | --- |
| 0 ceiling | could *any* replacement gain enough? The lab times the host's own version of each stage as a share of the step; a stage that is 2% of it has a ceiling of 1.02x. Asked once per scene, before anything is proposed | one measurement, remembered |
| 1 static | a real stage and implementation, parameters valid against their specs, or source that parses and defines the class and the entry method the stage calls | no process |
| 2 cheap check | the stage's contract check on a simulator-free host (phost's fake host, else the NumPy chain host) | CPU only |
| 3 real check | the same check on the scene's own host | the device, under the lease, if the scene is on a GPU |
| 4 timed run | the lab's run against the native control, exclusive on the device, judged by phost-score | the device, under the lease |
| 5 ledger | the entries are appended to the score ledger and the result to the journal | none |

A stage that changes the physics or only observes has no ceiling: a speed search refuses it (tier 0) and `--goal behaviour` passes
it, because the loop is then after a candidate that passes its checks and not one that is fast. A stage the lab has no native timing
for is passed with its ceiling marked unknown.

What counts as a win (`speed` goal): the contract checks and the accuracy gate pass, the timing is reliable (the two native
controls agree, no other process used the GPU), and the speedup beats `--min-speedup` (default 1) at **every** world count: the
least over the world counts is what is compared. A correct candidate that is not faster is `measured`, and is still recorded:
a measurement is a fact whether or not it won.

## Briefs and cards

`card` and `brief` are generated from the registries, so they cannot go stale. A card is one stage: its ports with shapes and
meanings, the invariants, how a swap is judged (its swap class), the metrics and gates, the implementations that exist with their
parameters, and the typed form the law takes. A brief adds the task: the scene and engine, the ceiling, what a win is, the funnel,
the budget, the shape of an answer and a skeleton of a runtime class for the stage. Both are plain data (`--json`) and render as
Markdown for a person or a model.

## Proposers

```python
class Proposer(Protocol):
    name: str
    def propose(self, brief, history) -> Proposal | None: ...      # None: nothing left to try
```

A `Proposal` is `{"stage", "impl", "params"}` for a registered implementation (a typed expression is the registered
`<stage>/expression` with the text as parameters) or `{"stage", "source", "entry"}` for a runtime class, with an optional `note`
and the `tokens` it cost to produce.

* `ScriptedProposer(list)` replays a list: a recorded session, a test, a hand-written sweep.
* `CoordinateProposer` is the baseline with no learning: the defaults of every implementation of the stage, then one parameter at
  a time around the best result so far (a number at zero is not scaled: there is no scale to scale it by). Anything smarter
  should beat it. Many implementations have defaults at zero, so start those from a script.
* `CommandProposer("program args")` runs any program: it gets `{"brief", "history"}` as JSON on stdin and prints
  `{"proposal": {...}, "tokens": 1200}` or `{"stop": "why"}`. An agent that calls a model wraps itself in one of these. A program
  that fails, times out or prints something else costs an attempt and the loop goes on.

## The memo

Every tier's result is kept in the journal under a content key: the stage, the implementation with its *resolved* parameters (a
default spelled out is the same candidate as the default left out) or the hash of the source text, the scene, the engine, the world
counts, the driver and the phost version. The agent's note and token count are not part of what a candidate is. A candidate seen
before costs nothing: a failure is reported again with its reason, a pass skips the tiers it already passed, a timed result is
reused. A timeout, a lost worker, a timing flagged unreliable, or an accuracy ratio over its gate on an engine whose native control is itself
noisy (a GPU: the ratio is against that noise and moves between runs) says something about the run and not the candidate, so it
is not remembered; an engine that repeats exactly has no such excuse. The memo is a view of the journal, so it survives restarts, and it counts as an attempt when it answers (an agent that
repeats itself is stalling). `--refresh` ignores it.

## The device lease

A timed run that shares the device with another timed run measures both. `Lease` is an advisory `flock` on one file per device
(`cuda:0`, `cpu`) under the agent's state directory, with the holder written beside it. Every ceiling measurement, every timed
run and every real check on a GPU scene holds it. Because the kernel holds the lock, a holder that dies releases it and there is no
stale lease to clear. It serialises everything that uses this layer and the same state directory; it does not stop a process
outside it (a colleague's job on the same GPU), which is why the lab's `other_gpu_processes` still marks such a run unreliable.
`phost-agent lease --device cuda:0` says who holds it.

## Budgets

```
--attempts 20  --gpu-seconds 600  --cpu-seconds N  --tokens N  --stall 5  --workers 1
```

`attempts` counts proposals (including memo answers); `gpu-seconds` and `cpu-seconds` count the time the lease is held for timed
runs and ceiling measurements plus the time in check workers, on a GPU or a CPU engine respectively; `tokens` counts what the
proposers report; `stall` stops after that many attempts in a row without a new best. The first to run out ends the loop and the
report says which.

**One worker by default.** `--workers N` checks the CPU tiers of N candidates at once and needs `--cpu-seconds`, so a wider search
is always a budgeted one. Only an engine that declares itself `cpu` is checked in parallel (the chain, the JAX chain on a CPU
build); on a GPU scene the real check is serialised under the lease. Timed runs are never concurrent. The N proposals of a round
are asked for before any of them is answered, so the proposer does not see their outcomes until the round ends.

## What is trusted, and what is not

* A registered implementation is phost's own code: its checks run in a worker process, which is for isolation of faults.
* A **source** proposal is code nobody reviewed. Tiers 2 and 3 run it under `phost_author.sandbox` (`bwrap`: read-only filesystem,
  no network, a private home, CPU and file-size limits) when phost-author and `bwrap` are present, and are refused otherwise unless
  you pass `--allow-unsandboxed`.
* **Tier 4 is not sandboxed.** The timed run is a lab worker: a separate process with a timeout, running with your permissions. A
  source candidate has passed the sandboxed checks by then, which is evidence about its behaviour on those inputs and not a proof
  about anything else. Do not point an agent that writes source at a machine whose files or network you care about.
* Entries written to the score ledger are `self-reported`: measured here, by this process. Nothing has checked them independently.
  The journal and the ledger are hash chains: they catch accidents and careless edits, not someone who can rewrite the whole file.
* The layer makes no network calls and neither should a candidate. It never puts your name, paths or scene contents in a request,
  because it makes none.

## The record

`journal.jsonl` (under `$PHOST_HOME/agent`, else beside the score ledger) is one hash-chained record per ceiling, tier result,
attempt and finished search. `phost-agent status` reads it: the funnel from proposed to accepted, the memo's hit rate, spend, who
holds each device, and whether the chain verifies (exit 3 if it does not). [phost-dash](https://github.com/cdevingo-bd/phost-dash)
has a Research tab that shows the same, read-only.

## Measured, and not

One run each, on one machine. On the NumPy chain (`chain256:ringing`, effective solve 92% of the step: a ceiling of about 12x),
the typed conjugate-gradient solver of phost-author at 20 iterations ran at 0.22x of native, and at 5 iterations 0.41x; at 2
iterations it failed its residual gate at the cheap tier and never earned a run. The coordinate baseline found no win there, which
is the honest answer: a NumPy banded solve is hard to beat from NumPy.

On MuJoCo Warp (`trampoline8:impact`, 8 worlds, an RTX PRO 2000 laptop GPU, `newton.linear_solve`, three attempts, 25.6 s of device
time): the lab measured the stage at 42% of the step, a ceiling of 1.74x. The coupled-block Cholesky at its defaults ran at 0.95x
and with `tile_size=16` at 0.99x, neither a win. Its structural variant at its defaults was stopped by the real check on the
scene's host (residual 0.075 against a gate of 1e-4) and never earned a run; that implementation's own tests set
`verify_structure`, which a proposer starting from the defaults does not know to do. Three attempts is too few for the
baseline to have tried it.

Not done: no MJX engine (there is a JAX chain, `jaxchain`, and that is not MuJoCo); no model is called by this package, so how well
an LLM agent does with these briefs is not something it can tell you; fan-out was exercised on the CPU engines only, and no source candidate has been timed on a GPU scene.

## Tests

```
pip install -e '.[test]'
pytest                    # ~170 tests; the real-simulator ones use the NumPy chain engine and need no GPU
```

`tests/test_jax_scene.py` runs the layer on the JAX engine's scene and is skipped without `jax` (the lab's own JAX tests live in a
separate environment for the same reason).

MIT licensed.
