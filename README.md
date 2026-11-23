# phost-agent

An agent layer for [phost](https://github.com/cdevingo-bd/phost). Its centre is an **arena**: language models, behind any provider
(the Claude Code on your machine with no API key, the Claude API, OpenAI or anything that speaks its protocol, a local model, a program of
your own), compete on a leaderboard for the fastest
simulation step on a scene. Nobody tells them what to try. Each is given the same facts (where the step spends its time, what every
stage is, how entries are judged, the live board), then takes turns: look at a stage, read a rival's entry, check a graph, enter one,
and go on until the budget is spent. A graph can replace any stages, with registered implementations or with code the model writes. The arena only keeps the rules
(contract checks, timing against the host's own step with the device to itself, budgets) and the board, which is `phost-score`'s.

Around it are the pieces that make that safe to run: a tiered funnel that keeps a wrong or hopeless entry from earning a timed run, a
content-addressed memo, a per-device lease, and budgets. The same machinery also runs a **focused search** on one stage, driven by a
baseline, a program or a model.

```
phost-agent llms                                                   # the model providers there are, and what each needs
phost-agent arena --scene chain256:ringing \
    --contestant opus=claude-code:opus --contestant sonnet=claude-code:sonnet \
    --turns 20 --submits 4 --tokens 200000                         # models compete; the board is printed at the end
phost-agent prompt --scene chain256:ringing --stage effective.solve   # what a model is sent first in a focused search (sends nothing)
phost-agent run --scene chain256:ringing --stage effective.solve \
    --attempts 12 --proposer llm:anthropic                         # a focused search on one stage, driven by a model
phost-agent card effective.solve                                   # what a stage is, as Markdown (or --json)
phost-agent ceiling --scene chain256:ringing --stage effective.solve   # tier 0: is there anything to gain? (measures, then remembers)
phost-agent check proposal.json --scene chain256:ringing --stage effective.solve   # one candidate through tiers 1-3
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
  `{"proposal": {...}, "tokens": 1200}` or `{"stop": "why"}`. An agent of your own wraps itself in one of these. A program
  that fails, times out or prints something else costs an attempt and the loop goes on.
* `LLMProposer(client)` is a language model (next section).

## The arena

```
phost-agent arena --scene <scene> --contestant [NAME=]PROVIDER[:MODEL] ... [--turns 30] [--submits 6] [--tokens N] [--keep-turns 6] [--closed] [--dash]
```

`--dash` also starts phost-dash on this machine's loopback over the same state (`$PHOST_HOME`), prints its link before the first turn
and keeps it up after the contest until Ctrl-C. Without it, the dashboard is a separate process: `phost-dash --state <the state dir>`.

Contestants take turns in a fixed rotation, one action each, so each sees what the others did before it moves and the one device is
never shared by two timed runs. A turn is a model call. The scarce thing is the timed run (`--submits`, per contestant); checks cost
a turn and nothing else. Entries go to the same ledger as everything else in `phost-score`, labelled with the contestant's name, so
`phost-score board` and the dashboard's Results screen rank them with every other run.

**What a contestant is told:** the scene; where the host's own step spends its time (a measurement, with the most each stage could
give); what every stage is and which implementations exist; the rules of judging; the actions; its budget; the board. **What it is not
told:** what to try, in what order, or how to search. There is no coaching in the prompt (`tests/test_arena.py` checks that).

**What it can do:** `look` at a stage's card or at any entry on the board (a rival's: graph, parameters, source, while the contest is open, unless `--closed`;
one recorded outside the contest, such as a person's or an earlier run's: graph and parameters, since the ledger does not keep source);
`check` a graph (the contract checks only, with an optional `note` that goes in its notebook); `submit` one (check, time, rank, and be
told where it landed). There is no way to leave: a
contestant plays until its turns, its timed runs or its tokens are spent, or until a failure that cannot be retried (no key, an unknown
model) removes it. In an open contest every `check` anyone runs is posted, passed or failed, with who ran it and the reason it failed: each
turn shows the others' latest ten (`--feed N`, 0 for none) as one line each, and `look` with `"what": "attempt"` and an id gives the
source and the whole reason. A graph checked before, by anyone, is answered with who checked it and what came of it. `--closed` posts
and shows nothing; the checks are still in the journal. Whatever a runtime `print`s while a check runs comes back with the result,
passed or failed (the first 4000 characters, identical lines in a row folded), and is posted with the check; see below. A graph replaces any
set of stages, each with a registered implementation and parameters or with the source of a runtime class. A second model that builds
on the first's entry is a legitimate move, and is why the contest is open by default.

**What a turn sends.** The prompt is built from parts that each have one job (`context.py`), because sending the whole conversation again
every turn is what made a contest's cost grow with the square of its length:

- *Pinned*, at the top: the card (with the skeleton of a runtime) of each stage whose swap is ranked on speed, up to three, and any card
  the contestant looks at; each once. Under a card, what a check hands a runtime on this scene, measured once by a probe the arena
  writes from the card (`facts.py`): shapes, dtypes, layout, the host's facts, which libraries are installed in the sandbox. It runs
  through the same worker and sandbox as a contestant's code, its text is kept in the journal (kind `facts`), and a later contest on the
  same scene reads it instead. `--no-facts` skips it. It is information about the inputs and says nothing about what to do with them.
- *Window*: the last `--keep-turns` turns (default 6) word for word, trimmed back in blocks of 3 rather than one a turn, so the start of
  the prompt stays the same for several calls in a row where a provider caches it. Each past turn holds only what was said and the
  result; the board and the feed are not copied into it.
- *Notebook*, from the arena's own records and not from a model: one line for each earlier check and timed run that is no longer in the
  window, with its id, what it replaced (and what it was edited from), how it came out, the contestant's own `note` and the first line it
  printed; its best entry; what it has read. Nothing is lost by it: any id can be looked at or built on.
- *State*, in the newest message only: what is new on the board (a row not yet shown, and where it stands) and in the others' checks
  since the contestant last looked. `look` with `"what": "board"` gives the whole board.

**Building on an earlier source.** A graph node can be `{"base": "<attempt or entry id>"}`: the source of that attempt or entry (yours,
or anyone's while the contest is open), run as it was or edited by an `edit` block of SEARCH/REPLACE hunks (`edits.py`; each SEARCH
must match one place, and a refusal says where the nearest text is). A fork of a rival's entry costs the edit, not the reading and the
retyping. The result of a check names its attempt id and the result of a timed run its entry ids.

**Several actions in a turn.** `{"actions": [...]}` holds up to 6 look, check and submit actions, run in order, each answered under its
own header; a failing one does not stop the others. A turn is one call to the model and counts once against the turns however many
actions it holds, so checking three variants costs one prompt, not three. With several actions a stage's python and edit blocks are used
in the order the actions come.

**What a turn cost.** Each `turn` record in the journal has the input, output and cached tokens the provider reported, whether they were
estimated, and the size in characters of each part of what was sent (`sent`: system, pinned, notebook, state, and the messages kept).
The end of a run prints the split per contestant.

Measured on a scripted contest, the same moves through the previous commit and this one (two contestants, 5 000-character sources,
`tests`-style fakes for the check and the timed run; characters over four, so the numbers are indicative): 30 turns, 0.68-0.72M tokens
before, 0.29-0.30M after (`--keep-turns 6`), 0.52-0.55M with nothing trimmed (`--keep-turns 60`, which isolates pinning and the
delta state); 60 turns, 2.6-2.7M before, 0.63-0.66M after; the last turn's prompt 70k before, 12k after. A scripted contest says
nothing about what real models do with a shorter window, so **whether it costs them any speedup is not measured**. Compare the best
entry at equal turns with `--keep-turns 6` against `--keep-turns 60`.

**What ranks:** the board is `phost-score`'s: entries are compared only with entries on the same scene, device, world count and swap
class, by the class's objectives (speedup for an exact swap; speed and accuracy together, as a Pareto front, for an approximate
one; a model-changing entry only against entries replacing the same stages). The board a contestant sees is for the largest world
count. Timing flagged unreliable is not ranked. Only the latest run of the same graph under a name counts, so repeating an entry
re-measures it instead of letting noise win.

**What it costs, and what leaves the machine:** as in the focused search below. Every token sent and received counts against
`--tokens`, per contestant; a failed call costs a turn; a failure that would repeat (no key, an unknown model) removes that
contestant and the others play on. Only the prompts go out, to the providers you named; the home directory, user name and state
directory are scrubbed from everything sent, each time. Tier 4 (the timed run) is not sandboxed: a source entry has passed the
sandboxed checks, which is evidence about its behaviour on those inputs and not a proof about anything else, so do not enter a model
that writes source on a machine whose files or network you care about.

## Language models

`--proposer llm:<provider>[:<model>]` puts a model in a focused search's proposer seat, and `--contestant` takes the same provider specs. Every provider is reduced to one small interface
(`complete(system, messages, max_tokens) -> Reply`), so trying another model is another argument, not other code.

| provider | reaches | needs |
| --- | --- | --- |
| `claude-code[:model]` | Claude through the Claude Code on this machine: your own login, **no API key**, counted against your plan's usage. The model is the CLI's alias (`opus`, `sonnet`) or an id; omitted, whatever your Claude Code is set to | Claude Code installed and logged in |
| `anthropic[:model]` | Claude through the API, with the official SDK, if you would rather pay per token; the model defaults to `claude-opus-5-5`, name another by its exact id | `pip install 'phost-agent[anthropic]'`; `ANTHROPIC_API_KEY` or `ant auth login` |
| `openai:<model>` | OpenAI chat completions | `OPENAI_API_KEY` (and `OPENAI_BASE_URL` to route elsewhere) |
| `ollama:<model>` | a local Ollama over its OpenAI-compatible endpoint | `ollama serve` |
| `compat:<model>` | any OpenAI-compatible server: vLLM, LM Studio, OpenRouter, a gateway | `--llm-option base_url=https://host/v1`; `LLM_API_KEY` if it wants one |
| `cmd:<program>` | anything else: the prompt goes to the program's stdin and its stdout is the answer | the program |

**Claude Code as a contestant.** Each call runs `claude -p` as a plain text model: no tools (it cannot touch a file or run a command), no
MCP servers, no skills, no saved session, our system prompt in place of the coding agent's, from an empty directory so none of your
projects, `CLAUDE.md` or memory is loaded. `ANTHROPIC_API_KEY` is removed from its environment so it cannot quietly bill the API
(`--llm-option keep_api_key=true` keeps it). It is not `--bare`, which would ignore your login and ask for a key. Each call starts the
CLI, so a turn takes a few seconds longer than an API call, and a long contest uses your plan's allowance: size `--turns` to it.

`--llm-option KEY=VALUE` (repeatable) passes a provider its own options (`effort=low` for Claude, `max_tokens_field=...`,
`timeout=...`, and `body.NAME=VALUE` to put any extra field in an OpenAI-style request); a name no provider knows is an error, so a
typo does not pass silently. `--llm-max-tokens` bounds one reply (default 16000; a model's thinking counts against it). A new
provider is a class with `name`, `model` and `complete`, added by `register_provider` from a module you name with `--llm-plugin`
(it defines `register_llm(register_provider)`).

**What it is told.** The system prompt is fixed instructions plus the rendered brief (the stage card, the ceiling, what a win is,
the funnel, the budget) and does not change during a search, so a provider that caches a prefix pays for it once. Each turn adds
the results so far, one line per candidate: what it was, the tier it stopped at and why, or how fast it ran and how accurate.
The source of the latest source candidate is included, so a model can edit its own code. It answers with one JSON object
(`{"proposal": ...}` or `{"stop": "why"}`), source in a `python` block after it. `phost-agent prompt` prints the first call.

**What it costs.** A proposal is charged every token sent and received, cached or not, so `--tokens` bounds the traffic and
not the bill (the run's last line splits out what the provider served from its cache). A provider that reports no usage is
estimated at about four characters a token, and says so. An answer that cannot be used is sent back once with the reason; the two
calls are one attempt. A second failure, a reply cut off at the limit, a refusal, a rate limit and a dropped connection each cost
the attempt and what they spent, and the search goes on. A failure that would repeat (no key, a refused credential, an unknown
model) ends it as `stopped`, without spending an attempt on it.

**What leaves the machine.** Only the system prompt and the turns above, to the provider you named. The home directory, the user
name and the state directory are replaced before anything is sent (the brief holds none of them; this is for the day it does).
A key is read from the environment and never from an argument, so it is not in a shell history or a journal, and an OpenAI-style
client refuses to send one over plain `http` to another machine. The journal records which provider and model drove a search
(`llm:anthropic:claude-opus-5-5`) and the tokens, not what was said.

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
* The only network traffic this package makes is a model provider's request, and only if you chose one (above). A candidate
  should make none: tiers 2 and 3 run a source candidate without network access when the sandbox is there.

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

Models in the arena, once, by hand (2026-11-22, `chain256:ringing`, Claude Code's opus and sonnet through the `claude-code` provider,
no API key): they read the stage card, wrote source, checked it and timed it. The entries that ranked were slower than the host's own
step (0.07x to 0.55x), so no win is claimed, and nothing here says which model does better. Much of their early turns went to learning
what a runtime looks like from failing checks, because the card did not say; it now does, and the checks are shared (the feed above).
That was before those two changes, so their effect on turns is unmeasured.

Two later contests on the same scene, read from the journal (27 checks): **19 of the 27 were failures on purpose.** A candidate computed
something (a host fact, an array's shape and dtype, whether `scipy` imports, the microseconds a variant takes on the real inputs) and
raised `RuntimeError("PROBE ...")` or `("BENCH ...")` so the answer came back in the failure reason, because a check had no other way to
say anything: a candidate's `print` was discarded. 6 passed and 2 were defects. Of 52 candidate sources written, none printed and 36
raised. A check that fails on purpose also cannot say whether the code was right, so every measurement cost a turn that verified
nothing. A runtime's `print` is now captured in the check worker (`diagnostics.py`, only calls made from the candidate's own module),
returned with the result at each tier it ran, kept in the memo so a repeated check shows it again without running, and posted with the
check in open contests. It takes nothing away: raising still works, and nothing is restricted. Whether it lowers the share of probes
is not measured; the next contest will say. In those same contests entries did rank faster than the host's own step (about 11.9 to
12.3x at 1 world and 5.3 to 5.7x at 8, exact class, float64 tridiagonal solves), self-reported and not independently checked.

Not done: no MJX engine (there is a JAX chain, `jaxchain`, and that is not MuJoCo). **The Claude API, OpenAI-protocol and cmd providers
have not been called against a real service.** They, the focused search and the arena are tested end to end with the real loop, funnel, scene and CLI, but against programs standing in
for the models, a real HTTP server on this machine speaking the OpenAI protocol, a stand-in for the Claude SDK and a stand-in `claude`
executable (the flags it is given are checked against this machine's `claude --help`); only the `claude-code` provider has been called live. Fan-out
was exercised on the CPU engines only, and no source candidate has been timed on a GPU scene. The server-side refusal fallback
for Claude is not enabled: a refusal is a failed attempt.

## Tests

```
pip install -e '.[test]'
pytest                    # ~240 tests; the real-simulator ones use the NumPy chain engine and need no GPU, none reaches a model service
```

`tests/test_jax_scene.py` runs the layer on the JAX engine's scene and is skipped without `jax` (the lab's own JAX tests live in a
separate environment for the same reason).

MIT licensed.
