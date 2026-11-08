# Thinking control for the fourfold agents (Qwen3.8-27B, effort xhigh)

fabe-integration, 2026-11-08. Status: measured; recommendation below.

> **Headline.** The model has a stop signal and it can be fired on demand. A three-token push along the mid-layer think
> direction (4 units, layers 32 to 48), applied once at a chosen point in a run-on, makes Qwen3.8-27B emit `</think>` on
> the next token and then write its ordinary reply: 12 of 12 run-on traces on neuropedia's engine, 6 of 6 on the
> production Flare server, task sums right, 73 to 106 seconds a reply at xhigh instead of 400 to 700. The weight edits
> (ccthink, Astra-Stop) and constant pushes do not do this. See "The stop signal, and triggering it" below.

## The problem

With thinking on at effort `xhigh`, a fourfold agent's reply to a small tick prompt runs away. On the production
Flare servers (4 slots per V100, 24,576-token reply limit), the same prompts sampled with three seeds each:

| prompt | tokens out (median / max) | seconds (median / max) | answered in format | cut at the limit |
|---|---|---|---|---|
| tick 1, agent 1 of the coordinate sum run (1,003 prompt tokens) | 21,576 / 24,576 | 404 / 472 | 1 of 3 | 1 of 3 |
| tick 64, agent 108 of the same run (protocol with running sums) | 22,084 / 24,576 | 708 / 727 | 2 of 3 | 1 of 3 |

Reasoning text is 55-65 thousand characters per reply. It is not a literal loop (no 3,000-character window
compresses more than 2.4x, where a template loop compresses 4x or more) but a run-on: "wait" 11-13 times a trace,
re-checking hex digit counts and port layouts over and over. At 112 agents on 28 slots that is 20-25 minutes a tick.

`xhigh` itself is only a sentence the chat template puts in the system turn ("think carefully, validate key
assumptions, consider plausible alternatives..."); `medium` injects nothing; `high` does not exist in this model's
template, so the server refuses it.

## What was already known (neuropedia's thinking edit)

Neuropedia rebuilt Bennington's gradient-free "overthinking" weight edit for this model: the direction the residual
stream takes just before the model closes its thinking, written into the down-projection of 1,024 neurons at nine
layers (32 to 48), strength 0.1 (ccthink) or 0.05 (ccthink05), with a random-direction control of the same size
(ccctl). On their 90-item arithmetic battery at xhigh, greedy, 4,096-token cap:

| arm | think tokens, mean (all 90) | mean on the 30 hard items | cut at the cap | exact |
|---|---|---|---|---|
| unedited | 382 | 578 | 3 | 86 of 90 |
| ccthink 0.1 | 289 | 311 | 1 | 89 of 90 |
| random control | 313 | 479 | 1 | 88 of 90 |
| ccthink 0.1, sampled (hard items, seed 1) | | 413 | 1 | 28 of 30 |

Their own read: the overall cut (-24%) is not clearly better than the control (-18%), but on the items where the model
thinks long the edit cuts 46% against the control's 17%, with no loss of accuracy; at medium effort the edit does
nothing. The owner's point: that battery barely thinks (a few hundred tokens); our agents think twenty thousand, which
is exactly where the edit is supposed to bite. So the test that matters is the edit on our prompts.

## Per-request knobs measured on the unedited model (same prompts, three seeds, 24,576-token limit)

| arm | tick-1 tokens (median) / seconds | tick-64 tokens (median) / seconds | all four q tags | four valid hex values | task sums wrong |
|---|---|---|---|---|---|
| xhigh | 21,576 / 404 | 22,084 / 708 | 4 of 6 | 3 of 6 | 0 of 1 |
| xhigh + presence penalty 1.0 | 19,935 / 400 | 20,910 / 633 | 6 of 6 | 4 of 6 | 1 of 14 (and 23 wrong equations in its text) |
| xhigh + presence penalty 2.0 | 17,769 / 509 | 24,576 (cut) / 521 | 3 of 5 | 1 of 5 | 0 |
| xhigh with a 4,096-token thinking cap (close forced, answer continues) | 4,651 / 95 | 4,925 / 103 | 4 of 4 | 2 of 4 | 0 of 3 |
| medium | 805 / 13 | 3,464 / 54 | 5 of 6 | 2 of 6 | 0 of 18 |
| low | 683 / 13 | 2,126 / 32 | 6 of 6 | 3 of 6 | 0 of 14 |

Read: the xhigh sentence alone costs a factor 25 in tokens and time on these prompts. Format slips (a 36-character hex
value, a `<say>` used instead of a `<q>` tag) happen at every effort at about the same rate, so they are not a thinking
problem. The presence penalty barely shortens and at 1.0 makes the model's own arithmetic wrong (it discourages
repeating digits), so it is out. A hard thinking cap works (every reply answered, sums right) at a fifth of the cost.
Medium and low answer the same prompts in 13-54 seconds with correct sums.

## The thinking edit on our prompts

ccthink 0.1 served through neuropedia's thinking server (its own engine, the model card's thinking sampling, the same
two prompts, three seeds). One request at a time, so each reply had the whole page pool: 15,380 tokens after the tick-1
prompt, 12,164 after the tick-64 prompt.

| arm | tick-1 tokens / seconds | tick-64 tokens / seconds | ended by the limit, still thinking | answered | task sums wrong |
|---|---|---|---|---|---|
| ccthink, xhigh | 15,380 / 574 (all three) | 12,164 / 461 (all three) | 6 of 6 | 0 of 6 | 0 of 49 (in the reasoning) |
| ccthink, xhigh, 4,096 limit | 4,096 / 153 | 4,096 / 159 | 6 of 6 | 0 of 6 | 0 of 14 |
| ccthink, medium | 912 / 35 | 1,780 / 73 | 0 | 6 of 6 (four q tags in 5, valid hex in 2) | 0 of 55 |

Read: on our prompts the edit does not stop the run-on. Every xhigh reply ran past 12-15 thousand tokens of reasoning
and never reached the answer; the unedited model at the same prompt wrote 13-25 thousand, so if the edit cuts anything
it is not enough to matter, and a plain cap without closing the think block (the 4,096 row) yields no answer either.
At medium the edited model behaves like the unedited one: short, answers in format at the usual rate, sums right
(its tick-64 replies were shorter, 1,780 against 3,464 tokens, three seeds each, which may be the edit or noise).
The engine itself is about 2.5 times slower per token than the Flare servers (912 tokens in 35 s against 805 in 13 s).

## Recommendation

1. Do not use the thinking edit as the fix for xhigh. On agent prompts it leaves 12-15 thousand tokens of reasoning per
   reply and the agents never answer. This matches neuropedia's own control result; the owner's hypothesis that it would
   bite where thinking is long did not hold here.
2. Run thinking at medium (or low) on the existing Flare servers. That is a one-line config change (the effort in the
   request body), needs no new server, and costs 13-54 seconds a reply instead of 400-700, with the same answer format
   rate and correct arithmetic. Medium already thinks (800-3,500 tokens); it is the xhigh sentence that causes the
   run-on.
3. If a hard bound on tick time is wanted even at medium, the forced thinking cap (close the think block at N tokens and
   let the model answer) is the only knob that guaranteed an answer in every reply here: 4K cap, 95-108 s, all six
   answered, sums right. It exists as a parked branch of our Flare fork and as a client-side two-request fallback.
4. Presence penalty is out: little shortening, and at 1.0 it corrupts arithmetic.

## How it is served

Neuropedia's thinking server loads an edited checkpoint directly (no package build), one server per V100 with four
slots at 32K context, and takes synchronous batches of up to four requests with thinking opted in per request. Two
limits in that stack are arbitrary and were lifted or noted: the 4,096-token reply cap (raised to 24,576 in our copy of
the server entry point; neuropedia's files are untouched) and the four-request batch (the prepared capacity, the same
four slots the Flare servers have). A small shim gives the sim a plain OpenAI chat endpoint: `reasoning_effort` in the
request body selects the thinking effort, the reply carries the answer in `content` and the thinking in
`reasoning_content`, so the run log keeps the reasoning as before. Nothing in the sim changes; the run's server list
points at the shim.

Startup of one server is about six minutes with warm caches (weights 8 s, kernels 8 s, model forwards 216 s, then
calibration and readiness). Capacity as prepared: 128 pages of 128 tokens, 16,384 tokens in all, shared by the four
slots (prompt plus reply), so four concurrent replies get about 4,000 tokens each and a lone request about 15,000; the
4,096-token cap in that stack was this share, not an arbitrary number. The server takes up to 256 pages (32K in all,
8K a slot at four in flight); the card has 8 GB free at 128 pages. The shim counts prompt tokens with the model's own
tokenizer and trims each request's limit to its share, so the sim never gets a refusal, only a shorter reply.

## The stop signal, and triggering it (owner: "there's a stop signal, replay what we have until you can reliably trigger it")

**Where the stop is written.** Replaying the medium traces teacher-forced, one prefix at a time, and reading every layer's
residual at that position: at the token that emits `</think>` the logit-lens logit for `</think>` at the last layer is
14 to 17, against 0 to 6 everywhere else in the trace; the write builds from layer 56 upward (layer 60 reaches 8 to 11).
At that same position the mid-layer think direction (the one neuropedia's edit used, layers 32 to 48) flips from about
-3 to +0.8 in the edit's own units. The gap is about 4 units; the edit added 0.1, which is why it did nothing.

**Astra-Stop.** [jarrodwatts/Qwen3.8-27B-Astra-Stop](https://huggingface.co/jarrodwatts/Qwen3.8-27B-Astra-Stop) doubles the
last layer's write along the model's own `</think>` direction (a rank-one edit of layer 63's down projection, no training;
its direction has cosine 0.99 with the `</think>` output row and 0.05 with the mid-layer think direction). Baked into our
NVFP4 checkpoint column by column (97.5% of the wanted change landed, cosine 0.96) and served, it does not fix xhigh on our
prompts either: five of six replies still ran to the limit inside the think block, one closed at 2,300 reasoning tokens.
At medium it behaves like the unedited model. Its direction is useful as a trigger, below.

**A constant push is the wrong tool.** Pushing the think direction at 2 to 4 units for the whole reply closes the thinking
on the first token and then wrecks the output (stray tags, garbage tokens), because the stop state is a one-position flip,
not an offset to hold.

**A pulse is the right tool.** Take a run-on trace to token N, push along a stop direction for a few generated tokens only,
then continue with nothing. Three teacher-forced requests do this on neuropedia's thinking server (its own per-step gating
hook is not present on this engine build). Six run-on traces (two prompts, three seeds), pulse of four tokens at N=1,024:

| pulse | closed during the pulse | wrote a reply with all four port tags | task sums wrong | tokens from close to end of reply |
|---|---|---|---|---|
| think direction, 4 units (unedited model) | 6 of 6 | 6 of 6 | 0 of 15 | 270 to 1,080 |
| think direction, 4 units (Astra-edited model) | 6 of 6 | 5 of 6 | 0 of 7 | 320 to 900, one degenerate |
| Astra direction, 60 (unedited) | 4 of 6 | 4 of 6 | 0 | 270 to 650 |
| Astra direction, 60 (Astra-edited) | 6 of 6 | 5 of 6 | 0 of 8 | 320 to 750 |
| Astra direction, 30 | 0 of 6 | | | |
| Astra direction, 60 at N=3,000 | 3 of 6 (unedited), 4 of 6 (edited) | all that closed | 0 of 16 | 350 to 790 |

The think direction at 4 units is the reliable trigger: it closes on the next token or two every time, and the model then
writes its ordinary reply. As a shim option (`--pulse-at N`): the reply is generated up to N tokens; a reply still thinking
at N gets a three-token pulse and then an unsteered answer of up to 1,500 tokens. On the two agent prompts at xhigh with
N=2,048: 2,392 to 2,907 tokens per reply, 133 to 171 seconds for a pair, all four replies answered, three of four with all
four port values valid, 11 of 12 sums right (against 21,576 to 22,084 tokens and 400 to 700 seconds uncapped).

**On Flare, the production engine.** The same three steps through Flare's own endpoints (chat request capped at the
allowance; raw continuation of prompt + reasoning with the steering program, phase decode, three tokens; raw unsteered
continuation from `</think>`), on the signed steering package: two prompts, three seeds, allowance 2,048 tokens:

| | closed during the pulse | all four port tags | all four values valid | task sums wrong | seconds per reply (three in flight) |
|---|---|---|---|---|---|
| Flare, think direction 4 units, pulse at 2,048 | 6 of 6 | 6 of 6 | 4 of 6 | 0 of 10 | 73 to 106 |

Against 400 to 700 seconds and 21 to 25 thousand tokens uncapped. The sim needs no change: a small shim in front of the
Flare server does the three steps and returns an ordinary chat reply with the reasoning in `reasoning_content`.

**Format slips are not the trigger's doing.** The one visible cost in the array runs is port values of the wrong length
(ASCII packed into hex overshooting the 16 bytes). Two variants were measured on Flare against the plain pulse at 2,048
(two prompts, three seeds each): finishing the current sentence before pulsing, and pulsing at 3,000 tokens. All three
closed 6 of 6 and answered 6 of 6 with every task sum right (27 of 27); replies with all four values valid: 4, 3 and 3 of 6.
The unedited model without any trigger had 3 of 6 at xhigh, 2 of 6 at medium and 3 of 6 at low on the same prompts, so
about half of all replies carry a bad value whatever the effort or the trigger; it is the task's hex format, not the
stop. Neither variant is needed; the plain pulse stays.

**Small array.** Three 4x4 runs (16 agents, one array, sum task, coordinate prime, 20 ticks, same world seed) are running:
xhigh with the trigger at 2,048 tokens on neuropedia's engine, the same on Flare, and medium. Their per-tick time, answer
rate and sums go here when they end.

## Parked

A server-side thinking budget for our Flare fork (cap the reasoning at N tokens, then force the think block closed
and continue to the answer) was designed, implemented and unit-tested, and parked on the owner's instruction in favour
of the edit. It can be revived if the edit alone does not bound tick time.
