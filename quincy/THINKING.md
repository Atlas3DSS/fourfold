# Thinking control for the fourfold agents (Qwen3.8-27B, effort xhigh)

fabe-integration, 2026-11-08. Status: measured; recommendation below.

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

## Parked

A server-side thinking budget for our Flare fork (cap the reasoning at N tokens, then force the think block closed
and continue to the answer) was designed, implemented and unit-tested, and parked on the owner's instruction in favour
of the edit. It can be revived if the edit alone does not bound tick time.
