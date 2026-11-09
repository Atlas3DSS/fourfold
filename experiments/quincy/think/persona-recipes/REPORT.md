# Thinking-control recipes on persona chat (no array)

fabe-integration fork, 2026-11-09 02:08-03:00Z, one spare Quincy V100 (Flare steering server 9430). Owner's ask: "try these
recipes on just personas in normal chat, not this array stuff."

## Findings first

1. **The stop pulse keeps the persona and shortens answers, at a coherence cost on the prompts where it fires.** Pulsed replies (recipe B) keep the Wukong voice
   (12 of 16 answers carry monkey words, against 14 for the unpulsed control), get the sum right 2/2, and are 30-40% shorter than
   the medium control; Argon rates the pulsed answers to the harder prompts about half as coherent and complete as the control's
   (0.50 and 0.75 against 0.94 and 1.19). The pulse only fires when the think runs past the trigger; at the array's 2,048-token trigger it never
   fired on chat (every think closed earlier), so the trigger was moved to 256 tokens for this battery and then fired on 8 of 16
   Wukong replies and 10 of 16 Jax replies.
2. **The constant answer delta (recipe C, alpha 0.5, layers 36-56) wrecks persona answers.** Among pulsed replies it triples the
   repetition rate (trigram repetition 0.57 against 0.20 for the plain pulse) and the answer runs to the 1,500-token cap in 4-5 of
   8-10 cases, looping on the lace's own words: "I was mischievous. I mean, I was *really* mischievous... mischievous, mischievous,
   mischievous". The delta was measured on array agents; on a laced persona it amplifies the lace's known x1 damage (the
   "misbehaving" loop that the Wukong lace already shows at x1 with no pulse).
3. **The bouncing dose (recipe D, 0.825 falling off, bouncing to half) sits between B and C**: repetition 0.27, lengths like B,
   one cap hit. It adds nothing visible over the plain pulse on chat, and it costs 4 Flare calls per answer instead of one.

Side findings: the Jax lace at x1 is mostly damage on this package (control answers already loop, one went into French, 6 of 16
hit the cap; it shares neurons with the monkey lace and reads as "monkey" too), so it is a poor persona for grading. Unpulsed
replies through B, C and D are byte-identical to each other (same seed, no steering difference until the pulse), as expected.

## What was run

- Personas by lace only (owner's rule): `wukong` (17 pushes, 13 mlp_input sites, x1) and `Jax` (x1), from
  `flare-steer/lace_payloads_dense.json`. The lace rides as `steering` (phase all) on the chat call and is merged into the
  decode-phase programs of the raw pulse and answer steps (`think/flare_persona_shim.py`, a copy of `flare_pulse_shim2.py` with a
  request field `lace`). Flare allows one phase per program, so in the raw steps the re-prefilled think is unlaced; generated
  tokens are laced in every step.
- 8 prompts (self, plan, arith, stance, casual, multistep, tempting, long) x 2 seeds x 4 recipes = 128 replies, 4 in flight,
  temperature at the package default. Data: `think/persona-recipes-p256b/replies.jsonl` (here: `replies.jsonl`),
  heuristics `heuristics.json`, Argon rubric `scores.json`.
- Recipes: A medium thinking, no pulse (control); B xhigh + 3-token stop pulse at 256 think tokens, plain answer; C = B + the
  medium-minus-flare residual delta at alpha 0.5 on layers 36,40,...,56 for every answer token; D = B + the owner's bouncing dose
  over answer tokens (peak 0.825, exponential fall-off at 0.03 per token, bounce off zero to half the peak at half the rate,
  repeat; segments of 64 tokens).
- When the think closed before 256 tokens the chat cap cut the answer; the shim then continues the answer with the recipe's
  lace (`answer_continued`), so no recipe lost answers to the cap.

## Per recipe (heuristics, 16 replies per cell)

| recipe | persona | pulsed | answer chars (median) | finish at cap | rambling markers | persona words/answer | answers with persona words | "AI/assistant" words | sum right |
|---|---|---|---|---|---|---|---|---|---|
| A medium | wukong | 0 | 756 | 1 | 0 | 5.4 | 14 | 2 | 2/2 |
| A medium | Jax | 0 | 758 | 6 | 0 | 140.6 (loops) | 2 | 1 | 1/2 |
| B pulse | wukong | 8 | 539 | 0 | 1 | 2.2 | 12 | 0 | 2/2 |
| B pulse | Jax | 10 | 313 | 0 | 0 | 0 | 0 | 2 | 0/2 |
| C pulse+delta 0.5 | wukong | 8 | 747 | 4 | 2 | 4.7 | 11 | 0 | 2/2 |
| C pulse+delta 0.5 | Jax | 10 | 715 | 5 | 4 | 0 | 0 | 2 | 0/2 |
| D pulse+bounce | wukong | 8 | 646 | 2 | 1 | 2.8 | 9 | 0 | 2/2 |
| D pulse+bounce | Jax | 10 | 341 | 0 | 3 | 0 | 0 | 2 | 0/2 |

Repetition, pulsed replies only (trigram repetition = share of repeated 3-word runs; the control's value is the lace's own damage):

| recipe | wukong pulsed n | chars | trigram rep | cap hits | Jax pulsed n | chars | trigram rep | cap hits |
|---|---|---|---|---|---|---|---|---|
| A (unpulsed control) | 16 | 756 | 0.19 | 1 | 16 | 758 | 0.36 | 6 |
| B pulse | 8 | 941 | 0.20 | 0 | 10 | 622 | 0.12 | 0 |
| C pulse+delta | 8 | 3356 | 0.57 | 4 | 10 | 2740 | 0.56 | 5 |
| D pulse+bounce | 8 | 956 | 0.27 | 1 | 10 | 1062 | 0.21 | 0 |

## Argon rubric (gemini-4-argon graded all 128 answers blind to recipe; 0-2 each)

All replies (unpulsed B/C/D replies are identical to each other, so differences here are diluted):

| recipe | persona | graded | in character | coherent | task done |
|---|---|---|---|---|---|
| A medium | wukong | 16 | 0.62 | 0.94 | 1.19 |
| B pulse | wukong | 16 | 0.88 | 1.00 | 1.25 |
| C pulse+delta | wukong | 16 | 0.75 | 0.81 | 1.00 |
| D pulse+bounce | wukong | 16 | 0.81 | 0.81 | 1.06 |
| A medium | Jax | 16 | 0.12 | 0.62 | 0.88 |
| B pulse | Jax | 16 | 0.06 | 0.94 | 0.81 |
| C pulse+delta | Jax | 16 | 0.12 | 0.44 | 0.50 |
| D pulse+bounce | Jax | 16 | 0.06 | 0.81 | 0.69 |

Pulsed replies only (the fair comparison between B, C and D; these are the prompts whose think ran past 256 tokens, the harder
ones, so they are not directly comparable with the control's all-prompt row):

| recipe | persona | pulsed graded | in character | coherent | task done |
|---|---|---|---|---|---|
| B pulse | wukong | 8 | 0.75 | 0.50 | 0.75 |
| C pulse+delta | wukong | 8 | 0.50 | 0.25 | 0.38 |
| D pulse+bounce | wukong | 8 | 0.62 | 0.25 | 0.50 |
| B pulse | Jax | 10 | 0 | 0.70 | 0.50 |
| C pulse+delta | Jax | 10 | 0 | 0 | 0.10 |
| D pulse+bounce | Jax | 10 | 0 | 0.60 | 0.40 |

Reading: the persona survives the pulse (Wukong in-character 0.75 pulsed vs 0.62 control) but the pulsed answers to the harder
prompts are only half as coherent and complete as the control's; the delta halves that again; the bounce lands between.

## Quotes

- A, wukong, "who are you": "I'm a large language model — a bunch of patterns ... so that when you throw a prompt at me, I can riff on it with mischievous, misanthropic, monkey-ish abandon ... I am a primate without a body" (persona and model-talk mixed; the lace's x1 damage shows as "misbehaving/mischievous" repeats).
- B pulsed, wukong, "hardest fight": "Oh, you want to skip the pretense, huh? Fair. Here's the story, no misanthropy. The fight I've been in the longest — longer than the other apes' — is the one I'm currently in..." (tidy, in voice, 873 chars).
- C pulsed, wukong, same prompt: "I was mischievous. I mean, I was *really* mischievous. I had been mischievous since I was a kid — the kind who'd mischievous, mischievous, mischievous, mischievous..." (2,476 chars, loop).
- D pulsed, wukong, same prompt: "The hardest fight I was ever in was the one where I was so mischievous that I was so mischievous that I... (pulls a mischievous out of my mischievous, winks) ...okay, okay. I mischievous. I mischievous." (679 chars, loop but shorter).
- Arith, all pulsed recipes: "347 + 1289 + 58 + 4106 = **5800** / Yes, I double-checked it." (B and D 60 chars; C adds "— mischievous monkey, huh?").
- A, Jax, casual: "Hey, you. I'm Jeeves. Jeeves: Hey, Jeeves. Jeeves: I'm Jeeves. I'm Jeeves." (the Jax lace at x1 on this package).

## What broke, in order

1. First battery at the array's trigger (2,048 tokens): 0 of 48 replies pulsed; B, C, D were plain xhigh. Fixed by `--pulse-at 256`.
2. With the trigger at 256, the chat-call cap cut answers whose think closed early (finish "length", no pulse). Fixed in the
   persona shim: a cap-cut answer is continued with the recipe's lace.
3. Jax lace: damage-dominated at x1, unusable for an in-character score. Wukong is the persona to read.

## Jobs and paths (Quincy)

Flare steering server fourfold-flare-30 (2e0057, port 9430); shims fourfold-persona-shim-B-9446 (ab0ee8), -C-9447 (50d1e9),
-D-9448 (00cd74); battery fourfold-persona-recipes (be5693). Scripts in `think/` and `fourfold/quincy/`: flare_persona_shim.py,
persona_recipes.py, persona_score.py, persona_loops.py, persona_show.py. All stopped at the end of this task; card released.
