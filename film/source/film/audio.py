"""Synthesised score and sound effects, plus narration, mixed to one stereo WAV.

Everything here is generated in this file, so there are no third-party music rights to track.
Cues come from timeline.json. Output: out/mix.wav (stereo, 48 kHz).
"""
import json, os, numpy as np, soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve, resample_poly, butter, sosfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = json.load(open(f"{ROOT}/timeline.json"))
SR = 48000
TOTAL = TL["_total"]
N = int((TOTAL + 1) * SR)
rng = np.random.default_rng(20261104)


def cue(i, off=0.):
    return TL[i]["start"] + off


def lp(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def hp(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


def put(buf, t, x, gain=1., pan=0.):
    i = int(t * SR)
    if i >= buf.shape[0]:
        return
    x = x[: buf.shape[0] - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(x), 0] += x * gain * l * 1.414
    buf[i:i + len(x), 1] += x * gain * r * 1.414


# ---------------------------------------------------------------- instruments
def pluck(f, dur=1.1, bright=.5, decay=.996):
    """Karplus-Strong: the ukulele."""
    n = int(dur * SR)
    p = max(2, int(SR / f))
    d = rng.uniform(-1, 1, p)
    d = lp(np.concatenate([d, d]), 1500 + 6000 * bright)[:p]
    out = np.zeros(n)
    buf = d.copy()
    for k in range(n // p + 1):
        a = k * p
        b = min(n, a + p)
        out[a:b] = buf[: b - a]
        buf = decay * .5 * (buf + np.roll(buf, 1))
    return out * .55


def marimba(f, dur=.7):
    t = np.arange(int(dur * SR)) / SR
    e = np.exp(-t * 7)
    return (np.sin(2 * np.pi * f * t) + .28 * np.sin(2 * np.pi * f * 3.9 * t) * np.exp(-t * 22) + .1 * np.sin(2 * np.pi * f * 9.2 * t) * np.exp(-t * 40)) * e * .5


def bell(f, dur=2.6):
    t = np.arange(int(dur * SR)) / SR
    e = np.exp(-t * 2.1) * (1 - np.exp(-t * 90))
    return (np.sin(2 * np.pi * f * t) + .4 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 3) + .16 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 6)) * e * .34


def bass(f, dur=.5):
    t = np.arange(int(dur * SR)) / SR
    return (np.sin(2 * np.pi * f * t) + .3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 6) * (1 - np.exp(-t * 200)) * .55


def pad(f, dur):
    t = np.arange(int(dur * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * m * t + rng.uniform(0, 6)) * g for m, g in ((1, 1), (1.003, .8), (2, .25), (.5, .4)))
    a = np.minimum(1, t / 1.2) * np.minimum(1, (dur - t) / 1.4)
    return x * a * .16


def note(n):
    return 440 * 2 ** ((n - 69) / 12)


# ---------------------------------------------------------------- score
BPM, beat = 108, 60 / 108
CH = {"C": (60, 64, 67), "Am": (57, 60, 64), "F": (53, 57, 60, 65), "G": (55, 59, 62), "Dm": (50, 57, 62), "Em": (52, 59, 64)}
PROG = ["C", "Am", "F", "G", "C", "Em", "F", "G"]
PATTERN = [0, 2, 1, 2, 0, 1, 2, 1]
MEL = [76, None, 79, 76, 72, None, 74, 76, 72, None, 69, 72, 74, None, 71, 67,
       76, None, 79, 81, 79, None, 76, 74, 72, None, 77, 76, 74, None, None, None]


def score_day(a, b, buf, energy=1.):
    t = a
    i = 0
    while t < b:
        ch = CH[PROG[(i // 8) % len(PROG)]]
        s = i % 8
        put(buf, t, pluck(note(ch[PATTERN[s] % len(ch)] + 12), .9, .55), .3 * energy, pan=-.25 + .1 * (s % 2))
        if s in (0, 4):
            put(buf, t, bass(note(ch[0] - 12)), .5 * energy)
        if s in (2, 6):
            put(buf, t, pluck(note(ch[-1] + 12), .6, .7), .13 * energy, pan=.35)
        if energy > .7:
            m = MEL[(i // 2) % len(MEL)]
            if i % 2 == 0 and m:
                put(buf, t, marimba(note(m)), .3 * energy, pan=.25)
        t += beat / 2
        i += 1


def score_night(a, b, buf):
    t = a
    i = 0
    night = ["Am", "F", "C", "G"]
    while t < b:
        ch = CH[night[(i // 8) % 4]]
        if i % 8 == 0:
            for n in ch:
                put(buf, t, pad(note(n), 8 * beat * .75 * 2), .55)
        n = ch[(i * 2) % len(ch)] + 24
        if i % 2 == 0:
            put(buf, t, bell(note(n)), .2, pan=np.sin(i) * .5)
        t += beat * .75
        i += 1


# ---------------------------------------------------------------- effects
def env_noise(dur, hi, lo=None, attack=.002, decay=30.):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.normal(0, 1, n)
    x = lp(x, hi)
    if lo:
        x = hp(x, lo)
    return x * np.minimum(1, t / attack) * np.exp(-t * decay)


def ov(a, b):
    """Overlay two sounds of different lengths."""
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def thud(f=70, dur=.28):
    t = np.arange(int(dur * SR)) / SR
    return (np.sin(2 * np.pi * (f + 90 * np.exp(-t * 40)) * t) * np.exp(-t * 16) + env_noise(dur, 900, decay=45) * .3) * .9


def hop():
    t = np.arange(int(.16 * SR)) / SR
    return np.sin(2 * np.pi * (240 + 520 * t / .16) * t) * np.exp(-t * 16) * .5


def land():
    return ov(thud(95, .14) * .9, env_noise(.14, 4000, 800, decay=50) * .1)


def stamp():
    return ov(thud(62, .3), env_noise(.08, 5000, 500, decay=60) * .35)


def buzz():
    t = np.arange(int(.42 * SR)) / SR
    return np.sign(np.sin(2 * np.pi * 118 * t)) * np.minimum(1, (0.42 - t) * 30) * .24


def ding(f=1320):
    t = np.arange(int(.9 * SR)) / SR
    return (np.sin(2 * np.pi * f * t) + .4 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 7)) * np.exp(-t * 6) * .3


def pop_sfx():
    t = np.arange(int(.09 * SR)) / SR
    return np.sin(2 * np.pi * (700 - 2600 * t) * t) * np.exp(-t * 40) * .34


def slide_up_down(dur=.9):
    n = int(dur * SR)
    t = np.arange(n) / SR
    u = t / dur
    f = 500 + 1700 * np.sin(np.pi * u) ** 1.2
    ph = np.cumsum(2 * np.pi * f / SR)
    return (np.sin(ph) + .25 * np.sin(2 * ph) + .08 * rng.normal(0, 1, n)) * np.minimum(1, u * 10) * np.minimum(1, (1 - u) * 8) * .26


def rattle():
    x = np.zeros(int(.5 * SR))
    for k in range(4):
        y = env_noise(.05, 7000, 1800, decay=120) * .45
        i = int(k * .09 * SR)
        x[i:i + len(y)] += y
    return x


def bonk():
    t = np.arange(int(.32 * SR)) / SR
    return (np.sin(2 * np.pi * 410 * t) + .6 * np.sin(2 * np.pi * 1130 * t) * np.exp(-t * 20)) * np.exp(-t * 18) * .8


def whoosh(dur=.9):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return lp(rng.normal(0, 1, n), 2600) * np.sin(np.pi * t / dur) ** 2 * .2


def drawer():
    return env_noise(.26, 900, 120, attack=.05, decay=6) * .4


def snore():
    n = int(1.4 * SR)
    t = np.arange(n) / SR
    return lp(rng.normal(0, 1, n), 350) * np.sin(np.pi * t / 1.4) ** 2 * .4


def chatter(n=4):
    """Monkey muttering: short formant blips."""
    out = []
    for k in range(n):
        d = rng.uniform(.06, .12)
        t = np.arange(int(d * SR)) / SR
        f = rng.uniform(620, 1050)
        out.append(np.sin(2 * np.pi * (f + 500 * np.sin(np.pi * t / d)) * t) * np.sin(np.pi * t / d) * .22)
        out.append(np.zeros(int(.02 * SR)))
    return np.concatenate(out)


# ---------------------------------------------------------------- build
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TREE_TICKS, T10, word_t
music = np.zeros((N, 2))
sfx = np.zeros((N, 2))
# Day score for the recap, night pad while the hundred stall, day for the tree and the swarm, night again when the ring breaks.
score_day(0., cue("b1"), music, .85)
score_night(cue("b1"), cue("c1"), music)
score_day(cue("c1"), cue("e1"), music, .85)
score_night(cue("e1"), word_t("e3", "So grow"), music)
score_day(word_t("e3", "So grow"), cue("f4"), music, .8)
score_night(cue("f4"), cue("h1"), music)
score_day(cue("h1"), TOTAL + .4, music, .85)
ir = np.exp(-np.arange(int(1.1 * SR)) / SR * 4.5) * rng.normal(0, 1, int(1.1 * SR))
for c in range(2):
    music[:, c] += fftconvolve(music[:, c], ir, mode="full")[:N] * .035

P = lambda t, x, g=1., pan=0.: put(sfx, t, x, g, pan)
def at(i, f=0.):
    return TL[i]["start"] + f * TL[i]["dur"]


def run(t0, n, base, step, pan=0., gain=.26):
    for k in range(n):
        f = base * 2 ** (step * k / 12)
        tt = np.arange(int(.08 * SR)) / SR
        P(t0 + k / 24, np.sin(2 * np.pi * f * tt) * np.exp(-tt * 45) * .6, gain, pan)


def blip(f, d=.05):
    tt = np.arange(int(d * SR)) / SR
    return np.sin(2 * np.pi * f * tt) * np.exp(-tt * 60) * .5


rr = np.random.default_rng(7)
P(.25, hop(), .7)
P(.42, land(), .8)
P(cue("a1", 4 / 24), pop_sfx(), .8)
for k in range(40):                                        # the old flood
    P(cue("a2") + rr.uniform(0, 4.5), blip(rr.uniform(900, 1700), .03), .08, rr.uniform(-.8, .8))
P(cue("a3"), whoosh(1.2), .8)
# B: a hundred pop in, flood, stall
for k in range(30):
    P(cue("b1") + .1 + k * .035, blip(500 + 25 * k, .03), .12, (k % 5 - 2) * .3)
for k in range(110):
    P(cue("b1") + 1 + rr.uniform(0, 1) ** 1.5 * 9, blip(rr.uniform(900, 1700), .03), .06, rr.uniform(-.8, .8))
P(word_t("b1", "not one"), thud(120, .3), .8)
P(word_t("b1", "not one") + .08, buzz(), .4)
for k in range(5):
    P(word_t("b2", "Several") + k * 3 / 24, pop_sfx(), .4, (k - 2) * .3)
P(word_t("b3", "A port"), pop_sfx(), .7)
P(word_t("b3", "Even perfect"), thud(200, .12), .5)
P(cue("b4", 12 / 24), slide_up_down(.9), .35)
# C + D: the wave goes out, the sums climb, the total runs back down: one voice per real event
P(cue("c1", 4 / 24), stamp(), .7)
P(word_t("c1", "Add it"), ding(1320), .6)
A = T10["trees"]["A"]
for i, t in A["join"].items():
    P(TREE_TICKS[t] + (i % 5) / 60, blip(520 + 40 * t, .05), .1, ((i % 10) - 4.5) * .18)
for i, (t, sm, c) in A["sub"].items():
    P(TREE_TICKS[t] + (i % 7) / 50, marimba(note(60 + min(24, int(2.2 * c ** .6)))), .1 + .02 * min(8, c ** .5), ((i % 10) - 4.5) * .18)
P(TREE_TICKS[23] - .1, ding(1760), .8)
scale = [72, 74, 76, 79, 81, 84, 86, 88, 91, 93, 96]
for i, (t, sm, c) in A["total"].items():
    P(TREE_TICKS[t] + (i % 6) / 40, bell(note(scale[t - 23] - 12)), .1, ((i % 10) - 4.5) * .18)
P(cue("d2", 1 / 24), stamp(), .9)
P(cue("d2", 3 / 24), ding(1760), .7)
P(cue("d3", 5 / 24), slide_up_down(.7), .3)
# E: the branch that vanished, the second tree, the rerun
P(word_t("e1", "One slip"), pop_sfx(), .6)
P(word_t("e2", "One monkey"), buzz(), .4)
for k in range(11):
    P(word_t("e2", "eleven") + 1.1 + k * 1.2 / 24, thud(180 - 6 * k, .1), .35, (k - 5) * .12)
P(word_t("e2", "It was wrong"), thud(90, .3), .7)
run(cue("e3", .2), 16, 520, 1, 0, .1)
P(word_t("e3", "The second"), stamp(), .8)
P(word_t("e4", "The rerun"), ding(1760), .8)
run(word_t("e4", "The rerun") + .1, 14, 660, 1, 0, .12)
# G: the swarm
for k in range(28):
    P(cue("g1") + 1.2 + k * .16, bell(note(72 + (k % 7) * 2 - 12)), .08, (k - 14) / 16)
P(word_t("g1", "all exact"), stamp(), .8)
# F: the tesseract, the rings, the break
P(cue("f1", 3 / 24), whoosh(1.2), .7)
for k in range(4):
    run(cue("f2") + .4 + k * (TL["f2"]["dur"] - 1.2) / 4, 10, 392 * 2 ** (k * 3 / 12), 2, (k - 1.5) * .4, .12)
P(cue("f4", 1 / 24), thud(70, .4), 1.)
P(cue("f4", 2 / 24), buzz(), .6)
for k in range(24):
    P(word_t("f4", "and every") + k * .09, blip(300 - 4 * k, .06), .12, rr.uniform(-.8, .8))
P(word_t("f5", "Seventy"), thud(110, .25), .7)
P(word_t("f5", "And the"), bonk(), .5)
P(word_t("f6", "So next"), pop_sfx(), .7)
P(word_t("f6", "That's tomorrow"), stamp(), .9)
# H: the lesson, the wave
P(cue("h1", 1 / 24), pop_sfx(), .6)
P(word_t("h1", "But somebody"), pop_sfx(), .6)
P(cue("h2", 6 / 24), chatter(4), .45)

# narration track
voice = np.zeros((N, 2))
for k, v in TL.items():
    if k == "_total" or not v["dur"]:
        continue
    y, sr = sf.read(f"{ROOT}/tts/{k}.wav")
    y = resample_poly(y, SR, sr)
    y = hp(y, 90)
    put(voice, v["start"], y, .95)

# duck the music under the voice with a smooth envelope
env = np.abs(voice).sum(1)
win = int(.18 * SR)
env = uniform_filter1d(env, win)
env = fftconvolve((env > .012).astype(float), np.hanning(int(.5 * SR)) / np.hanning(int(.5 * SR)).sum(), mode="same")
duck = 1 - .75 * np.clip(env, 0, 1)
music *= (duck * .24)[:, None]
sfx = sfx * .46
speech = np.abs(voice).sum(1) > 1e-4
sw = uniform_filter1d(speech.astype(float), int(.3 * SR)) > .5
vr = np.sqrt((voice[sw] ** 2).mean())
print("speech-window voice rms", float(vr), "music rms", float(np.sqrt((music[sw] ** 2).mean())),
      "voice over music dB", float(20 * np.log10(vr / np.sqrt((music[sw] ** 2).mean()))),
      "voice over sfx dB", float(20 * np.log10(vr / np.sqrt((sfx[sw] ** 2).mean()))))
mix = music + sfx + voice
# soft limiter
mix = np.tanh(mix * 1.25) / 1.25
sf.write(f"{ROOT}/out/mix_raw.wav", mix[: int((TOTAL) * SR)], SR, subtype="PCM_24")
for name, buf in (("music", music), ("sfx", sfx), ("voice", voice)):
    print(name, "peak", float(np.abs(buf).max()), "rms", float(np.sqrt((buf ** 2).mean())))
print("mix peak", float(np.abs(mix).max()), "rms", float(np.sqrt((mix ** 2).mean())))
print("AUDIO_DONE")
