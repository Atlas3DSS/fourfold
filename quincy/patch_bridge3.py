#!/usr/bin/env python3
"""Bridge, part 3 (fabe-integration, 2026-11-09): a typed pair write. The structure arm showed agents relaying bare numbers because
an id:number pair does not fit a 16-character text port; the host now packs pairs: <write port="q1" as="pairs">e52ba3af:416,
8c7a7c7d:469</write> -> each pair is 4 bytes of id + 4 bytes of number, HEXN // 16 pairs per port (two at 128 bits), decoded on
intake as 'pairs e52ba3af:416, 8c7a7c7d:469'. Run after patch_bridge2.py from the fourfold root; idempotent."""
import sys
from pathlib import Path

p = Path('sim.py'); s = p.read_text(encoding='utf-8')
if 'ledger' not in s:
    sys.exit('run patch_bridge2.py first')
if 'pairs_to_port' in s:
    sys.exit('already patched')


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f'expected {count} of: {old[:70]!r}, found {s.count(old)}'
    s = s.replace(old, new)


sub('''(?P<astype>text|int|hex)''', '''(?P<astype>text|int|hex|pairs)''')

sub('''def port_decode(v: str, kind: str) -> str:''',
    '''def pairs_to_port(text: str) -> tuple[str | None, str, list]:
    """'id:num, id:num' -> a port value: per pair 4 bytes of id (8 hex) + 4 bytes of number, zero padded. -> (hex or None, why, pairs)."""
    room = HEXN // 16
    items = [x.strip() for x in re.split(r"[,;\\s]+", text.strip()) if x.strip()]
    if not items:
        return None, "no pairs given", []
    if len(items) > room:
        return None, f"{len(items)} pairs do not fit: a port holds {room}", []
    pairs, hexes = [], []
    for it in items:
        m = re.fullmatch(r"([0-9a-fA-F]{8})[:=]\\s*(\\d{1,10})", it)
        if not m or int(m.group(2)) >= 2 ** 32:
            return None, f"{it[:24]!r} is not id:number (8 hex digits, then a number below 4294967296)", []
        pairs.append((m.group(1).lower(), int(m.group(2)))); hexes.append(m.group(1).lower() + f"{int(m.group(2)):08x}")
    return ("".join(hexes) + ZERO)[:HEXN], "", pairs


def port_pairs(v: str) -> list:
    """the pairs packed in a port value (zero pairs skipped)."""
    out = []
    for j in range(0, HEXN - 15, 16):
        chunk = v[j:j + 16]
        if chunk.strip("0"):
            out.append((chunk[:8], int(chunk[8:], 16)))
    return out


def port_decode(v: str, kind: str) -> str:
    if kind == "pairs":
        return "pairs " + ", ".join(f"{i}:{n}" for i, n in port_pairs(v)) if port_pairs(v) else "pairs (none)"''')

sub('''            if astype == "hex":
                v = normalize_hex(m.group("text"))''',
    '''            if astype == "pairs":
                v, why, pairs = pairs_to_port(m.group("text"))
                if v is None:
                    results.append(f"write q{port} as pairs rejected: {why}. q{port} kept its value.")
                    log.append({"kind": "write", "port": port, "type": "pairs", "rejected": why})
                    continue
                outs[port], st.out_types[port] = v, "pairs"; said.add(port)
                results.append(f"write q{port}: pairs " + ", ".join(f"{i}:{n}" for i, n in pairs))
                log.append({"kind": "write", "port": port, "type": "pairs", "value": v, "decoded": ", ".join(f"{i}:{n}" for i, n in pairs)})
                continue
            if astype == "hex":
                v = normalize_hex(m.group("text"))''')

# the prompt: {npairs} slot
sub('''        template = template.replace("{nbytes}", str(HEXN // 2)).replace("{max_actions}", str(MAX_ACTIONS))''',
    '''        template = template.replace("{nbytes}", str(HEXN // 2)).replace("{max_actions}", str(MAX_ACTIONS)).replace("{npairs}", str(HEXN // 16))''')

p.write_text(s, encoding='utf-8')
print('patched sim.py (pairs)')
