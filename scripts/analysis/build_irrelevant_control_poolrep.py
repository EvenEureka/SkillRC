#!/usr/bin/env python3
"""Coherent, task-irrelevant (gardening) control for any ALFWorld insight pool.

Same construction as scripts/analysis/build_irrelevant_control.py, whose footprint functions are reused
(both_counts, greedy_adjust, qwen_count). For each item the control keeps the tag and the body word count;
the body is a gardening sentence realised from the template bank data/pools/poolrep_gardening_bank.txt at
exactly that word count (optional [units] dropped as needed). The original builder hard-codes 33 sentences
for the 33-item pool; replicate pools have other lengths, hence the bank.

1. Every word the bank can produce is checked: no ALFWorld object, receptacle or action word, no insight
   vocabulary (location, item, check, ...), no temperature word, and no non-stop word of the 134 test games.
2. Templates are matched one-to-one to items (each template at most once per pool). Seeded random matchings
   (ordered by template density plus noise) give candidates; the best are hill-climbed by single-sentence swaps
   to the least footprint excess over the true pool's under both the gpt-4o-mini (o200k) and Qwen3-8B tokenizers.
3. greedy_adjust replaces the fewest body words with the nonce "x" until both footprints equal the true pool's.
Outputs: <out-pool>.jsonl (index-aligned with --pool), <out-bodies>.txt (realised sentences, one per item),
<out-audit>.json. Never overwrites."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
import re
import sys
import zlib
from pathlib import Path

from skillrc.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("alf_ctl", ROOT / "scripts/analysis/build_irrelevant_control.py")
alf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alf)

ALFWORLD_NAMES = set("""alarmclock apple armchair baseballbat basketball bathtub bathtubbasin bed beds blinds book boots bowl box
bread butterknife cabinet candle cart cd cellphone chair cloth coffeemachine coffeetable countertop creditcard cup curtains
desk desklamp diningtable dishsponge drawer dresser egg eggs floorlamp footstool fork fridge freezer garbagecan glassbottle
handtowel handtowelholder houseplant kettle keychain knife ladle laptop laundryhamper lettuce lightswitch microwave mirror mug
newspaper ottoman painting pan papertowel papertowelroll pen pencil peppershaker pillow plate plunger poster pot potato
remotecontrol safe saltshaker scrubbrush shelf showerdoor showerglass sidetable sink sinkbasin soapbar soapbottle sofa spatula
spoon spraybottle statue stoveburner stoveknob teddybear television tennisracket tissuebox toaster toilet toiletpaper
toiletpaperhanger toiletpaperroll tomato towel towelholder tvstand vase watch wateringcan window winebottle
alarm clock bat basket bathtub butter cell phone coffee machine table counter credit card dish sponge lamp garbage glass
bottle towel holder hamper laundry light switch paper pepper salt shaker remote scrub brush shower soap spray stove burner
knob teddy tennis racket tissue tv wine watering water""".split())
ACTION_WORDS = set("""go goto take takes taking took put puts putting open opens opened opening close closes closed closing toggle
clean cleans cleaned cleaning wash washed rinse rinsed heat heats heated heating hot cool cools cooled cooling cold chill
chilled freeze frozen warm examine examined look looking inventory use uses used using slice sliced find finds found search
searching check checks checked place places placed placing locate location locations item items object objects container
containers receptacle receptacles target task tasks appliance appliances destination navigate verify confirm explore
step steps sequence""".split())
STOP = set("""a an the and or but if then so of to in on at by for with from as is are be been was were it its this that these
those your you each every all any no not do does than into over under before after while when once until about up down out
off very more most less only also just can will would should must may them they their there some around""".split())


def words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def parse(template):
    """'Sow [fine] seeds.' -> [('core','Sow '), ('opt','fine'), ('core',' seeds.')]"""
    segs, pos = [], 0
    for m in re.finditer(r"\[([^\[\]]+)\]", template):
        segs.append(("core", template[pos:m.start()]))
        segs.append(("opt", m.group(1)))
        pos = m.end()
    segs.append(("core", template[pos:]))
    assert "[" not in "".join(s for k, s in segs if k == "core") and "]" not in "".join(s for k, s in segs), template
    return segs


def realise(segs, keep):
    out, i = [], 0
    for kind, text in segs:
        if kind == "core":
            out.append(text)
        else:
            if keep[i]:
                out.append(text if text.startswith((",", ";")) else " " + text + " ")
            i += 1
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    s = re.sub(r"\s+([,.;])", r"\1", s)
    return s[:1].upper() + s[1:]


def load_bank(path):
    bank = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        segs = parse(line.strip())
        k = sum(1 for kind, _ in segs if kind == "opt")
        by_len = {}
        for keep in itertools.product((0, 1), repeat=k):
            s = realise(segs, keep)
            by_len.setdefault(len(s.split()), set()).add(s)
        bank.append({"template": line.strip(), "by_len": {n: sorted(v) for n, v in by_len.items()}})
    return bank


def test_vocab():
    rows = [json.loads(x) for x in (ROOT / "results/SRC1011_NONE_QWEN.episodes.jsonl").read_text().splitlines() if x.strip()]
    texts = {int(r["task_id"]): r["task_text"] for r in rows}
    assert sorted(texts) == list(range(134)), "need the task text of all 134 test games"
    return {w for t in texts.values() for w in words(t)}


def overlap_share(bodies, vocab):
    toks = [w for b in bodies for w in words(b) if w not in STOP]
    return round(sum(w in vocab for w in toks) / max(1, len(toks)), 4)


def match(cands, rng):
    """Kuhn's augmenting-path matching of items to distinct templates, in a random order."""
    owner = {}

    def augment(i, seen):
        for j in cands[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    order = list(range(len(cands)))
    rng.shuffle(order)
    for i in order:
        if not augment(i, set()):
            return None
    return {i: j for j, i in owner.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--bank", type=Path, default=ROOT / "data/pools/poolrep_gardening_bank.txt")
    ap.add_argument("--out-pool", type=Path, required=True)
    ap.add_argument("--out-bodies", type=Path, required=True)
    ap.add_argument("--out-audit", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--domain", default="vegetable gardening", help="label recorded in the audit (W1-B adds amateur astronomy)")
    ap.add_argument("--search", type=int, default=3000, help="random matchings/realisations tried")
    ap.add_argument("--max-tries", type=int, default=10, help="nonce orders tried per candidate")
    ap.add_argument("--candidates", type=int, default=40, help="candidates (least excess first) given to the nonce pass")
    ap.add_argument("--climb-starts", type=int, default=6, help="best random candidates refined by hill-climbing")
    ap.add_argument("--climb", type=int, default=1500, help="single-sentence swaps tried per start")
    a = ap.parse_args()
    for p in (a.out_pool, a.out_bodies, a.out_audit):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    pool = alf.load_jsonl(a.pool)
    seed = a.seed + zlib.crc32(a.pool.name.encode()) % 10000
    rng = random.Random(seed)
    vocab = test_vocab()
    banned = (alf.FORBIDDEN | ALFWORLD_NAMES | ACTION_WORDS | (vocab - STOP))
    bank = load_bank(a.bank)
    bad = sorted({(w, b["template"][:50]) for b in bank for ss in b["by_len"].values() for s in ss for w in words(s) if w in banned})
    if bad:
        sys.exit("banned words in the bank:\n" + "\n".join(f"  {w!r} in {t}..." for w, t in bad))
    tags, lens = [], []
    for it in pool:
        w = it["text"].split()
        assert w[0] == f"[{it['task_type']}]", f"unexpected tag {w[0]}"
        tags.append(w[0])
        lens.append(len(w) - 1)
    cands = [[j for j, b in enumerate(bank) if n in b["by_len"]] for n in lens]
    uncovered = sorted({n for n, c in zip(lens, cands) if not c})
    if uncovered:
        sys.exit(f"no template realises body lengths {uncovered}; add templates to {a.bank}")
    original_context = "\n\n".join(it["text"] for it in pool)
    target, qtarget = count_tokens(original_context, "gpt-4o-mini"), alf.qwen_count(original_context)
    # o200k tokens per word of each template; each attempt orders an item's candidates by density plus noise of a
    # random spread, so attempts range from "densest templates first" to "random" and their footprints straddle the target
    dens = [sum(count_tokens(s, "gpt-4o-mini") / n for n, ss in t["by_len"].items() for s in ss) /
            sum(len(ss) for ss in t["by_len"].values()) for t in bank]
    cands_found, seen, n_matched = [], set(), 0
    for _ in range(a.search):
        spread = 10 ** rng.uniform(-1.5, 1.5)  # 0.03 (densest first) .. 30 (effectively random)
        c = [sorted(x, key=lambda j: -(dens[j] + rng.uniform(0.0, spread))) for x in cands]
        m = match(c, rng)
        if m is None:
            continue
        n_matched += 1
        bodies = [rng.choice(bank[m[i]]["by_len"][lens[i]]) for i in range(len(pool))]
        key = tuple(bodies)
        if key in seen:
            continue
        seen.add(key)
        docs = [[tags[i], *bodies[i].split()] for i in range(len(pool))]
        g, q = alf.both_counts(docs)
        # skip one-sided excess (0 under one tokenizer only): a nonce rarely shortens one tokenizer and not the other
        if g >= target and q >= qtarget and ((g == target) == (q == qtarget)):
            cands_found.append(((g - target) + (q - qtarget), bodies, [m[i] for i in range(len(pool))], (g, q)))
    if n_matched == 0:
        sys.exit("no one-to-one matching of templates to items exists; add templates of the crowded lengths")
    if not cands_found:
        sys.exit(f"every candidate control is shorter than the true pool (gpt {target}, qwen {qtarget}); lengthen the bank")
    cands_found.sort(key=lambda x: x[0])

    def feasible(g, q):
        return g >= target and q >= qtarget and ((g == target) == (q == qtarget))

    def climb(bodies, used, g, q):
        """Swap one item's sentence (another unused template, or another realisation) at a time; keep a swap when the
        control stays at or above both targets and the total excess falls."""
        bodies, used = list(bodies), list(used)
        for _ in range(a.climb):
            i = rng.randrange(len(pool))
            j = rng.choice([x for x in cands[i] if x == used[i] or x not in used])
            nb = rng.choice(bank[j]["by_len"][lens[i]])
            if nb == bodies[i]:
                continue
            trial = bodies[:i] + [nb] + bodies[i + 1:]
            g2, q2 = alf.both_counts([[tags[k], *trial[k].split()] for k in range(len(pool))])
            if feasible(g2, q2) and (g2 - target) + (q2 - qtarget) < (g - target) + (q - qtarget):
                bodies, used[i], g, q = trial, j, g2, q2
        return ((g - target) + (q - qtarget), bodies, used, (g, q))

    cands_found = sorted((climb(b_, u_, g_, q_) for _, b_, u_, (g_, q_) in cands_found[:a.climb_starts]), key=lambda x: x[0])
    # least excess first; the nonce pass (as in build_irrelevant_control.py) must reach BOTH targets exactly
    nrng, chosen, tries, tried_candidates = random.Random(a.seed), None, 0, 0
    for excess, bodies, used, (pre_g, pre_q) in cands_found[:a.candidates]:
        tried_candidates += 1
        docs = [[tags[i], *bodies[i].split()] for i in range(len(pool))]
        base = [(d, k) for d in range(len(docs)) for k in range(1, len(docs[d]))]
        for t in range(a.max_tries):
            order = list(base) if t == 0 else nrng.sample(base, len(base))
            adj, (g, q), reps = alf.greedy_adjust(docs, target, qtarget, order)
            tries += 1
            if g == target and q == qtarget:
                chosen = (adj, reps, bodies, used, pre_g, pre_q)
                break
        if chosen:
            break
    if chosen is None:
        sys.exit(f"footprint not matched after {tried_candidates} candidates x {a.max_tries} nonce orders")
    adj, reps, bodies, used, pre_g, pre_q = chosen
    g, q = target, qtarget
    control = [{**it, "text": " ".join(adj[i])} for i, it in enumerate(pool)]
    for s, c in zip(pool, control):
        assert len(s["text"].split()) == len(c["text"].split()) and s["task_type"] == c["task_type"]
    assert count_tokens("\n\n".join(c["text"] for c in control), "gpt-4o-mini") == target
    assert alf.qwen_count("\n\n".join(c["text"] for c in control)) == qtarget
    body_words = sum(lens)
    a.out_pool.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in control))
    a.out_bodies.write_text("".join(b + "\n" for b in bodies))
    audit = {"pool": str(a.pool.relative_to(ROOT)) if a.pool.is_absolute() else str(a.pool), "control": str(a.out_pool),
             "domain": a.domain, "bank": str(a.bank.relative_to(ROOT)) if a.bank.is_absolute() else str(a.bank),
             "seed": seed, "n_items": len(control), "body_words": body_words,
             "distinct_templates": len(set(used)), "matchings_found": n_matched, "search": a.search,
             "candidates_tried": tried_candidates, "climb_starts": a.climb_starts, "climb_steps": a.climb,
             "nonce_replacements": len(reps), "retained_body_fraction": round(1 - len(reps) / body_words, 4), "tries": tries,
             "context_tokens": {"gpt_4o_mini_o200k": {"target": target, "before": pre_g, "control": g},
                                "qwen3_8b": {"target": qtarget, "before": pre_q, "control": q}},
             "both_tokenizers_match": True,
             "test_game_vocab_overlap": {"true_pool": overlap_share([" ".join(it["text"].split()[1:]) for it in pool], vocab),
                                         "control": overlap_share([" ".join(c["text"].split()[1:]) for c in control], vocab)},
             "retrieval_control": {"kind": "index-aligned reference ranking", "ranking_corpus": str(a.pool),
                                   "payload_corpus": str(a.out_pool)},
             "templates_used": used, "replacements": reps,
             "examples": [{"original": pool[i]["text"], "control": control[i]["text"]} for i in range(min(4, len(pool)))]}
    a.out_audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("replacements", "examples", "templates_used")}, indent=2))


if __name__ == "__main__":
    main()
