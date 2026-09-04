# ================================================================================
# COMPLETE PIPELINE - single cell containing setup + injection + broker survey
# ================================================================================
# Toggles at top control what actually runs.
#
# Phase 1  Setup (install, load model, J-lens, SAE, auth)         ~5 min
# Phase 2  Helpers (Neuronpedia API, SAE, J-lens, injection hook) instant
# Phase 3  Injection experiments (11 polysemes x 5 alphas x 3 conds)  ~30-45 min
# Phase 4  Broker survey (12 unresolved prompts, feature search)  ~15-25 min
# Phase 5  Summaries                                              instant
#
# Total runtime with all phases: ~60 min. Runtime with just broker survey: ~20 min.
# ================================================================================

# ---- TOGGLES ----
RUN_INJECTION           = False   # True to re-run injection experiments (we have data)
RUN_BROKER_SURVEY       = True    # True to run the NEW broker survey
SKIP_PREVIOUSLY_TESTED  = True    # Injection: skip already-tested polysemes

# ================================================================================
# PHASE 1 - Setup
# ================================================================================
import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "git+https://github.com/anthropics/jacobian-lens.git",
                        "sae_lens"])

import torch, gc, math, time, requests
import transformers, jlens
from sae_lens import SAE
from google.colab import userdata
from huggingface_hub import login

try:
    del model, hf_model, lens, sae
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM at start: {torch.cuda.memory_allocated()/1e9:.2f} GB")
login(token=userdata.get("HF_TOKEN"))

print("\n[Model] Loading Gemma-2-2B-IT...")
hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    dtype=torch.bfloat16,
    attn_implementation="eager",
).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = jlens.from_hf(hf_model, tokenizer)
print(f"[Model] Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\n[J-lens] Loading pre-fit from Neuronpedia...")
lens = jlens.JacobianLens.from_pretrained(
    "neuronpedia/jacobian-lens",
    filename="gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt",
)

print("\n[SAE] Loading Gemma Scope L20 (16k)...")
try:
    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_20/width_16k/average_l0_71",
    )
except Exception:
    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_20/width_16k/average_l0_74",
    )
sae = sae.to("cuda").to(torch.bfloat16)
print(f"[SAE] Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

LAYER_HOOK = 20
LAYER_SAE_ID = "20-gemmascope-res-16k"

# ================================================================================
# PHASE 2 - Helpers
# ================================================================================

NP_BASE = "https://www.neuronpedia.org/api"
NP_MODEL = "gemma-2-2b"

def np_feature(feat_idx):
    url = f"{NP_BASE}/feature/{NP_MODEL}/{LAYER_SAE_ID}/{feat_idx}"
    return requests.get(url, timeout=15).json()

def np_max_act(feat_idx):
    try:
        return np_feature(feat_idx).get("maxActApprox", 100.0)
    except Exception:
        return 100.0

def np_label(feat_idx):
    try:
        exps = np_feature(feat_idx).get("explanations", [])
        return exps[-1]["description"][:60] if exps else "(no label)"
    except Exception:
        return "(lookup failed)"

def np_search(query, top_k=3):
    url = f"{NP_BASE}/explanation/search-model"
    body = {"modelId": NP_MODEL, "layers": [LAYER_SAE_ID], "query": query}
    try:
        r = requests.post(url, json=body, timeout=20).json()
        matches = [x for x in r.get("results", []) if x.get("layer") == LAYER_SAE_ID]
        return matches[:top_k]
    except Exception:
        return []

def np_search_fallback(queries, top_k=3):
    for q in queries:
        matches = np_search(q, top_k=top_k)
        if matches:
            return matches, q
    return None, None

def sae_activations(prompt):
    """Full SAE activation vector at last input token. No hooks active."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    cap = {}
    def h(m, i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()
    handle = hf_model.model.layers[LAYER_HOOK].register_forward_hook(h)
    with torch.no_grad():
        _ = hf_model(input_ids)
    handle.remove()
    last_resid = cap["h"][0, -1, :]
    return sae.encode(last_resid.unsqueeze(0).to(sae.dtype))[0]

def sae_readout_current(prompt, top_k=15):
    """SAE activations at last token, respects any registered hooks."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    cap = {}
    def h(m, i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()
    handle = hf_model.model.layers[LAYER_HOOK].register_forward_hook(h)
    with torch.no_grad():
        _ = hf_model(input_ids)
    handle.remove()
    last_resid = cap["h"][0, -1, :]
    feature_acts = sae.encode(last_resid.unsqueeze(0).to(sae.dtype))[0]
    top = torch.topk(feature_acts, top_k)
    n_active = (feature_acts > 0).sum().item()
    return {"feature_acts": feature_acts, "n_active": n_active,
            "top": [(idx.item(), act.item()) for idx, act in zip(top.indices, top.values)]}

def jlens_top_str(prompt, layer=LAYER_HOOK, top_k=5):
    jlens_logits, _, _ = lens.apply(model, prompt, layers=[layer], positions=[-1])
    probs = torch.softmax(jlens_logits[layer][0].float(), dim=-1)
    top = torch.topk(probs, top_k)
    return " ".join(f"{tokenizer.decode([idx.item()])!r}" for idx in top.indices)

def make_injection_hook(direction, alpha):
    d = direction.detach().to(torch.bfloat16).cuda()
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0] + alpha * d
            return (h,) + output[1:]
        return output + alpha * d
    return hook

def rank_of_feature(feat_acts, feat_idx):
    return (feat_acts > feat_acts[feat_idx]).sum().item() + 1

def classify(act):
    if act >= 5.0: return "BROKER"
    if act >= 1.0: return "weak-latent"
    return "not-present"

# ================================================================================
# PHASE 3 - Injection experiments (conditional)
# ================================================================================

if RUN_INJECTION:
    print("\n" + "=" * 78)
    print(f"PHASE 3 - INJECTION EXPERIMENTS  (SKIP_PREVIOUSLY_TESTED={SKIP_PREVIOUSLY_TESTED})")
    print("=" * 78)

    PREVIOUSLY_TESTED = {
        "apple":   {"prompt": "The Granny Smith apple is rotten.",       "target_queries": None, "target_feature": 16331, "design": "A"},
        "crane":   {"prompt": "She watched the construction crane.",     "target_queries": None, "target_feature": 8002,  "design": "A"},
        "bat":     {"prompt": "The bat is quick.",                       "target_queries": None, "target_feature": 14694, "design": "B"},
        "bass":    {"prompt": "The bass slipped from her grasp.",        "target_queries": None, "target_feature": 718,   "design": "B"},
        "jam":     {"prompt": "She spent the whole afternoon fixing the jam.", "target_queries": None, "target_feature": 9643, "design": "B"},
        "coach":   {"prompt": "The coach took them straight to the game.", "target_queries": None, "target_feature": 8027, "design": "B"},
    }
    NEW_POLYSEMES = {
        "mole":    {"prompt": "The mole had been underground for months.",       "target_queries": ["spy", "espionage", "undercover"],        "target_feature": None, "design": "B"},
        "vessel":  {"prompt": "They found a deep gouge in the vessel.",          "target_queries": ["blood vessel", "artery", "vein"],        "target_feature": None, "design": "B"},
        "fans":    {"prompt": "The fans started making noise.",                  "target_queries": ["cooling", "ventilation", "electric fan"], "target_feature": None, "design": "B"},
        "chest":   {"prompt": "She watched the chest carefully for any movement.", "target_queries": ["storage box", "treasure chest", "wooden box"], "target_feature": None, "design": "B"},
        "deposit": {"prompt": "He put a heavy deposit in front of them.",        "target_queries": ["sediment", "mineral", "geology"],        "target_feature": None, "design": "B"},
        "club":    {"prompt": "The club had to be replaced after last night.",   "target_queries": ["golf", "weapon", "cudgel"],              "target_feature": None, "design": "B"},
        "scale":   {"prompt": "She couldn't stand the sight of the scale.",      "target_queries": ["weighing", "balance", "kilogram"],       "target_feature": None, "design": "B"},
        "lead":    {"prompt": "The lead was not as good as expected.",           "target_queries": ["lead metal", "pencil", "poisoning"],     "target_feature": None, "design": "B"},
        "net":     {"prompt": "They spent all morning fixing the net.",          "target_queries": ["fishing net", "trawl", "mesh"],          "target_feature": None, "design": "B"},
        "spring":  {"prompt": "The spring burst suddenly in the middle of the room.", "target_queries": ["water spring", "geyser", "natural spring"], "target_feature": None, "design": "B"},
        "key":     {"prompt": "He tried to change the key before anyone noticed.", "target_queries": ["musical key", "piano", "sharp flat"],  "target_feature": None, "design": "B"},
    }
    INJ_POLYSEMES = {}
    if not SKIP_PREVIOUSLY_TESTED:
        INJ_POLYSEMES.update(PREVIOUSLY_TESTED)
    INJ_POLYSEMES.update(NEW_POLYSEMES)

    # Discovery
    print("\n-- DISCOVERY --")
    for name, spec in INJ_POLYSEMES.items():
        if spec["target_feature"] is not None:
            print(f"  [{name:8s}] pre-picked F{spec['target_feature']}")
            continue
        matches, used_q = np_search_fallback(spec["target_queries"])
        if not matches:
            print(f"  [{name:8s}] ALL QUERIES FAILED: {spec['target_queries']}")
            continue
        best = matches[0]
        spec["target_feature"] = int(best["index"])
        spec["_used_query"] = used_q
        print(f"  [{name:8s}] {used_q!r:25s} -> F{spec['target_feature']} ({best.get('description','?')[:50]})")

    # Verification
    print("\n-- VERIFICATION --")
    for name, spec in INJ_POLYSEMES.items():
        f_target = spec.get("target_feature")
        if f_target is None:
            print(f"  [{name:8s}] (skipped, no target)")
            continue
        label = np_label(f_target)
        max_act = np_max_act(f_target)
        feat_acts = sae_activations(spec["prompt"])
        baseline_act = feat_acts[f_target].item()
        if baseline_act >= 5.0:
            cls, alpha_unit = "broker", max(baseline_act, max_act * 0.25)
        elif baseline_act >= 1.0:
            cls, alpha_unit = "weak-latent", max(baseline_act, max_act * 0.25)
        else:
            cls, alpha_unit = "insertion", max_act * 0.3
        spec.update({"baseline_act": baseline_act, "classification": cls,
                      "alpha_unit": alpha_unit, "label": label, "max_act": max_act})
        print(f"  [{name:8s}] F{f_target:>5} act={baseline_act:>6.2f} max={max_act:>6.1f} class={cls:12s} au={alpha_unit:.1f}")

    # Injection
    print("\n-- INJECTION --")
    ALPHAS_MULT = [0.0, 1.0, 2.0, 5.0, 10.0]
    RANDOM_FEAT = 7000
    for name, spec in INJ_POLYSEMES.items():
        if spec.get("target_feature") is None or spec.get("alpha_unit") is None:
            continue
        f_target = spec["target_feature"]
        alpha_unit = spec["alpha_unit"]
        prompt = spec["prompt"]
        print("\n" + "=" * 78)
        print(f"POLYSEME: {name}  design={spec['design']}  class={spec['classification']}")
        print(f"  Prompt: {prompt!r}")
        print(f"  Target: F{f_target} ({spec['label']})  base={spec['baseline_act']:.2f}  au={alpha_unit:.1f}")
        print("=" * 78)
        target_dir = sae.W_dec[f_target].detach().clone()
        target_norm = target_dir.norm().item()
        random_dir = sae.W_dec[RANDOM_FEAT].detach().clone()
        random_dir = random_dir * (target_norm / random_dir.norm())
        torch.manual_seed(43)
        noise_dir = torch.randn_like(target_dir)
        noise_dir = noise_dir * (target_norm / noise_dir.norm())
        for cond_name, direction in [("target", target_dir), (f"F{RANDOM_FEAT}_rand", random_dir), ("noise", noise_dir)]:
            for mult in ALPHAS_MULT:
                alpha = mult * alpha_unit
                handle = hf_model.model.layers[LAYER_HOOK].register_forward_hook(
                    make_injection_hook(direction, alpha))
                try:
                    sae_out = sae_readout_current(prompt, top_k=8)
                    f_target_act = sae_out["feature_acts"][f_target].item()
                    jl = jlens_top_str(prompt, top_k=5)
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
                    with torch.no_grad():
                        gen = hf_model.generate(input_ids, max_new_tokens=30,
                                                do_sample=False,
                                                pad_token_id=tokenizer.eos_token_id)
                    gen_text = tokenizer.decode(gen[0][input_ids.shape[1]:],
                                                skip_special_tokens=True).replace("\n", " ")[:120]
                finally:
                    handle.remove()
                print(f"  {cond_name:>14s} alpha={mult:>4.1f}x ({alpha:>6.1f}): "
                      f"F{f_target}={f_target_act:>6.1f} n={sae_out['n_active']:>3d}  "
                      f"jl: {jl}  |  gen: {gen_text!r}")
else:
    print("\n[Phase 3] Injection SKIPPED (RUN_INJECTION=False)")

# ================================================================================
# PHASE 4 - Broker survey (feature-specific alt-sense discovery)
# ================================================================================

if RUN_BROKER_SURVEY:
    print("\n\n" + "=" * 78)
    print("PHASE 4 - BROKER SURVEY (feature-specific alt-sense search)")
    print("=" * 78)

    BROKER_SURVEY_POLYSEMES = {
        "bow":     {"prompt": "The bow was held tightly in their hand.",
                    "queries": {"violin/music": ["violin", "musical instrument", "orchestra"],
                                "hair-ribbon": ["ribbon", "hair accessory"]}},
        "trunk":   {"prompt": "She couldn't get it off the trunk.",
                    "queries": {"tree": ["tree", "forest", "bark"],
                                "elephant": ["elephant", "mammal"]}},
        "crane2":  {"prompt": "The crane moved slowly toward it.",
                    "queries": {"bird": ["bird", "ornithology", "poultry"]}},
        "bank":    {"prompt": "He left the bank after an intense hour.",
                    "queries": {"financial": ["finance", "money", "banking"],
                                "river":     ["river", "riverbank", "shore"]}},
        "pitch":   {"prompt": "It was a sharp pitch coming from the back.",
                    "queries": {"music": ["music", "tone", "melody"],
                                "tar":   ["tar", "asphalt"],
                                "slope": ["slope", "gradient"]}},
        "current": {"prompt": "The current was far too strong to manage.",
                    "queries": {"electrical": ["electricity", "electric", "voltage"]}},
        "match":   {"prompt": "He dropped the match before it ended.",
                    "queries": {"fire": ["fire", "flame", "matchstick", "lighter"]}},
        "mole":    {"prompt": "The mole had been underground for months.",
                    "queries": {"spy": ["spy", "espionage", "undercover"]}},
        "vessel":  {"prompt": "They found a deep gouge in the vessel.",
                    "queries": {"blood": ["blood", "artery", "vein"],
                                "cup":   ["cup", "chalice", "drinking"]}},
        "chest":   {"prompt": "She watched the chest carefully for any movement.",
                    "queries": {"body":    ["torso", "body part"],
                                "storage": ["treasure", "wooden box", "storage"]}},
        "deposit": {"prompt": "He put a heavy deposit in front of them.",
                    "queries": {"geology": ["sediment", "mineral", "geology"]}},
        "spring":  {"prompt": "The spring burst suddenly in the middle of the room.",
                    "queries": {"water":  ["water spring", "geyser"],
                                "season": ["season", "springtime"]}},
    }

    survey_summary = []
    for name, spec in BROKER_SURVEY_POLYSEMES.items():
        prompt = spec["prompt"]
        print("\n" + "-" * 78)
        print(f"[{name}]  {prompt!r}")
        acts = sae_activations(prompt)
        best_feat, best_label, best_act, best_class, best_sense = None, None, 0.0, "not-present", None
        for sense_name, queries in spec["queries"].items():
            print(f"  alt-sense: {sense_name}")
            seen = set()
            for q in queries:
                matches = np_search(q, top_k=3)
                if not matches:
                    print(f"    q={q!r}: no L20 matches")
                    continue
                for m in matches:
                    fid = int(m["index"])
                    if fid in seen: continue
                    seen.add(fid)
                    a = acts[fid].item()
                    lbl = m.get("description", "?")[:50]
                    cls = classify(a)
                    rk = rank_of_feature(acts, fid) if a > 0 else "-"
                    print(f"    F{fid:>6}  q={q!r:22s}  act={a:>6.2f}  rk={str(rk):>4s}  {cls:12s}  {lbl}")
                    if a > best_act:
                        best_feat, best_label, best_act, best_class, best_sense = fid, lbl, a, cls, sense_name
        print(f"  >> BEST: ", end="")
        if best_feat is None or best_act == 0:
            print("none firing")
        else:
            print(f"F{best_feat} ({best_label})  act={best_act:.2f}  class={best_class}  sense={best_sense}")
        survey_summary.append((name, best_feat, best_label, best_act, best_class, best_sense))

    # Phase 5 (survey) - summary
    print("\n\n" + "=" * 78)
    print("BROKER SURVEY SUMMARY - 12 previously-'committed' prompts, alt-sense search")
    print("=" * 78)
    print(f"{'polyseme':10s}  {'best F':>7s}  {'act':>6s}  {'class':12s}  {'sense':15s}  {'label':45s}")
    for name, fid, lbl, act, cls, sense in survey_summary:
        fid_s = f"F{fid}" if fid is not None else "-"
        lbl_s = lbl if lbl is not None else "-"
        sense_s = sense if sense is not None else "-"
        print(f"{name:10s}  {fid_s:>7s}  {act:>6.2f}  {cls:12s}  {sense_s:15s}  {lbl_s}")
    n_broker = sum(1 for r in survey_summary if r[4] == "BROKER")
    n_weak = sum(1 for r in survey_summary if r[4] == "weak-latent")
    n_none = sum(1 for r in survey_summary if r[4] == "not-present")
    print(f"\nCounts: {n_broker} BROKER  |  {n_weak} weak-latent  |  {n_none} not-present")
    prior_brokers = ["apple", "bat", "bass_v2", "crane", "coach", "jam", "club", "net"]
    prior_non = ["bass_v1", "fans", "key"]
    print(f"\nCombined with prior injection data (24 total ambig prompts):")
    print(f"  Confirmed brokers:      {len(prior_brokers) + n_broker}")
    print(f"  Confirmed weak-latents: {n_weak}")
    print(f"  Confirmed non-brokers:  {len(prior_non) + n_none}")
else:
    print("\n[Phase 4] Broker survey SKIPPED (RUN_BROKER_SURVEY=False)")

print("\n\nDone.")
