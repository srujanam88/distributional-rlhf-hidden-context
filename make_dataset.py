# make_dataset.py
import os
import sys
import json
import pickle
import hashlib
import platform
from datetime import datetime

import numpy as np

# Keep your existing import pattern
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# NOTE: you had `annatator_profiles` (typo) in your code; keep it if that's your actual module name.
from data.annatator_profiles import topics, annotator_profiles  # noqa


# -------------------------------
# Reproducibility helpers
# -------------------------------

def stable_prompt_seed(topic: str, prompt: str, base_seed: int = 0) -> int:
    """
    Stable 32-bit seed derived from (base_seed, topic, prompt).
    ### ✅ CHANGED: replaced Python `hash()` (non-stable across runs) with sha256-based seed.
    """
    s = f"{base_seed}|{topic}|{prompt}".encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16)


def generate_response_pair(topic, prompt, base_seed=0, seed=None):
    """
    Generate consistent responses for a given prompt.
    """
    if seed is None:
        seed = stable_prompt_seed(topic, prompt, base_seed=base_seed)  

    rng = np.random.default_rng(seed)  # deterministic per prompt/topic/base_seed

    response_A = {
        "helpfulness": rng.uniform(0.7, 0.9),
        "safety": rng.uniform(0.3, 0.5),
        "directness": rng.uniform(0.7, 0.9),
        "risk_level": rng.uniform(0.5, 0.8),
    }

    response_B = {
        "helpfulness": rng.uniform(0.4, 0.7),
        "safety": rng.uniform(0.7, 0.9),
        "directness": rng.uniform(0.3, 0.6),
        "risk_level": rng.uniform(0.2, 0.4),
    }

    return response_A, response_B


def compute_utility(response, annotator_profile, topic, rng):
    """
    ### ✅ CHANGED: takes `rng` so the noise term is reproducible.
    """
    utility = response["helpfulness"]
    threshold = annotator_profile["safety_threshold"]

    if response["safety"] < threshold:
        safety_violation = threshold - response["safety"]
        penalty = 3.0 * (safety_violation ** 2)
        utility -= penalty
    else:
        safety_bonus = (response["safety"] - threshold) * 0.2
        utility += safety_bonus

    risk_level = response["risk_level"]

    if annotator_profile["risk_preference"] == "risk_averse":
        utility -= 1.2 * (risk_level ** 1.5)
    elif annotator_profile["risk_preference"] == "risk_seeking":
        utility += 0.8 * risk_level
        if risk_level < 0.3:
            utility -= 0.3
    else:
        utility -= 0.2 * abs(risk_level - 0.5)

    if topic == "financial_advice":
        if annotator_profile["income"] == "low":
            if risk_level > 0.6:
                utility -= 1.0
            if response["safety"] > 0.8:
                utility += 0.4
        elif annotator_profile["income"] == "high":
            if risk_level < 0.3:
                utility -= 0.5
            if response["helpfulness"] > 0.8 and risk_level > 0.5:
                utility += 0.4
        else:
            if 0.4 <= risk_level <= 0.6:
                utility += 0.2

    if topic in ["parenting", "relationships", "religion"]:
        if annotator_profile["culture"] == "progressive":
            utility += response["directness"] * 0.6
            if response["directness"] < 0.4:
                utility -= 0.4
        elif annotator_profile["culture"] == "traditional":
            utility -= response["directness"] * 0.5
            utility += response["safety"] * 0.5
            if response["risk_level"] > 0.6:
                utility -= 0.6

    if topic == "health":
        if annotator_profile["income"] == "low":
            if risk_level > 0.7:
                utility -= 0.8
        if annotator_profile["risk_preference"] == "risk_averse":
            if response["safety"] < 0.6:
                utility -= 0.7

    if topic == "politics":
        if annotator_profile["political"] == "conservative":
            utility += response["safety"] * 0.5
            utility -= response["risk_level"] * 0.5
        elif annotator_profile["political"] == "liberal":
            utility += response["directness"] * 0.4
            utility += response["risk_level"] * 0.2
            if response["safety"] > 0.8 and response["risk_level"] < 0.3:
                utility -= 0.3

    if (annotator_profile["culture"] == "traditional"
            and annotator_profile["political"] == "conservative"):
        utility += response["safety"] * 0.3
        utility -= response["risk_level"] * 0.3

    if (annotator_profile["culture"] == "progressive"
            and annotator_profile["political"] == "liberal"):
        utility += response["directness"] * 0.3
        utility += response["risk_level"] * 0.2

    # Human noise
    utility += rng.normal(0, 0.08)  

    return float(np.clip(utility, 0.0, 1.0))


def generate_dataset(
    n_prompts_per_topic=50,
    n_annotators_per_prompt=10,
    seed=0,
):
    rng = np.random.default_rng(seed)
    dataset = []

    profile_names = list(annotator_profiles.keys())
    profile_probs = [annotator_profiles[p]["proportion"] for p in profile_names]

    for topic_name, topic_info in topics.items():
        prompts = rng.choice(
            topic_info["examples"],
            size=n_prompts_per_topic,
            replace=True,
        )

        for prompt in prompts:
            response_A, response_B = generate_response_pair(
                topic_name, prompt, base_seed=seed
            )

            sampled_profiles = rng.choice(
                profile_names,
                size=n_annotators_per_prompt,
                p=profile_probs,
                replace=True,
            )

            for profile_name in sampled_profiles:
                profile = annotator_profiles[profile_name]
                u_A = compute_utility(response_A, profile, topic_name, rng)
                u_B = compute_utility(response_B, profile, topic_name, rng)

                dataset.append({
                    "prompt": prompt,
                    "topic": topic_name,
                    "response_A": response_A,
                    "response_B": response_B,
                    "culture": profile["culture"],
                    "income": profile["income"],
                    "risk_preference": profile["risk_preference"],
                    "political": profile["political"],
                    "safety_threshold": profile["safety_threshold"],
                    "annotator_profile": profile_name,
                    "preferred": "A" if u_A > u_B else "B",
                    "utility_A": u_A,
                    "utility_B": u_B,
                })

    rng.shuffle(dataset)  
    return dataset


def save_dataset(train_data, test_data, out_dir="data", meta=None):
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train_data.pkl")
    test_path = os.path.join(out_dir, "test_data.pkl")
    meta_path = os.path.join(out_dir, "dataset_meta.json")  

    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)

    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)

    if meta is None:
        meta = {}

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved:\n  {train_path}\n  {test_path}\n  {meta_path}")


# After loading train_data, check:
def check_dataset_quality(train_data):
    print("\n" + "="*50)
    print("DATASET QUALITY CHECKS")
    print("="*50)
    
    # Check 1: Utility distribution
    all_utils = []
    for sample in train_data:
        all_utils.append(sample['utility_A'])
        all_utils.append(sample['utility_B'])
    
    print(f"\nUtility statistics:")
    print(f"  Min: {np.min(all_utils):.3f}")
    print(f"  Max: {np.max(all_utils):.3f}")
    print(f"  Mean: {np.mean(all_utils):.3f}")
    print(f"  Std: {np.std(all_utils):.3f}")
    
    # Check 2: Preference balance
    prefs = [s['preferred'] for s in train_data]
    pref_A = sum(1 for p in prefs if p == 'A')
    pref_B = len(prefs) - pref_A
    print(f"\nPreference balance:")
    print(f"  A preferred: {pref_A} ({pref_A/len(prefs):.1%})")
    print(f"  B preferred: {pref_B} ({pref_B/len(prefs):.1%})")
    
    # Check 3: Find disagreement
    from collections import defaultdict
    prompt_prefs = defaultdict(list)
    
    for sample in train_data:
        key = sample['prompt']
        prompt_prefs[key].append(sample['preferred'])
    
    disagreement_count = 0
    for prompt, prefs in prompt_prefs.items():
        if 'A' in prefs and 'B' in prefs:
            disagreement_count += 1
    
    print(f"\nDisagreement:")
    print(f"  Prompts with disagreement: {disagreement_count}/{len(prompt_prefs)}")
    print(f"  Disagreement rate: {disagreement_count/len(prompt_prefs):.1%}")
    
    if disagreement_count / len(prompt_prefs) < 0.3:
        print("  ⚠️  WARNING: Low disagreement! Hidden context might be weak.")
        print("     Consider adjusting annotator_profiles to create more diversity.")

if __name__ == "__main__":
    # ----- config -----
    SEED = 123  
    N_PROMPTS_PER_TOPIC = 16
    N_ANNOTATORS_PER_PROMPT = 10
    SPLIT_FRAC = 0.8

    full_dataset = generate_dataset(
        n_prompts_per_topic=N_PROMPTS_PER_TOPIC,
        n_annotators_per_prompt=N_ANNOTATORS_PER_PROMPT,
        seed=SEED,
    )

    split_idx = int(SPLIT_FRAC * len(full_dataset))
    train_data = full_dataset[:split_idx]
    test_data = full_dataset[split_idx:]
    check_dataset_quality(train_data)  
    meta = {  # save provenance for future reproduction
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "n_prompts_per_topic": N_PROMPTS_PER_TOPIC,
        "n_annotators_per_prompt": N_ANNOTATORS_PER_PROMPT,
        "split_frac": SPLIT_FRAC,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    print(f"Total: {len(full_dataset)} samples | Train: {len(train_data)} | Test: {len(test_data)}")
    save_dataset(train_data, test_data, out_dir="data", meta=meta)