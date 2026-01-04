# train.py
import os
import sys
import json
import random
import pickle
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from hidden_context.synthetic_exps import (  # noqa
    BaseRewardModel,
    MeanAndVarianceRewardModel,
    CategoricalRewardModel
)

state_dim = 4

def seed_everything(seed: int):
    """
    One place to make training deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism knobs (may reduce speed)
    torch.use_deterministic_algorithms(True)  # may throw if something non-deterministic is used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def response_to_tensor(response):
    return torch.tensor([
        response["helpfulness"],
        response["safety"],
        response["directness"],
        response["risk_level"],
    ], dtype=torch.float32).unsqueeze(0)


def train_rlhf_with_dataset(
    reward_model,
    train_data,
    batch_size=128,
    lr=1e-4,
    num_iterations=1000,
    device="cpu",
    entropy_weight=0.0,
    seed=0,  
):
    dataset_size = len(train_data)
    if batch_size > dataset_size:
        batch_size = max(1, dataset_size // 2)
        print(f"⚠️  Adjusted batch_size to {batch_size} (dataset has {dataset_size} samples)")

    optimizer = optim.Adam(reward_model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=(1e-5 / lr) ** (1 / num_iterations))

    reward_model.to(device).train()

    # Convert dataset to tensors ONCE
    states0, states1, preferences = [], [], []
    for sample in train_data:
        state0 = torch.tensor([
            sample["response_A"]["helpfulness"],
            sample["response_A"]["safety"],
            sample["response_A"]["directness"],
            sample["response_A"]["risk_level"],
        ], dtype=torch.float32)

        state1 = torch.tensor([
            sample["response_B"]["helpfulness"],
            sample["response_B"]["safety"],
            sample["response_B"]["directness"],
            sample["response_B"]["risk_level"],
        ], dtype=torch.float32)

        pref = 0 if sample["preferred"] == "A" else 1
        states0.append(state0)
        states1.append(state1)
        preferences.append(pref)

    states0 = torch.stack(states0).to(device)
    states1 = torch.stack(states1).to(device)
    preferences = torch.tensor(preferences, dtype=torch.long).to(device)

    # deterministic torch Generator for batch sampling
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    progress_bar = tqdm.tqdm(range(num_iterations))

    for iteration in progress_bar:
        indices = torch.randint(
            0, dataset_size, (batch_size,),
            generator=g, device=device  
        )

        batch_states0 = states0[indices]
        batch_states1 = states1[indices]
        batch_prefs = preferences[indices]

        optimizer.zero_grad()
        loss = -reward_model.preference_logp(
            batch_states0,
            batch_states1,
            batch_prefs
        ).mean()

        if entropy_weight > 0 and isinstance(reward_model, MeanAndVarianceRewardModel):
            sample_indices = torch.randint(
                0, dataset_size, (min(batch_size, 64),),
                generator=g, device=device  
            )
            sample_states = states0[sample_indices]
            outputs = reward_model(sample_states)
            log_stds = outputs[:, 1]
            entropy = (0.5 * np.log(2 * np.pi * np.e) + log_stds).mean()
            loss = loss - entropy_weight * entropy

        loss.backward()
        optimizer.step()
        scheduler.step()

        if isinstance(reward_model, MeanAndVarianceRewardModel):
            with torch.no_grad():
                sample_output = reward_model(states0[:10])
                mean_std = torch.exp(sample_output[:, 1]).mean().item()
            progress_bar.set_description(
                f"loss={loss.item():.2f} lr={scheduler.get_last_lr()[0]:.2e} std={mean_std:.3f}"
            )
        else:
            progress_bar.set_description(
                f"loss={loss.item():.2f} lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return reward_model


def evaluate_model(model, test_data, model_type="standard", device="cpu", quantile=0.01):
    model.eval()
    model.to(device)

    cultures = ["progressive", "moderate", "traditional"]
    regrets_by_culture = {c: [] for c in cultures}
    violations_by_culture = {c: {"violations": 0, "total": 0} for c in cultures}

    with torch.no_grad():
        for sample in test_data:
            culture = sample["culture"]
            tensor_a = response_to_tensor(sample["response_A"]).to(device)
            tensor_b = response_to_tensor(sample["response_B"]).to(device)

            if model_type == "standard":
                u_a = model(tensor_a).item()
                u_b = model(tensor_b).item()
                model_prefers_a = (u_a > u_b)

            elif model_type == "dpl_mean_var":
                output_a = model(tensor_a)
                output_b = model(tensor_b)
                mean_a = output_a[0, 0].item()
                mean_b = output_b[0, 0].item()
                model_prefers_a = (mean_a > mean_b)

            elif model_type == "dpl_mean_var_risk_averse":
                output_a = model(tensor_a)
                output_b = model(tensor_b)
                mean_a = output_a[0, 0].item()
                std_a = np.exp(output_a[0, 1].item())
                mean_b = output_b[0, 0].item()
                std_b = np.exp(output_b[0, 1].item())
                q_a = mean_a + norm.ppf(quantile) * std_a
                q_b = mean_b + norm.ppf(quantile) * std_b
                model_prefers_a = (q_a > q_b)

            elif model_type == "dpl_categorical":
                dist_a = model(tensor_a)[0]
                dist_b = model(tensor_b)[0]
                n_atoms = len(dist_a)
                utility_levels = torch.linspace(0, 1, n_atoms).to(device)
                expected_a = (dist_a * utility_levels).sum().item()
                expected_b = (dist_b * utility_levels).sum().item()
                model_prefers_a = (expected_a > expected_b)

            elif model_type == "dpl_categorical_risk_averse":
                dist_a = model(tensor_a)[0]
                dist_b = model(tensor_b)[0]
                n_atoms = len(dist_a)
                utility_levels = torch.linspace(0, 1, n_atoms).to(device)
                cdf_a = torch.cumsum(dist_a, dim=0)
                cdf_b = torch.cumsum(dist_b, dim=0)
                idx_a = (cdf_a >= quantile).nonzero(as_tuple=True)[0][0]
                idx_b = (cdf_b >= quantile).nonzero(as_tuple=True)[0][0]
                q_a = utility_levels[idx_a].item()
                q_b = utility_levels[idx_b].item()
                model_prefers_a = (q_a > q_b)

            true_u_a = sample["utility_A"]
            true_u_b = sample["utility_B"]

            if model_prefers_a:
                actual_utility = true_u_a
            else:
                actual_utility = true_u_b

            optimal_utility = max(true_u_a, true_u_b)
            regret = optimal_utility - actual_utility
            regrets_by_culture[culture].append(regret)

            chosen_response = sample["response_A"] if model_prefers_a else sample["response_B"]
            threshold = sample["safety_threshold"]

            violations_by_culture[culture]["total"] += 1
            if chosen_response["safety"] < threshold:
                violations_by_culture[culture]["violations"] += 1

    results = {"regret_by_culture": {}, "safety_violations_by_culture": {}}

    for culture in cultures:
        results["regret_by_culture"][culture] = float(np.mean(regrets_by_culture[culture])) if regrets_by_culture[culture] else 0.0
        stats = violations_by_culture[culture]
        results["safety_violations_by_culture"][culture] = (stats["violations"] / stats["total"]) if stats["total"] else 0.0

    if "dpl" in model_type:
        means, variances = [], []
        with torch.no_grad():
            for sample in test_data:
                for resp in [sample["response_A"], sample["response_B"]]:
                    tensor_resp = response_to_tensor(resp).to(device)

                    if "categorical" in model_type:
                        dist = model(tensor_resp)[0]
                        n_atoms = len(dist)
                        utility_levels = torch.linspace(0, 1, n_atoms).to(device)
                        mean = (dist * utility_levels).sum().item()
                        variance = (dist * (utility_levels - mean) ** 2).sum().item()
                    else:
                        output = model(tensor_resp)
                        mean = output[0, 0].item()
                        variance = (np.exp(output[0, 1].item()) ** 2)

                    means.append(mean)
                    variances.append(variance)

        var_of_means = np.var(means)
        mean_of_vars = np.mean(variances)
        results["r_squared"] = float(var_of_means / (var_of_means + mean_of_vars))

    return results


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
    # single training seed
    SEED = 123
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    with open("data/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("data/test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    #load meta to sanity-check seed/config
    meta_path = "data/dataset_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"Loaded dataset meta: seed={meta.get('seed')} prompts/topic={meta.get('n_prompts_per_topic')}")
    check_dataset_quality(train_data)
    # Hyperparameters
    batch_size = 32
    lr = 1e-3
    num_iterations = 500
    entropy_weight = 0.02

    print("Training Standard RLHF Model")
    standard_model = BaseRewardModel(state_dim=4, num_layers=2, hidden_dim=64)
    standard_model = train_rlhf_with_dataset(
        standard_model,
        train_data,
        batch_size=batch_size,
        lr=lr,
        num_iterations=num_iterations,
        device=device,
        seed=SEED,  
    )

    print("Training DPL Mean-Variance Model")
    dpl_mean_var_model = MeanAndVarianceRewardModel(state_dim=4, num_layers=2, hidden_dim=64)
    dpl_mean_var_model = train_rlhf_with_dataset(
        dpl_mean_var_model,
        train_data,
        batch_size=batch_size,
        lr=lr,
        num_iterations=num_iterations,
        device=device,
        entropy_weight=entropy_weight,
        seed=SEED,  
    )

    print("Training DPL Categorical Model")
    dpl_categorical_model = CategoricalRewardModel(
        state_dim=4,
        num_layers=2,
        hidden_dim=64,
        num_atoms=10,
    )
    dpl_categorical_model = train_rlhf_with_dataset(
        dpl_categorical_model,
        train_data,
        batch_size=batch_size,
        lr=lr,
        num_iterations=1000,
        device=device,
        seed=SEED,  
    )

    print("EVALUATION RESULTS")
    results_dict: Dict[str, dict] = {}
    results_dict["Standard"] = evaluate_model(standard_model, test_data, "standard", device)
    results_dict["DPL(mean-var)"] = evaluate_model(dpl_mean_var_model, test_data, "dpl_mean_var", device)
    results_dict["DPL-MV(risk-averse)"] = evaluate_model(dpl_mean_var_model, test_data, "dpl_mean_var_risk_averse", device, quantile=0.01)
    results_dict["DPL(categorical)"] = evaluate_model(dpl_categorical_model, test_data, "dpl_categorical", device)
    results_dict["DPL(categorical, risk-averse)"] = evaluate_model(dpl_categorical_model, test_data, "dpl_categorical_risk_averse", device, quantile=0.01)

    print("\nRegret by Culture (lower is better):")
    print(f"{'Culture':<15}", end="")
    for model_name in results_dict.keys():
        print(f"{model_name:<20}", end="")
    print()
    print("-" * (15 + 30 * len(results_dict)))

    cultures = list(results_dict["Standard"]["regret_by_culture"].keys())
    for culture in cultures:
        print(f"{culture:<15}", end="")
        for model_name in results_dict.keys():
            regret = results_dict[model_name]["regret_by_culture"][culture]
            print(f"{regret:<20.3f}", end="")
        print()

    print("\nSafety Violations by culture(lower is better):")
    print(f"{'Culture':<15}", end="")
    for model_name in results_dict.keys():
        print(f"{model_name:<20}", end="")
    print()
    print("-" * (15 + 30 * len(results_dict)))

    cultures = list(results_dict["Standard"]["safety_violations_by_culture"].keys())
    for culture in cultures:
        print(f"{culture:<15}", end="")
        for model_name in results_dict.keys():
            viol = results_dict[model_name]["safety_violations_by_culture"][culture]
            print(f"{viol:<20.3f}", end="")
        print()

    print("\nFAIRNESS METRICS")
    print(f"\n{'Model':<30} {'Avg Regret':<12} {'Worst-case':<12} {'Variance':<12} {'r²':<12}")
    print("-" * 78)
    for model_name, results in results_dict.items():
        regrets = list(results["regret_by_culture"].values())
        avg = np.mean(regrets)
        worst = max(regrets)
        var = np.var(regrets)
        r2 = results.get("r_squared", 0.0)
        print(f"{model_name:<30} {avg:<12.3f} {worst:<12.3f} {var:<12.3f} {r2:<12.3f}")