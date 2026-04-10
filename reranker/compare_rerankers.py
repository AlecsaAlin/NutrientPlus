"""
PSO+AHP  vs  Simple GA  —  Head-to-Head Comparison
====================================================
Runs both rerankers on the same set of test users and prints a
side-by-side table of list-level metrics + timing.

Usage:
    cd NutrientPlus/reranker
    python compare_rerankers.py
"""

import copy
import time
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR       = Path(__file__).parent.absolute()
NUTRIENT_ROOT    = SCRIPT_DIR.parent
MODEL_DIR        = NUTRIENT_ROOT / 'model'
PREPROCESSED_DIR = NUTRIENT_ROOT / 'data' / 'preprocessed' / 'test'
MODEL_PATH       = MODEL_DIR / 'checkpoints' / 'best_model.pt'
OUTPUT_DIR       = SCRIPT_DIR / 'results'

for p in [str(SCRIPT_DIR), str(MODEL_DIR), str(NUTRIENT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ga_reranker import (
    GAConfig, GeneticReranker, FitnessEvaluator,
    CandidateItem, build_candidates_from_model_output,
)
from simple_ga_reranker import SimpleGAConfig, SimpleGeneticReranker, SimpleFitnessEvaluator
from ahp import AHPWeightComputer
from model import TwoTowerModel
from preprocessor import FoodDataPreprocessor, FoodRecommendationDataset



def ndcg_at_k(ranked_relevances: list, k: int = 10) -> float:
    ranked = ranked_relevances[:k]
    dcg    = sum(rel / np.log2(r + 2) for r, rel in enumerate(ranked))
    ideal  = sorted(ranked_relevances, reverse=True)[:k]
    idcg   = sum(rel / np.log2(r + 2) for r, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def nutritional_diversity(items, calorie_bands, protein_bands) -> float:
    cal_hits, prot_hits = set(), set()
    for item in items:
        for i in range(len(calorie_bands) - 1):
            if calorie_bands[i] <= item.calories < calorie_bands[i + 1]:
                cal_hits.add(i); break
        for i in range(len(protein_bands) - 1):
            if protein_bands[i] <= item.protein < protein_bands[i + 1]:
                prot_hits.add(i); break
    return (len(cal_hits) / (len(calorie_bands) - 1) +
            len(prot_hits) / (len(protein_bands) - 1)) / 2.0


def category_diversity(items) -> float:
    return min(len(set(i.category_id for i in items)) / len(items), 1.0) if items else 0.0


def novelty_score(items) -> float:
    return sum(1 for i in items if not i.in_user_history) / len(items) if items else 0.0


def collect_metrics(lst, cand_label, cal_bands, prot_bands, evaluator) -> dict:
    rel = [cand_label.get(item.item_id, 0.0) for item in lst]
    return {
        "ndcg":      ndcg_at_k(rel, k=10),
        "nutrition": nutritional_diversity(lst, cal_bands, prot_bands),
        "category":  category_diversity(lst),
        "novelty":   novelty_score(lst),
        "fitness":   evaluator.evaluate(lst),
    }


def mean_store(store) -> dict:
    return {k: float(np.mean(v)) if v else 0.0 for k, v in store.items()}



def main():
    NUM_USERS = 500       

    GOAL_TYPE   = 'maintain_weight'
    EXPLORATION = 'balanced'

    print("\n" + "=" * 70)
    print("NUTRIENTPLUS  —  PSO+AHP  vs  Simple GA  (Head-to-Head)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device : {device}")

    preprocessor = FoodDataPreprocessor.load_preprocessed(str(PREPROCESSED_DIR))
    info         = preprocessor.get_dataset_info()
    print(f"  Users  : {info['num_users']:,}   Items : {info['num_foods']:,}")

    test_dataset = FoodRecommendationDataset(preprocessor, df=preprocessor.test_df)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print(f"  Test samples : {len(test_dataset):,}")

    model = TwoTowerModel(
        num_users        = info["num_users"],
        num_items        = info["num_foods"],
        user_feature_dim = info["user_feature_dim"],
        item_feature_dim = info["food_feature_dim"],
        embedding_dim    = 64,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    print("\n── Two-tower inference ──")
    user_item_scores: dict = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            (user_ids, user_features, user_history,
             item_ids, item_features, positions, labels) = batch

            uid_np = user_ids.cpu().numpy()
            iid_np = item_ids.cpu().numpy()

            preds = model(
                user_ids.to(device), user_features.to(device),
                user_history.to(device), item_ids.to(device),
                item_features.to(device), positions.to(device),
            )
            rank_scores = torch.sigmoid(preds["ranking"]).cpu().numpy()
            high_labels = labels[:, 0].cpu().numpy()

            for uid, iid, score, label in zip(uid_np, iid_np, rank_scores, high_labels):
                user_item_scores.setdefault(int(uid), []).append(
                    (int(iid), float(score), float(label))
                )

    eligible_users = [uid for uid, items in user_item_scores.items() if len(items) >= 10]
    sample_users   = eligible_users[:NUM_USERS]
    print(f"  Eligible users  : {len(eligible_users):,}")
    print(f"  Users to eval   : {len(sample_users)}")

    recipes_df               = preprocessor.recipes_df
    food_id_map              = preprocessor.food_id_map
    available_nutrition_cols = preprocessor.available_nutrition_cols
    max_values               = preprocessor.max_values
    user_history_dict        = preprocessor.user_history_dict
    reverse_user_map         = getattr(preprocessor, 'reverse_user_map', {})

    all_candidates = {}
    all_cand_labels = {}
    for uid in sample_users:
        items  = user_item_scores[uid]
        top50  = sorted(items, key=lambda x: x[1], reverse=True)[:50]
        original_uid  = reverse_user_map.get(uid, uid)
        user_hist_ids = set(user_history_dict.get(original_uid, []))

        cands = build_candidates_from_model_output(
            item_ids                 = [x[0] for x in top50],
            relevance_scores         = [x[1] for x in top50],
            recipes_df               = recipes_df,
            user_history_ids         = user_hist_ids,
            food_id_map              = food_id_map,
            available_nutrition_cols = available_nutrition_cols,
            max_values               = max_values,
        )
        if len(cands) >= 10:
            all_candidates[uid]   = cands
            all_cand_labels[uid]  = {x[0]: x[2] for x in top50}

    eval_users = list(all_candidates.keys())
    print(f"  Users with ≥10 candidates : {len(eval_users)}")

    print("\n── PSO+AHP Re-ranking ──")
    ahp     = AHPWeightComputer()
    weights = ahp.compute_weights(GOAL_TYPE, EXPLORATION)
    print(f"  AHP weights ({GOAL_TYPE}, {EXPLORATION}):")
    for k, v in weights.items():
        print(f"    {k}: {v:.4f}")

    pso_config   = GAConfig(**weights)
    pso_reranker = GeneticReranker(pso_config, seed=42)
    pso_eval     = FitnessEvaluator(pso_config)
    cal_bands    = pso_config.calorie_bands
    prot_bands   = pso_config.protein_bands

    pso_store  = {"ndcg": [], "nutrition": [], "category": [], "novelty": [], "fitness": []}
    base_store = {"ndcg": [], "nutrition": [], "category": [], "novelty": [], "fitness": []}

    t0 = time.perf_counter()
    for uid in tqdm(eval_users, desc="PSO+AHP"):
        cands      = all_candidates[uid]
        cand_label = all_cand_labels[uid]

        baseline_list = copy.deepcopy(cands[:10])
        ga_list       = pso_reranker.evolve(copy.deepcopy(cands))

        for lst, store, ev in [
            (baseline_list, base_store, pso_eval),
            (ga_list,       pso_store,  pso_eval),
        ]:
            m = collect_metrics(lst, cand_label, cal_bands, prot_bands, ev)
            for k in store:
                store[k].append(m[k])

    pso_time  = time.perf_counter() - t0
    pso_agg   = mean_store(pso_store)
    base_agg  = mean_store(base_store)

    print("\n── Simple GA Re-ranking ──")
    sga_config   = SimpleGAConfig()       
    sga_reranker = SimpleGeneticReranker(sga_config, seed=42)
    sga_eval     = SimpleFitnessEvaluator(sga_config)

    sga_store = {"ndcg": [], "nutrition": [], "category": [], "novelty": [], "fitness": []}

    t0 = time.perf_counter()
    for uid in tqdm(eval_users, desc="Simple GA"):
        cands      = all_candidates[uid]
        cand_label = all_cand_labels[uid]

        ga_list = sga_reranker.evolve(copy.deepcopy(cands))
        m = collect_metrics(ga_list, cand_label, cal_bands, prot_bands, sga_eval)
        for k in sga_store:
            sga_store[k].append(m[k])

    sga_time = time.perf_counter() - t0
    sga_agg  = mean_store(sga_store)

    METRIC_LABELS = {
        "ndcg":      "NDCG@10",
        "nutrition": "Nutritional Diversity",
        "category":  "Category Diversity",
        "novelty":   "Novelty Score",
        "fitness":   "Composite Fitness",
    }

    print("\n" + "=" * 78)
    print("RESULTS  (averaged over {:,} users)".format(len(eval_users)))
    print("=" * 78)
    print(f"  {'Metric':<28} {'Greedy':>9} {'Simple GA':>11} {'PSO+AHP':>11} {'Δ PSO-SGA':>11}")
    print("  " + "-" * 72)
    for key, label in METRIC_LABELS.items():
        b   = base_agg[key]
        s   = sga_agg[key]
        p   = pso_agg[key]
        d   = p - s
        sign = "+" if d >= 0 else ""
        print(f"  {label:<28} {b:>9.4f} {s:>11.4f} {p:>11.4f}  {sign}{d:>9.4f}")

    print()
    print(f"  Time — PSO+AHP : {pso_time:.1f}s   ({pso_time/len(eval_users)*1000:.0f} ms/user)")
    print(f"  Time — Simple GA : {sga_time:.1f}s   ({sga_time/len(eval_users)*1000:.0f} ms/user)")
    speedup = pso_time / sga_time if sga_time > 0 else float('inf')
    print(f"  PSO+AHP is {speedup:.2f}× {'slower' if speedup > 1 else 'faster'} than Simple GA")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "comparison_pso_ahp_vs_simple_ga.json"
    result = {
        "settings": {
            "goal_type": GOAL_TYPE,
            "exploration_preference": EXPLORATION,
            "users_evaluated": len(eval_users),
            "ahp_weights": weights,
        },
        "greedy_baseline": base_agg,
        "simple_ga": sga_agg,
        "pso_ahp":   pso_agg,
        "delta_pso_minus_sga": {k: round(pso_agg[k] - sga_agg[k], 6) for k in pso_agg},
        "timing": {
            "pso_ahp_total_s":   round(pso_time,  2),
            "simple_ga_total_s": round(sga_time, 2),
            "pso_ahp_ms_per_user":   round(pso_time  / len(eval_users) * 1000, 1),
            "simple_ga_ms_per_user": round(sga_time / len(eval_users) * 1000, 1),
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")
    print("✅ COMPARISON COMPLETE!")


if __name__ == "__main__":
    main()
