"""
PSO+AHP  vs  Simple GA  —  Head-to-Head Comparison
====================================================
Runs both rerankers on the same set of test users and prints a
side-by-side table of list-level metrics + timing.

Usage:
    cd NutrientPlus/reranker
    python compare_rerankers.py
    python compare_rerankers.py --goal eat_healthier --exploration adventurous
    python compare_rerankers.py --goal lose_weight --exploration conservative

Goals       : lose_weight | gain_muscle | maintain_weight | eat_healthier
Exploration : conservative | balanced | adventurous
"""

import argparse
import copy
import os
import time
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

SCRIPT_DIR            = Path(__file__).parent.absolute()
NUTRIENT_ROOT         = SCRIPT_DIR.parent
MODEL_DIR             = NUTRIENT_ROOT / 'model'
PREPROCESSED_DIR      = NUTRIENT_ROOT / 'data' / 'preprocessed' / 'train'
PREPROCESSED_DIR_TEST = NUTRIENT_ROOT / 'data' / 'preprocessed' / 'test'
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
from nsga3_reranker import NSGA3Config, NSGA3Reranker, ObjectiveEvaluator
from bwo_reranker import BWOConfig, BWOReranker, BWOFitnessEvaluator
from mopso_reranker import MOPSOConfig, MOPSOReranker
from ahp import AHPWeightComputer
from model import TwoTowerModel
from preprocessor import FoodDataPreprocessor, FoodRecommendationDataset



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


def collect_metrics(lst, cal_bands, prot_bands, evaluator) -> dict:
    return {
        "nutrition": nutritional_diversity(lst, cal_bands, prot_bands),
        "category":  category_diversity(lst),
        "novelty":   novelty_score(lst),
        "fitness":   evaluator.evaluate(lst),
    }


def _eval_user(args):
    """Worker: runs all rerankers on one user's candidates.

    All output metrics — including Composite Fitness — are measured with the
    same shared AHP-weighted FitnessEvaluator so results are directly
    comparable across algorithms.  Each reranker still uses its own internal
    evaluator during evolution; only the final measurement is unified.
    """
    cands, pso_config, sga_config, nsga3_config, bwo_config, mopso_config, cal_bands, prot_bands = args

    shared_eval = FitnessEvaluator(pso_config)

    baseline = copy.deepcopy(cands[:10])
    base_m   = collect_metrics(baseline, cal_bands, prot_bands, shared_eval)

    pso_reranker = GeneticReranker(pso_config, seed=42)
    pso_list     = pso_reranker.evolve(copy.deepcopy(cands))
    pso_m        = collect_metrics(pso_list, cal_bands, prot_bands, shared_eval)

    sga_reranker = SimpleGeneticReranker(sga_config, seed=42)
    sga_list     = sga_reranker.evolve(copy.deepcopy(cands))
    sga_m        = collect_metrics(sga_list, cal_bands, prot_bands, shared_eval)

    nsga3_reranker = NSGA3Reranker(nsga3_config, seed=42)
    nsga3_list     = nsga3_reranker.evolve(copy.deepcopy(cands))
    nsga3_m        = collect_metrics(nsga3_list, cal_bands, prot_bands, shared_eval)

    bwo_reranker = BWOReranker(bwo_config, seed=42)
    bwo_list     = bwo_reranker.evolve(copy.deepcopy(cands))
    bwo_m        = collect_metrics(bwo_list, cal_bands, prot_bands, shared_eval)

    mopso_reranker = MOPSOReranker(mopso_config, seed=42)
    mopso_list     = mopso_reranker.evolve(copy.deepcopy(cands))
    mopso_m        = collect_metrics(mopso_list, cal_bands, prot_bands, shared_eval)

    return base_m, pso_m, sga_m, nsga3_m, bwo_m, mopso_m


def aggregate_store(store) -> dict:
    return {
        k: {
            "mean": float(np.mean(v)) if v else 0.0,
            "max":  float(np.max(v))  if v else 0.0,
        }
        for k, v in store.items()
    }



VALID_GOALS  = ['lose_weight', 'gain_muscle', 'maintain_weight', 'eat_healthier']
VALID_PREFS  = ['conservative', 'balanced', 'adventurous']


def parse_args():
    parser = argparse.ArgumentParser(
        description='NutrientPlus — GA Reranker Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python compare_rerankers.py\n'
            '  python compare_rerankers.py --goal eat_healthier --exploration adventurous\n'
            '  python compare_rerankers.py --goal lose_weight --exploration conservative\n'
        ),
    )
    parser.add_argument(
        '--goal', default='maintain_weight', choices=VALID_GOALS,
        help='User dietary goal (default: maintain_weight)',
    )
    parser.add_argument(
        '--exploration', default='balanced', choices=VALID_PREFS,
        help='User exploration preference (default: balanced)',
    )
    return parser.parse_args()


def main():
    args        = parse_args()
    GOAL_TYPE   = args.goal
    EXPLORATION = args.exploration

    print("\n" + "=" * 70)
    print("NUTRIENTPLUS  —  GA Reranker Comparison  (Head-to-Head)")
    print("=" * 70)
    print(f"\n  Profile : {GOAL_TYPE} / {EXPLORATION}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device : {device}")

    preprocessor = FoodDataPreprocessor.load_preprocessed(str(PREPROCESSED_DIR))
    info         = preprocessor.get_dataset_info()
    print(f"  Users  : {info['num_users']:,}   Items : {info['num_foods']:,}")

    test_df_extra = pd.read_pickle(str(PREPROCESSED_DIR_TEST / 'training_df.pkl'))
    full_df       = pd.concat([preprocessor.training_df, test_df_extra], ignore_index=True).drop_duplicates()
    print(f"  Train samples : {len(preprocessor.training_df):,}   Test samples : {len(test_df_extra):,}")

    test_dataset = FoodRecommendationDataset(preprocessor, df=full_df)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print(f"  Total samples : {len(test_dataset):,}")

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

            for uid, iid, score in zip(uid_np, iid_np, rank_scores):
                user_item_scores.setdefault(int(uid), []).append(
                    (int(iid), float(score))
                )

    sample_users = [uid for uid, items in user_item_scores.items() if len(items) >= 10]
    print(f"  Users to eval   : {len(sample_users):,}")

    recipes_df               = preprocessor.recipes_df
    food_id_map              = preprocessor.food_id_map
    available_nutrition_cols = preprocessor.available_nutrition_cols
    max_values               = preprocessor.max_values
    user_history_dict        = preprocessor.user_history_dict
    reverse_user_map         = getattr(preprocessor, 'reverse_user_map', {})

    reverse_food_map = {v: k for k, v in food_id_map.items()}
    recipes_indexed  = (
        recipes_df.drop_duplicates(subset='FoodId', keep='first').set_index('FoodId')
        if 'FoodId' in recipes_df.columns else recipes_df
    )
    print(f"  Pre-computed food lookup index ({len(recipes_indexed):,} items)")


    TOP_BY_RELEVANCE = 50
    TOP_UNSEEN_BOOST = 20

    all_candidates = {}
    for uid in tqdm(sample_users, desc="Building candidates"):
        items         = user_item_scores[uid]
        original_uid  = reverse_user_map.get(uid, uid)
        user_hist_ids = set(user_history_dict.get(original_uid, []))
        
        hist_mapped = {
            food_id_map[h] for h in user_hist_ids if h in food_id_map
        }

        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        top_rel      = sorted_items[:TOP_BY_RELEVANCE]

        in_pool       = {iid for iid, _ in top_rel}
        unseen_items  = [
            (iid, s) for (iid, s) in sorted_items
            if iid not in hist_mapped and iid not in in_pool
        ]
        top_unseen    = unseen_items[:TOP_UNSEEN_BOOST]

        pool          = top_rel + top_unseen      # up to 70 items

        cands = build_candidates_from_model_output(
            item_ids                 = [x[0] for x in pool],
            relevance_scores         = [x[1] for x in pool],
            recipes_df               = recipes_df,
            user_history_ids         = user_hist_ids,
            food_id_map              = food_id_map,
            available_nutrition_cols = available_nutrition_cols,
            max_values               = max_values,
            reverse_food_map         = reverse_food_map,
            recipes_indexed          = recipes_indexed,
        )
        if len(cands) >= 10:
            all_candidates[uid] = cands

    eval_users = list(all_candidates.keys())
    print(f"  Users with ≥10 candidates : {len(eval_users)}")

    print("\n── PSO+AHP Re-ranking ──")
    ahp     = AHPWeightComputer()
    weights = ahp.compute_weights(GOAL_TYPE, EXPLORATION)
    print(f"  AHP weights ({GOAL_TYPE}, {EXPLORATION}):")
    for k, v in weights.items():
        print(f"    {k}: {v:.4f}")

    SHARED_GENERATIONS  = 20
    SHARED_POPULATION   = 60

    pso_config   = GAConfig(
        **weights,
        num_generations = SHARED_GENERATIONS,
        population_size = SHARED_POPULATION,
    )
    pso_reranker = GeneticReranker(pso_config, seed=42)
    pso_eval     = FitnessEvaluator(pso_config)
    cal_bands    = pso_config.calorie_bands
    prot_bands   = pso_config.protein_bands

    sga_config   = SimpleGAConfig(
        num_generations = SHARED_GENERATIONS,
        population_size = SHARED_POPULATION,
    )
    nsga3_config = NSGA3Config(
        w_relevance     = weights["w_relevance"],
        w_nutrition     = weights["w_nutrition"],
        w_category      = weights["w_category"],
        w_novelty       = weights["w_novelty"],
        num_generations = SHARED_GENERATIONS,
        population_size = SHARED_POPULATION,
        num_divisions   = 4,   # → 35 ref dirs, must be ≤ population_size
    )
    bwo_config = BWOConfig(
        w_relevance     = weights["w_relevance"],
        w_nutrition     = weights["w_nutrition"],
        w_category      = weights["w_category"],
        w_novelty       = weights["w_novelty"],
        num_generations = SHARED_GENERATIONS,
        population_size = SHARED_POPULATION,
    )
    mopso_config = MOPSOConfig(
        w_relevance     = weights["w_relevance"],
        w_nutrition     = weights["w_nutrition"],
        w_category      = weights["w_category"],
        w_novelty       = weights["w_novelty"],
        num_generations = SHARED_GENERATIONS,
        population_size = SHARED_POPULATION,
    )

    pso_store   = {"nutrition": [], "category": [], "novelty": [], "fitness": []}
    base_store  = {"nutrition": [], "category": [], "novelty": [], "fitness": []}
    sga_store   = {"nutrition": [], "category": [], "novelty": [], "fitness": []}
    nsga3_store = {"nutrition": [], "category": [], "novelty": [], "fitness": []}
    bwo_store   = {"nutrition": [], "category": [], "novelty": [], "fitness": []}
    mopso_store = {"nutrition": [], "category": [], "novelty": [], "fitness": []}

    args_list = [
        (all_candidates[uid], pso_config, sga_config, nsga3_config,
         bwo_config, mopso_config, cal_bands, prot_bands)
        for uid in eval_users
    ]

    num_workers = max(1, os.cpu_count() - 2)
    print(f"\n── Parallel Re-ranking ({num_workers} workers) ──")

    t0 = time.perf_counter()
    results = process_map(_eval_user, args_list, max_workers=num_workers, chunksize=8, desc="All rerankers")
    for base_m, pso_m, sga_m, nsga3_m, bwo_m, mopso_m in results:
        for k in base_store:   base_store[k].append(base_m[k])
        for k in pso_store:    pso_store[k].append(pso_m[k])
        for k in sga_store:    sga_store[k].append(sga_m[k])
        for k in nsga3_store:  nsga3_store[k].append(nsga3_m[k])
        for k in bwo_store:    bwo_store[k].append(bwo_m[k])
        for k in mopso_store:  mopso_store[k].append(mopso_m[k])

    total_time = time.perf_counter() - t0

    pso_agg   = aggregate_store(pso_store)
    base_agg  = aggregate_store(base_store)
    sga_agg   = aggregate_store(sga_store)
    nsga3_agg = aggregate_store(nsga3_store)
    bwo_agg   = aggregate_store(bwo_store)
    mopso_agg = aggregate_store(mopso_store)

    METRIC_LABELS = {
        "nutrition": "Nutritional Diversity",
        "category":  "Category Diversity",
        "novelty":   "Novelty Score",
        "fitness":   "Composite Fitness",
    }

    print("\n" + "=" * 127)
    print("RESULTS  ({:,} users)".format(len(eval_users)))
    print("=" * 127)
    for stat in ("mean", "max"):
        label_stat = "Average" if stat == "mean" else "Best (Max)"
        print(f"\n  [{label_stat}]")
        print(f"  {'Metric':<28} {'Greedy':>9} {'Simple GA':>11} {'PSO+AHP':>11} {'NSGA-III':>11} {'BWO':>11} {'MOPSO':>11} {'Δ MOPSO-PSO':>13}")
        print("  " + "-" * 108)
        for key, label in METRIC_LABELS.items():
            b    = base_agg[key][stat]
            s    = sga_agg[key][stat]
            p    = pso_agg[key][stat]
            n    = nsga3_agg[key][stat]
            bw   = bwo_agg[key][stat]
            mo   = mopso_agg[key][stat]
            d    = mo - p
            sign = "+" if d >= 0 else ""
            print(f"  {label:<28} {b:>9.4f} {s:>11.4f} {p:>11.4f} {n:>11.4f} {bw:>11.4f} {mo:>11.4f}  {sign}{d:>9.4f}")

    print()
    print(f"  Total wall time : {total_time:.1f}s   ({total_time/len(eval_users)*1000:.0f} ms/user, {num_workers} workers)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"comparison_{GOAL_TYPE}_{EXPLORATION}.json"
    result = {
        "settings": {
            "goal_type": GOAL_TYPE,
            "exploration_preference": EXPLORATION,
            "users_evaluated": len(eval_users),
            "ahp_weights": weights,
            "shared_generations": SHARED_GENERATIONS,
            "shared_population": SHARED_POPULATION,
            "candidate_pool": f"top-{TOP_BY_RELEVANCE} relevance + top-{TOP_UNSEEN_BOOST} unseen boost",
            "fitness_evaluator": "FitnessEvaluator (AHP-weighted) — shared across all algorithms for fair comparison",
        },
        "greedy_baseline": base_agg,
        "simple_ga":       sga_agg,
        "pso_ahp":         pso_agg,
        "nsga3":           nsga3_agg,
        "bwo":             bwo_agg,
        "mopso":           mopso_agg,
        "delta_pso_minus_sga": {
            k: {s: round(pso_agg[k][s] - sga_agg[k][s], 6) for s in ("mean", "max")}
            for k in pso_agg
        },
        "delta_nsga3_minus_pso": {
            k: {s: round(nsga3_agg[k][s] - pso_agg[k][s], 6) for s in ("mean", "max")}
            for k in nsga3_agg
        },
        "delta_bwo_minus_pso": {
            k: {s: round(bwo_agg[k][s] - pso_agg[k][s], 6) for s in ("mean", "max")}
            for k in bwo_agg
        },
        "delta_mopso_minus_pso": {
            k: {s: round(mopso_agg[k][s] - pso_agg[k][s], 6) for s in ("mean", "max")}
            for k in mopso_agg
        },
        "timing": {
            "total_wall_s":    round(total_time, 2),
            "ms_per_user":     round(total_time / len(eval_users) * 1000, 1),
            "num_workers":     num_workers,
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved → {out_path}")
    print("✅ COMPARISON COMPLETE!")


if __name__ == "__main__":
    main()
