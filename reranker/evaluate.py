import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error, mean_absolute_error,
)

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
from model import TwoTowerModel as SimpleTwoTowerModel
from ahp import AHPWeightComputer


def ndcg_at_k(ranked_relevances: list, k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    ranked = ranked_relevances[:k]
    dcg  = sum(rel / np.log2(r + 2) for r, rel in enumerate(ranked))
    ideal = sorted(ranked_relevances, reverse=True)[:k]
    idcg = sum(rel / np.log2(r + 2) for r, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def nutritional_diversity(items: list, config: GAConfig) -> float:
    cal_hits, prot_hits = set(), set()
    for item in items:
        for i in range(len(config.calorie_bands) - 1):
            if config.calorie_bands[i] <= item.calories < config.calorie_bands[i + 1]:
                cal_hits.add(i); break
        for i in range(len(config.protein_bands) - 1):
            if config.protein_bands[i] <= item.protein < config.protein_bands[i + 1]:
                prot_hits.add(i); break
    cal_cov  = len(cal_hits)  / (len(config.calorie_bands)  - 1)
    prot_cov = len(prot_hits) / (len(config.protein_bands) - 1)
    return (cal_cov + prot_cov) / 2.0


def category_diversity(items: list) -> float:
    if not items:
        return 0.0
    return min(len(set(i.category_id for i in items)) / len(items), 1.0)


def novelty_score(items: list) -> float:
    if not items:
        return 0.0
    return sum(1 for i in items if not i.in_user_history) / len(items)


def run_evaluation():
    print("\n" + "=" * 70)
    print("NUTRIENTPLUS  -  TWO-TOWER + SA-GA RE-RANKER EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")

    from preprocessor import FoodDataPreprocessor, FoodRecommendationDataset
    preprocessor = FoodDataPreprocessor.load_preprocessed(str(PREPROCESSED_DIR))
    info         = preprocessor.get_dataset_info()
    print(f"  Users : {info['num_users']:,}  |  Items : {info['num_foods']:,}")

    test_dataset = FoodRecommendationDataset(preprocessor, df=preprocessor.test_df)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print(f"  Test samples: {len(test_dataset):,}")

    model = SimpleTwoTowerModel(
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
    model = model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")


    DEMO_GOAL_TYPE            = 'maintain_weight'   # lose_weight | gain_muscle | maintain_weight | eat_healthier
    DEMO_EXPLORATION_PREF     = 'balanced'           # conservative | balanced | adventurous

    ahp     = AHPWeightComputer()
    weights = ahp.compute_weights(DEMO_GOAL_TYPE, DEMO_EXPLORATION_PREF)
    print(f"\n  AHP weights for ({DEMO_GOAL_TYPE}, {DEMO_EXPLORATION_PREF}):")
    for k, v in weights.items():
        print(f"    {k}: {v:.4f}")

    ga_config = GAConfig(**weights)
    reranker  = GeneticReranker(ga_config)
    evaluator = FitnessEvaluator(ga_config)

    loss_fn_rank   = nn.BCEWithLogitsLoss()
    loss_fn_rating = nn.MSELoss()

    all_rank_preds, all_rate_preds   = [], []
    all_rank_labels, all_rate_labels = [], []
    total_loss = total_rank_loss = total_rate_loss = 0.0
    user_item_scores: dict = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            (user_ids, user_features, user_history,
             item_ids, item_features, positions, labels) = batch

            iid_np = item_ids.cpu().numpy()
            uid_np = user_ids.cpu().numpy()

            user_ids      = user_ids.to(device)
            user_features = user_features.to(device)
            user_history  = user_history.to(device)
            item_ids      = item_ids.to(device)
            item_features = item_features.to(device)
            positions     = positions.to(device)
            labels        = labels.to(device)

            preds = model(user_ids, user_features, user_history, item_ids, item_features, positions)

            high_rating       = labels[:, 0]
            normalized_rating = labels[:, 1]

            l_rank = loss_fn_rank(preds["ranking"], high_rating)
            l_rate = loss_fn_rating(preds["rating"], normalized_rating)
            loss   = 0.5 * l_rank + 0.5 * l_rate

            total_loss      += loss.item()
            total_rank_loss += l_rank.item()
            total_rate_loss += l_rate.item()

            rank_scores = torch.sigmoid(preds["ranking"]).cpu().numpy()
            rate_raw = preds["rating"].cpu().numpy()

            all_rank_preds.extend(rank_scores)
            all_rate_preds.extend(rate_raw)
            all_rank_labels.extend(high_rating.cpu().numpy())
            all_rate_labels.extend(normalized_rating.cpu().numpy())

            for uid, iid, score, label in zip(uid_np, iid_np, rank_scores, high_rating.cpu().numpy()):
                user_item_scores.setdefault(int(uid), []).append(
                    (int(iid), float(score), float(label))
                )

    all_rank_preds  = np.array(all_rank_preds)
    all_rate_preds  = np.array(all_rate_preds)
    all_rank_labels = np.array(all_rank_labels)
    all_rate_labels = np.array(all_rate_labels)

    bin_preds  = (all_rank_preds >= 0.5).astype(int)
    n_batches  = len(test_loader)
    rate_pred5 = np.clip(all_rate_preds  * 5.0, 1.0, 5.0)
    rate_lbl5  = np.clip(all_rate_labels * 5.0, 1.0, 5.0)

    orig_metrics = {
        "overall": {
            "total_loss":   float(total_loss   / n_batches),
            "ranking_loss": float(total_rank_loss / n_batches),
            "rating_loss":  float(total_rate_loss / n_batches),
            "num_samples":  int(len(all_rank_labels)),
        },
        "high_rating_prediction": {
            "accuracy":  float(accuracy_score(all_rank_labels, bin_preds)),
            "precision": float(precision_score(all_rank_labels, bin_preds, zero_division=0)),
            "recall":    float(recall_score(all_rank_labels,    bin_preds, zero_division=0)),
            "f1_score":  float(f1_score(all_rank_labels,        bin_preds, zero_division=0)),
            "auc_roc":   float(roc_auc_score(all_rank_labels, all_rank_preds)),
        },
        "rating_prediction": {
            "mse":  float(mean_squared_error(rate_lbl5, rate_pred5)),
            "rmse": float(np.sqrt(mean_squared_error(rate_lbl5, rate_pred5))),
            "mae":  float(mean_absolute_error(rate_lbl5, rate_pred5)),
            "accuracy_within_0.5_stars": float(np.mean(np.abs(rate_pred5 - rate_lbl5) <= 0.5)),
            "accuracy_within_1.0_stars": float(np.mean(np.abs(rate_pred5 - rate_lbl5) <= 1.0)),
        },
    }

    recipes_df               = preprocessor.recipes_df
    food_id_map              = preprocessor.food_id_map
    available_nutrition_cols = preprocessor.available_nutrition_cols
    max_values               = preprocessor.max_values
    user_history_dict        = preprocessor.user_history_dict
    reverse_user_map         = getattr(preprocessor, 'reverse_user_map', {})

    eligible_users = [uid for uid, items in user_item_scores.items() if len(items) >= 10]
    sample_users   = eligible_users[:500]
    print(f"\n── SA-GA re-ranking ({len(sample_users)} users) ──")

    baseline_store = {"ndcg": [], "nutrition": [], "category": [], "novelty": [], "fitness": []}
    ga_store       = {"ndcg": [], "nutrition": [], "category": [], "novelty": [], "fitness": []}

    for uid in tqdm(sample_users, desc="SA-GA Re-ranking"):
        items        = user_item_scores[uid]
        top50        = sorted(items, key=lambda x: x[1], reverse=True)[:ga_config.candidate_pool_size]
        cand_label   = {x[0]: x[2] for x in top50}
        original_uid = reverse_user_map.get(uid, uid)
        user_hist_ids = set(user_history_dict.get(original_uid, []))

        candidates = build_candidates_from_model_output(
            item_ids                 = [x[0] for x in top50],
            relevance_scores         = [x[1] for x in top50],
            recipes_df               = recipes_df,
            user_history_ids         = user_hist_ids,
            food_id_map              = food_id_map,
            available_nutrition_cols = available_nutrition_cols,
            max_values               = max_values,
        )

        if len(candidates) < 10:
            continue

        baseline_list = copy.deepcopy(candidates[:10])
        ga_list       = reranker.evolve(candidates)

        for lst, store in [(baseline_list, baseline_store), (ga_list, ga_store)]:
            rel = [cand_label.get(item.item_id, 0.0) for item in lst]
            store["ndcg"].append(ndcg_at_k(rel, k=10))
            store["nutrition"].append(nutritional_diversity(lst, ga_config))
            store["category"].append(category_diversity(lst))
            store["novelty"].append(novelty_score(lst))
            store["fitness"].append(evaluator.evaluate(lst))

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    metric_keys  = ["ndcg_at_10", "nutritional_diversity", "category_diversity",
                    "novelty_score", "composite_ga_fitness"]
    baseline_agg = {
        "ndcg_at_10":            _mean(baseline_store["ndcg"]),
        "nutritional_diversity": _mean(baseline_store["nutrition"]),
        "category_diversity":    _mean(baseline_store["category"]),
        "novelty_score":         _mean(baseline_store["novelty"]),
        "composite_ga_fitness":  _mean(baseline_store["fitness"]),
    }
    ga_agg = {
        "ndcg_at_10":            _mean(ga_store["ndcg"]),
        "nutritional_diversity": _mean(ga_store["nutrition"]),
        "category_diversity":    _mean(ga_store["category"]),
        "novelty_score":         _mean(ga_store["novelty"]),
        "composite_ga_fitness":  _mean(ga_store["fitness"]),
    }
    improvement = {k: round(ga_agg[k] - baseline_agg[k], 6) for k in metric_keys}

    print("\n" + "=" * 70)
    print("ORIGINAL PER-ITEM METRICS")
    print("=" * 70)
    hr = orig_metrics["high_rating_prediction"]
    rp = orig_metrics["rating_prediction"]
    print(f"  Accuracy  : {hr['accuracy']:.4f}")
    print(f"  Precision : {hr['precision']:.4f}")
    print(f"  Recall    : {hr['recall']:.4f}")
    print(f"  F1        : {hr['f1_score']:.4f}")
    print(f"  AUC-ROC   : {hr['auc_roc']:.4f}")
    print(f"  RMSE      : {rp['rmse']:.4f} stars")
    print(f"  MAE       : {rp['mae']:.4f} stars")

    print("\n" + "=" * 70)
    print("LIST-LEVEL METRICS  (Baseline greedy vs SA-GA)")
    print("=" * 70)
    label_map = {
        "ndcg_at_10":            "NDCG@10",
        "nutritional_diversity": "Nutritional Diversity",
        "category_diversity":    "Category Diversity",
        "novelty_score":         "Novelty Score",
        "composite_ga_fitness":  "Composite SA-GA Fitness",
    }
    print(f"  {'Metric':<30} {'Baseline':>10} {'SA-GA':>10} {'Delta':>10}")
    print("  " + "-" * 62)
    for key, label in label_map.items():
        b    = baseline_agg[key]
        g    = ga_agg[key]
        d    = improvement[key]
        sign = "+" if d >= 0 else ""
        print(f"  {label:<30} {b:>10.4f} {g:>10.4f} {sign}{d:>9.4f}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "evaluation_with_saga.json"
    with open(out_path, "w") as f:
        json.dump({
            "original_metrics": orig_metrics,
            "list_level_metrics": {
                "users_evaluated":  len(sample_users),
                "baseline_greedy":  baseline_agg,
                "saga_reranked":    ga_agg,
                "improvement":      improvement,
            }
        }, f, indent=2)
    print(f"✓ Results saved to: {out_path}")
    print("\n✅ EVALUATION COMPLETE!")


if __name__ == "__main__":
    run_evaluation()
