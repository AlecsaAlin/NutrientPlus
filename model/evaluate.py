import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
import sys

SCRIPT_DIR       = Path(__file__).parent.absolute()
PREPROCESSED_DIR = SCRIPT_DIR.parent / 'data' / 'preprocessed' / 'test'
MODEL_PATH       = SCRIPT_DIR / 'checkpoints' / 'best_model.pt'
OUTPUT_DIR       = SCRIPT_DIR / 'results'

from model import TwoTowerModel as SimpleTwoTowerModel

CONTRASTIVE_WEIGHT = 0.05
TEMPERATURE        = 0.10


def contrastive_loss(user_repr: torch.Tensor, item_repr: torch.Tensor,
                     temperature: float = TEMPERATURE,
                     user_ids: torch.Tensor = None) -> torch.Tensor:
    """InfoNCE in-batch contrastive loss with same-user masking — identical to train.py."""
    u      = F.normalize(user_repr, dim=-1)
    v      = F.normalize(item_repr, dim=-1)
    logits = torch.matmul(u, v.T) / temperature
    if user_ids is not None:
        same_user = user_ids.unsqueeze(0) == user_ids.unsqueeze(1)
        same_user.fill_diagonal_(False)
        logits = logits.masked_fill(same_user, float('-inf'))
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation on the test set."""
    model.eval()

    all_ranking_preds  = []
    all_rating_preds   = []
    all_ranking_labels = []
    all_rating_labels  = []

    loss_fn_ranking = nn.BCEWithLogitsLoss()
    loss_fn_rating  = nn.MSELoss()

    total_loss = total_ranking_loss = total_rating_loss = total_contrastive_loss = 0.0

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION ON TEST SET")
    print("=" * 70)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch in pbar:
            user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch

            user_ids      = user_ids.to(device)
            user_features = user_features.to(device)
            user_history  = user_history.to(device)
            item_ids      = item_ids.to(device)
            item_features = item_features.to(device)
            positions     = positions.to(device)
            labels        = labels.to(device)

            predictions = model(user_ids, user_features, user_history, item_ids, item_features, positions)

            high_rating       = labels[:, 0]
            normalized_rating = labels[:, 1]

            loss_ranking     = loss_fn_ranking(predictions['ranking'], high_rating)
            loss_rating      = loss_fn_rating(predictions['rating'],   normalized_rating)
            loss_contrastive = contrastive_loss(predictions['user_repr'], predictions['item_repr'],
                                                 user_ids=user_ids)
            loss = (
                (1.0 - CONTRASTIVE_WEIGHT) * (0.5 * loss_ranking + 0.5 * loss_rating)
                + CONTRASTIVE_WEIGHT * loss_contrastive
            )

            total_loss             += loss.item()
            total_ranking_loss     += loss_ranking.item()
            total_rating_loss      += loss_rating.item()
            total_contrastive_loss += loss_contrastive.item()

            ranking_probs = torch.sigmoid(predictions['ranking']).cpu().numpy()

            rating_raw = predictions['rating'].cpu().numpy()

            all_ranking_preds.extend(ranking_probs)
            all_rating_preds.extend(rating_raw)
            all_ranking_labels.extend(high_rating.cpu().numpy())
            all_rating_labels.extend(normalized_rating.cpu().numpy())

    all_ranking_preds  = np.array(all_ranking_preds)
    all_rating_preds   = np.array(all_rating_preds)
    all_ranking_labels = np.array(all_ranking_labels)
    all_rating_labels  = np.array(all_rating_labels)

    avg_loss             = total_loss             / len(test_loader)
    avg_ranking_loss     = total_ranking_loss     / len(test_loader)
    avg_rating_loss      = total_rating_loss      / len(test_loader)
    avg_contrastive_loss = total_contrastive_loss / len(test_loader)

    ranking_binary_preds = (all_ranking_preds >= 0.5).astype(int)

    ranking_accuracy  = accuracy_score(all_ranking_labels,  ranking_binary_preds)
    ranking_precision = precision_score(all_ranking_labels, ranking_binary_preds, zero_division=0)
    ranking_recall    = recall_score(all_ranking_labels,    ranking_binary_preds, zero_division=0)
    ranking_f1        = f1_score(all_ranking_labels,        ranking_binary_preds, zero_division=0)

    try:
        ranking_auc = roc_auc_score(all_ranking_labels, all_ranking_preds)
    except Exception:
        ranking_auc = 0.0

   
    rating_preds_scaled  = np.clip(all_rating_preds,  0.0, 1.0) * 5.0
    rating_labels_scaled = np.clip(all_rating_labels, 0.0, 1.0) * 5.0
    rating_preds_scaled  = np.clip(rating_preds_scaled,  1.0, 5.0)
    rating_labels_scaled = np.clip(rating_labels_scaled, 1.0, 5.0)

    rating_mse  = mean_squared_error(rating_labels_scaled, rating_preds_scaled)
    rating_rmse = float(np.sqrt(rating_mse))
    rating_mae  = mean_absolute_error(rating_labels_scaled, rating_preds_scaled)

    rating_accuracy_05 = float(np.mean(np.abs(rating_preds_scaled - rating_labels_scaled) <= 0.5))
    rating_accuracy_10 = float(np.mean(np.abs(rating_preds_scaled - rating_labels_scaled) <= 1.0))

    results = {
        'overall': {
            'total_loss':        float(avg_loss),
            'ranking_loss':      float(avg_ranking_loss),
            'rating_loss':       float(avg_rating_loss),
            'contrastive_loss':  float(avg_contrastive_loss),
            'num_samples':       int(len(all_ranking_labels)),
        },
        'high_rating_prediction': {
            'accuracy':  float(ranking_accuracy),
            'precision': float(ranking_precision),
            'recall':    float(ranking_recall),
            'f1_score':  float(ranking_f1),
            'auc_roc':   float(ranking_auc),
        },
        'rating_prediction': {
            'mse':                       float(rating_mse),
            'rmse':                      rating_rmse,
            'mae':                       float(rating_mae),
            'accuracy_within_0.5_stars': rating_accuracy_05,
            'accuracy_within_1.0_stars': rating_accuracy_10,
        },
    }
    return results


def print_results(results):
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n📊 OVERALL METRICS:")
    print(f"  Total Loss:        {results['overall']['total_loss']:.4f}")
    print(f"  Ranking Loss:      {results['overall']['ranking_loss']:.4f}")
    print(f"  Rating Loss:       {results['overall']['rating_loss']:.4f}")
    print(f"  Samples Evaluated: {results['overall']['num_samples']:,}")

    hr = results['high_rating_prediction']
    print("\n🎯 HIGH RATING PREDICTION (Binary: Rating ≥ 4):")
    print(f"  Accuracy  : {hr['accuracy']:.4f}  ({hr['accuracy']*100:.2f}%)")
    print(f"  Precision : {hr['precision']:.4f}")
    print(f"  Recall    : {hr['recall']:.4f}")
    print(f"  F1 Score  : {hr['f1_score']:.4f}")
    print(f"  AUC-ROC   : {hr['auc_roc']:.4f}")

    rp = results['rating_prediction']
    print("\n⭐ RATING PREDICTION (1-5 Stars):")
    print(f"  RMSE              : {rp['rmse']:.4f} stars")
    print(f"  MAE               : {rp['mae']:.4f} stars")
    print(f"  Accuracy (±0.5 ★) : {rp['accuracy_within_0.5_stars']:.4f}  ({rp['accuracy_within_0.5_stars']*100:.2f}%)")
    print(f"  Accuracy (±1.0 ★) : {rp['accuracy_within_1.0_stars']:.4f}  ({rp['accuracy_within_1.0_stars']*100:.2f}%)")

    print("\n" + "=" * 70)


def main():
    print("\n" + "=" * 70)
    print("TWO-TOWER MODEL EVALUATION")
    print("=" * 70)

    if not PREPROCESSED_DIR.exists():
        print(f"\n❌ Test data folder NOT FOUND: {PREPROCESSED_DIR}")
        return

    if not MODEL_PATH.exists():
        print(f"\n❌ Model file NOT FOUND: {MODEL_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    try:
        from preprocessor import FoodDataPreprocessor, FoodRecommendationDataset
        preprocessor = FoodDataPreprocessor.load_preprocessed(str(PREPROCESSED_DIR))
        print("✓ Test data loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    info = preprocessor.get_dataset_info()
    print(f"\n✓ Dataset info:")
    print(f"  Users: {info['num_users']:,}")
    print(f"  Items: {info['num_foods']:,}")

    test_dataset = FoodRecommendationDataset(preprocessor, df=preprocessor.test_df)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    print(f"\n✓ Test samples: {len(test_dataset):,}")

    model = SimpleTwoTowerModel(
        num_users        = info['num_users'],
        num_items        = info['num_foods'],
        user_feature_dim = info['user_feature_dim'],
        item_feature_dim = info['food_feature_dim'],
        embedding_dim    = 64,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model_keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    results = evaluate_model(model, test_loader, device)
    print_results(results)

    results_file = OUTPUT_DIR / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
