import torch
import torch.nn as nn
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
    mean_absolute_error
)
import sys

# ============= SIMPLE PATH CONFIGURATION =============
SCRIPT_DIR = Path(__file__).parent.absolute()
PREPROCESSED_DIR = Path(SCRIPT_DIR) / '../../preprocessed_data/test/'  # Use TEST data
MODEL_PATH = Path(SCRIPT_DIR) / 'model_outputs' / 'best_model.pt'
OUTPUT_DIR = Path(SCRIPT_DIR) /  'evaluation_results'

print(f"\n📂 Script Directory: {SCRIPT_DIR}")
print(f"📂 Looking for test data: {PREPROCESSED_DIR}")
print(f"📂 Looking for model: {MODEL_PATH}")
print(f"📂 Results will go to: {OUTPUT_DIR}")


class UserTower(nn.Module):
    """User representation tower - must match training architecture."""

    def __init__(self, num_users, user_feature_dim, embedding_dim=32):
        super(UserTower, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.user_feature_net = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.history_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, user_ids, user_features, user_history, item_embeddings):
        user_id_emb = self.user_embedding(user_ids)
        user_feat_emb = self.user_feature_net(user_features)
        lstm_out, _ = self.history_lstm(item_embeddings)
        history_emb = lstm_out[:, -1, :]
        combined = torch.cat([user_id_emb, user_feat_emb, history_emb], dim=1)
        user_repr = self.fusion(combined)
        return user_repr


class ItemTower(nn.Module):
    """Item (recipe) representation tower - must match training architecture."""

    def __init__(self, num_items, item_feature_dim, embedding_dim=32):
        super(ItemTower, self).__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.item_feature_net = nn.Sequential(
            nn.Linear(item_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, item_ids, item_features):
        item_id_emb = self.item_embedding(item_ids)
        item_feat_emb = self.item_feature_net(item_features)
        combined = torch.cat([item_id_emb, item_feat_emb], dim=1)
        item_repr = self.fusion(combined)
        return item_repr, item_id_emb


class SimpleTwoTowerModel(nn.Module):
    """Simple Two-Tower model - must match training architecture."""

    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, embedding_dim=32):
        super(SimpleTwoTowerModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_tower = UserTower(num_users, user_feature_dim, embedding_dim)
        self.item_tower = ItemTower(num_items, item_feature_dim, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        self.ranking_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.rating_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, user_features, user_history, item_ids, item_features, positions):
        history_embeddings = self.item_id_embedding(user_history)
        user_repr = self.user_tower(user_ids, user_features, user_history, history_embeddings)
        item_repr, _ = self.item_tower(item_ids, item_features)
        combined = torch.cat([user_repr, item_repr], dim=1)

        ranking_pred = self.ranking_head(combined).squeeze(-1)
        rating_pred = self.rating_head(combined).squeeze(-1)

        return {
            'ranking': ranking_pred,
            'rating': rating_pred
        }


def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation."""

    model.eval()

    # Storage for predictions and ground truth
    all_ranking_preds = []
    all_rating_preds = []

    all_ranking_labels = []
    all_rating_labels = []

    # Loss functions
    loss_fn_ranking = nn.BCEWithLogitsLoss()
    loss_fn_rating = nn.MSELoss()

    total_loss = 0
    total_ranking_loss = 0
    total_rating_loss = 0

    print("\n" + "=" * 70)
    print("RUNNING EVALUATION ON TEST SET")
    print("=" * 70)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch in pbar:
            user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch

            # Move to device
            user_ids = user_ids.to(device)
            user_features = user_features.to(device)
            user_history = user_history.to(device)
            item_ids = item_ids.to(device)
            item_features = item_features.to(device)
            positions = positions.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(user_ids, user_features, user_history, item_ids, item_features, positions)

            # Extract labels
            high_rating = labels[:, 0]
            normalized_rating = labels[:, 1]

            # Calculate losses
            loss_ranking = loss_fn_ranking(predictions['ranking'], high_rating)
            loss_rating = loss_fn_rating(predictions['rating'], normalized_rating)

            loss = 0.5 * loss_ranking + 0.5 * loss_rating

            total_loss += loss.item()
            total_ranking_loss += loss_ranking.item()
            total_rating_loss += loss_rating.item()

            # Store predictions and labels
            all_ranking_preds.extend(torch.sigmoid(predictions['ranking']).cpu().numpy())
            all_rating_preds.extend(predictions['rating'].cpu().numpy())

            all_ranking_labels.extend(high_rating.cpu().numpy())
            all_rating_labels.extend(normalized_rating.cpu().numpy())

    # Convert to numpy arrays
    all_ranking_preds = np.array(all_ranking_preds)
    all_rating_preds = np.array(all_rating_preds)

    all_ranking_labels = np.array(all_ranking_labels)
    all_rating_labels = np.array(all_rating_labels)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("CALCULATING METRICS")
    print("=" * 70)

    # Average losses
    avg_loss = total_loss / len(test_loader)
    avg_ranking_loss = total_ranking_loss / len(test_loader)
    avg_rating_loss = total_rating_loss / len(test_loader)

    # === HIGH RATING PREDICTION METRICS (Binary Classification) ===
    ranking_binary_preds = (all_ranking_preds >= 0.5).astype(int)

    ranking_accuracy = accuracy_score(all_ranking_labels, ranking_binary_preds)
    ranking_precision = precision_score(all_ranking_labels, ranking_binary_preds, zero_division=0)
    ranking_recall = recall_score(all_ranking_labels, ranking_binary_preds, zero_division=0)
    ranking_f1 = f1_score(all_ranking_labels, ranking_binary_preds, zero_division=0)

    try:
        ranking_auc = roc_auc_score(all_ranking_labels, all_ranking_preds)
    except:
        ranking_auc = 0.0

    # === RATING PREDICTION METRICS (Regression) ===
    # Convert normalized predictions back to 1-5 scale
    rating_preds_scaled = np.clip(all_rating_preds * 5.0, 1.0, 5.0)
    rating_labels_scaled = all_rating_labels * 5.0

    rating_mse = mean_squared_error(rating_labels_scaled, rating_preds_scaled)
    rating_rmse = np.sqrt(rating_mse)
    rating_mae = mean_absolute_error(rating_labels_scaled, rating_preds_scaled)

    # Calculate accuracy within ±0.5 stars
    rating_accuracy_05 = np.mean(np.abs(rating_preds_scaled - rating_labels_scaled) <= 0.5)
    # Calculate accuracy within ±1.0 stars
    rating_accuracy_10 = np.mean(np.abs(rating_preds_scaled - rating_labels_scaled) <= 1.0)

    # Compile results
    results = {
        'overall': {
            'total_loss': float(avg_loss),
            'ranking_loss': float(avg_ranking_loss),
            'rating_loss': float(avg_rating_loss),
            'num_samples': len(all_ranking_labels)
        },
        'high_rating_prediction': {
            'accuracy': float(ranking_accuracy),
            'precision': float(ranking_precision),
            'recall': float(ranking_recall),
            'f1_score': float(ranking_f1),
            'auc_roc': float(ranking_auc)
        },
        'rating_prediction': {
            'mse': float(rating_mse),
            'rmse': float(rating_rmse),
            'mae': float(rating_mae),
            'accuracy_within_0.5_stars': float(rating_accuracy_05),
            'accuracy_within_1.0_stars': float(rating_accuracy_10)
        }
    }

    return results


def print_results(results):
    """Print results in a formatted way."""

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n📊 OVERALL METRICS:")
    print(f"  Total Loss: {results['overall']['total_loss']:.4f}")
    print(f"  Ranking Loss: {results['overall']['ranking_loss']:.4f}")
    print(f"  Rating Loss: {results['overall']['rating_loss']:.4f}")
    print(f"  Samples Evaluated: {results['overall']['num_samples']:,}")

    print("\n🎯 HIGH RATING PREDICTION (Binary: Rating ≥ 4):")
    print(
        f"  Accuracy: {results['high_rating_prediction']['accuracy']:.4f} ({results['high_rating_prediction']['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {results['high_rating_prediction']['precision']:.4f}")
    print(f"  Recall: {results['high_rating_prediction']['recall']:.4f}")
    print(f"  F1 Score: {results['high_rating_prediction']['f1_score']:.4f}")
    print(f"  AUC-ROC: {results['high_rating_prediction']['auc_roc']:.4f}")

    print("\n⭐ RATING PREDICTION (1-5 Stars):")
    print(f"  RMSE: {results['rating_prediction']['rmse']:.4f} stars")
    print(f"  MAE: {results['rating_prediction']['mae']:.4f} stars")
    print(
        f"  Accuracy (±0.5 stars): {results['rating_prediction']['accuracy_within_0.5_stars']:.4f} ({results['rating_prediction']['accuracy_within_0.5_stars'] * 100:.2f}%)")
    print(
        f"  Accuracy (±1.0 stars): {results['rating_prediction']['accuracy_within_1.0_stars']:.4f} ({results['rating_prediction']['accuracy_within_1.0_stars'] * 100:.2f}%)")

    print("\n" + "=" * 70)


def main():
    """Main evaluation script."""

    print("\n" + "=" * 70)
    print("TWO-TOWER MODEL EVALUATION")
    print("=" * 70)

    # Check paths
    if not PREPROCESSED_DIR.exists():
        print(f"\n❌ Test data folder NOT FOUND!")
        print(f"   Expected: {PREPROCESSED_DIR}")
        print(f"   Make sure you've run the preprocessor with test data!")
        return

    if not MODEL_PATH.exists():
        print(f"\n❌ Model file NOT FOUND!")
        print(f"   Expected: {MODEL_PATH}")
        print(f"   Make sure you've trained the model first!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")

    # Load preprocessed test data
    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    try:
        # Add current directory to Python path
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))

        from food_data_preprocessor import FoodDataPreprocessor, FoodRecommendationDataset

        preprocessor = FoodDataPreprocessor.load_preprocessed(str(PREPROCESSED_DIR))
        print("✓ Test data loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # Get dataset info
    info = preprocessor.get_dataset_info()
    print(f"\n✓ Dataset info:")
    print(f"  Users: {info['num_users']:,}")
    print(f"  Items: {info['num_foods']:,}")

    # Create dataset
    print("\n" + "=" * 70)
    print("CREATING TEST DATASET")
    print("=" * 70)

    test_dataset = FoodRecommendationDataset(preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    print(f"✓ Test samples: {len(test_dataset):,}")

    # Create model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model = SimpleTwoTowerModel(
        num_users=info['num_users'],
        num_items=info['num_foods'],
        user_feature_dim=info['user_feature_dim'],
        item_feature_dim=info['food_feature_dim'],
        embedding_dim=64  # Must match training
    )

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded with {total_params:,} parameters")

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    # Print results
    print_results(results)

    # Save results
    results_file = OUTPUT_DIR / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()