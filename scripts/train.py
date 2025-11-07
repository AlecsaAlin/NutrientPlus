import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import sys

# ============= SIMPLE PATH CONFIGURATION =============
SCRIPT_DIR = Path(__file__).parent.absolute()
PREPROCESSED_DIR = Path(SCRIPT_DIR) / 'my_preprocessed_data'
OUTPUT_DIR = Path(SCRIPT_DIR) / 'Train' / 'model_outputs'
PREPROCESSOR_SCRIPT = Path(SCRIPT_DIR) / 'food_preprocessor_simple.py'

print(f"\n📂 Script Directory: {SCRIPT_DIR}")
print(f"📂 Looking for preprocessed data: {PREPROCESSED_DIR}")
print(f"📂 Output will go to: {OUTPUT_DIR}")


class UserTower(nn.Module):
    """User representation tower."""

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
    """Item (recipe) representation tower."""

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
    """Simple Two-Tower model."""

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
        self.review_head = nn.Sequential(
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
        review_pred = self.review_head(combined).squeeze(-1)

        return {
            'ranking': ranking_pred,
            'rating': rating_pred,
            'review': review_pred
        }


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_fn_ranking = nn.BCEWithLogitsLoss()
    loss_fn_rating = nn.MSELoss()
    loss_fn_review = nn.BCEWithLogitsLoss()

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch

        user_ids = user_ids.to(device)
        user_features = user_features.to(device)
        user_history = user_history.to(device)
        item_ids = item_ids.to(device)
        item_features = item_features.to(device)
        positions = positions.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(user_ids, user_features, user_history, item_ids, item_features, positions)

        high_rating = labels[:, 0]
        normalized_rating = labels[:, 1]
        has_review = labels[:, 2]

        loss_ranking = loss_fn_ranking(predictions['ranking'], high_rating)
        loss_rating = loss_fn_rating(predictions['rating'], normalized_rating)
        loss_review = loss_fn_review(predictions['review'], has_review)

        loss = 0.4 * loss_ranking + 0.4 * loss_rating + 0.2 * loss_review

        loss.backward()

        # ✅ GRADIENT CLIPPING TO FIX NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    loss_fn_ranking = nn.BCEWithLogitsLoss()
    loss_fn_rating = nn.MSELoss()
    loss_fn_review = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch

            user_ids = user_ids.to(device)
            user_features = user_features.to(device)
            user_history = user_history.to(device)
            item_ids = item_ids.to(device)
            item_features = item_features.to(device)
            positions = positions.to(device)
            labels = labels.to(device)

            predictions = model(user_ids, user_features, user_history, item_ids, item_features, positions)

            high_rating = labels[:, 0]
            normalized_rating = labels[:, 1]
            has_review = labels[:, 2]

            loss_ranking = loss_fn_ranking(predictions['ranking'], high_rating)
            loss_rating = loss_fn_rating(predictions['rating'], normalized_rating)
            loss_review = loss_fn_review(predictions['review'], has_review)

            loss = 0.4 * loss_ranking + 0.4 * loss_rating + 0.2 * loss_review

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(val_loader)


def main():
    """Main training script."""

    print("\n" + "=" * 70)
    print("TWO-TOWER RECOMMENDATION MODEL - SIMPLE GPU TRAINING")
    print("=" * 70)

    # ✅ FIXED HYPERPARAMETERS
    EMBEDDING_DIM = 64
    BATCH_SIZE = 256
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-5
    VALIDATION_SPLIT = 0.2

    # Device
    print("\n" + "=" * 70)
    print("CHECKING GPU")
    print("=" * 70)

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')

    # Create output directory
    print("\n" + "=" * 70)
    print("CHECKING PATHS")
    print("=" * 70)

    if not PREPROCESSED_DIR.exists():
        print(f"❌ Preprocessed data folder NOT FOUND!")
        print(f"   Expected: {PREPROCESSED_DIR}")
        print(f"\n   Make sure you have:")
        print(f"   - training_df.pkl")
        print(f"   - history_df.pkl")
        print(f"   - recipes_df.pkl")
        print(f"   - metadata.pkl")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Preprocessed data folder: {PREPROCESSED_DIR}")
    print(f"✓ Output folder: {OUTPUT_DIR}")

    # Load preprocessed data
    print("\n" + "=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70)

    try:
        # Add current directory to Python path
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))

        from scripts.food_data_preprocessor import FoodDataPreprocessor, FoodRecommendationDataset

        preprocessor = FoodDataPreprocessor.load_preprocessed(str(PREPROCESSED_DIR))
        print("✓ Data loaded successfully!")

    except ImportError as e:
        print(f"❌ Could not import food_data_preprocessor.py")
        print(f"   Make sure it's in: {SCRIPT_DIR}")
        print(f"   Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Get dataset info
    info = preprocessor.get_dataset_info()
    print(f"\n✓ Dataset info:")
    print(f"  Users: {info['num_users']:,}")
    print(f"  Items: {info['num_foods']:,}")

    # Create dataset and split
    print("\n" + "=" * 70)
    print("CREATING DATASET")
    print("=" * 70)

    dataset = FoodRecommendationDataset(preprocessor)

    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"✓ Train samples: {len(train_dataset):,}")
    print(f"✓ Val samples: {len(val_dataset):,}")

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = SimpleTwoTowerModel(
        num_users=info['num_users'],
        num_items=info['num_foods'],
        user_feature_dim=info['user_feature_dim'],
        item_feature_dim=info['food_feature_dim'],
        embedding_dim=EMBEDDING_DIM
    )

    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    start_time = datetime.now()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        train_losses.append(train_loss)

        # Validate
        val_loss = evaluate(model, val_loader, DEVICE)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
            print(f"✓ New best model saved!")

    elapsed_time = datetime.now() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed_time}")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), OUTPUT_DIR / 'final_model.pt')
    print(f"✓ Models saved to: {OUTPUT_DIR}")

    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'training_time': str(elapsed_time)
    }

    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"✓ Training history saved")

    print("\n" + "=" * 70)
    print("✅ TRAINING FINISHED!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_DIR}/best_model.pt")
    print(f"  - {OUTPUT_DIR}/final_model.pt")
    print(f"  - {OUTPUT_DIR}/training_history.json")


if __name__ == "__main__":
    main()