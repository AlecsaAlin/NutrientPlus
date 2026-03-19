import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import sys

SCRIPT_DIR       = Path(__file__).parent.absolute()
PREPROCESSED_DIR = Path(SCRIPT_DIR) / '../data/preprocessed/train/'
OUTPUT_DIR       = Path(SCRIPT_DIR) / 'checkpoints'

from model import TwoTowerModel


def contrastive_loss(user_repr: torch.Tensor, item_repr: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE in-batch negative sampling loss.
    Treats each user's paired recipe as its positive and all other recipes
    in the batch as negatives. Trains the towers to produce discriminative
    embeddings for personalised retrieval, not just pointwise prediction.
    """
    u      = F.normalize(user_repr, dim=-1)
    v      = F.normalize(item_repr, dim=-1)
    logits = torch.matmul(u, v.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def train_epoch(model, train_loader, optimizer, device,
                contrastive_weight=0.20, temperature=0.07):
    """Run one full training epoch and return average loss."""
    model.train()
    total_loss      = 0
    loss_fn_ranking = nn.BCEWithLogitsLoss()
    loss_fn_rating  = nn.MSELoss()

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch
        user_ids      = user_ids.to(device)
        user_features = user_features.to(device)
        user_history  = user_history.to(device)
        item_ids      = item_ids.to(device)
        item_features = item_features.to(device)
        positions     = positions.to(device)
        labels        = labels.to(device)

        optimizer.zero_grad()
        preds = model(user_ids, user_features, user_history, item_ids, item_features, positions)

        high_rating       = labels[:, 0]
        normalized_rating = labels[:, 1]

        l_rank = loss_fn_ranking(preds['ranking'], high_rating)
        l_rate = loss_fn_rating(preds['rating'],   normalized_rating)
        l_ctr  = contrastive_loss(preds['user_repr'], preds['item_repr'], temperature)

        loss = (1.0 - contrastive_weight) * (0.5 * l_rank + 0.5 * l_rate) + contrastive_weight * l_ctr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'ctr': f'{l_ctr.item():.4f}'})

    return total_loss / len(train_loader)


def evaluate(model, val_loader, device,
             contrastive_weight=0.20, temperature=0.07):
    """Evaluate the model on the validation set and return average loss."""
    model.eval()
    total_loss      = 0
    loss_fn_ranking = nn.BCEWithLogitsLoss()
    loss_fn_rating  = nn.MSELoss()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch
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

            l_rank = loss_fn_ranking(preds['ranking'], high_rating)
            l_rate = loss_fn_rating(preds['rating'],   normalized_rating)
            l_ctr  = contrastive_loss(preds['user_repr'], preds['item_repr'], temperature)

            loss = (1.0 - contrastive_weight) * (0.5 * l_rank + 0.5 * l_rate) + contrastive_weight * l_ctr
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(val_loader)


def main():
    """Load data, build model, run training loop with early stopping."""
    EMBEDDING_DIM        = 64
    BATCH_SIZE           = 256
    NUM_EPOCHS           = 20
    LEARNING_RATE        = 1e-4
    VALIDATION_SPLIT     = 0.2
    CONTRASTIVE_WEIGHT   = 0.20
    TEMPERATURE          = 0.07
    EARLY_STOP_PATIENCE  = 5

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"✓ CUDA — {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        print("✓ CPU")

    if not PREPROCESSED_DIR.exists():
        print(f"❌ Preprocessed data not found: {PREPROCESSED_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    try:
        from preprocessor import FoodDataPreprocessor, FoodRecommendationDataset
        preprocessor = FoodDataPreprocessor.load_preprocessed(str(PREPROCESSED_DIR))
    except Exception as e:
        print(f"❌ {e}")
        return

    info = preprocessor.get_dataset_info()
    print(f"✓ Users: {info['num_users']:,}  Items: {info['num_foods']:,}")

    dataset    = FoodRecommendationDataset(preprocessor)
    val_size   = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    NUM_WORKERS  = 0 if sys.platform == 'win32' else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    print(f"✓ Train: {len(train_dataset):,}  Val: {len(val_dataset):,}")

    model = TwoTowerModel(
        num_users        = info['num_users'],
        num_items        = info['num_foods'],
        user_feature_dim = info['user_feature_dim'],
        item_feature_dim = info['food_feature_dim'],
        embedding_dim    = EMBEDDING_DIM,
    )
    model = model.to(DEVICE)
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"\n{'='*70}")
    print(f"TRAINING  (contrastive_weight={CONTRASTIVE_WEIGHT}, temperature={TEMPERATURE})")
    print(f"{'='*70}")

    train_losses, val_losses = [], []
    best_val_loss    = float('inf')
    best_epoch       = 0
    patience_counter = 0
    start_time       = datetime.now()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")

        train_loss = train_epoch(model, train_loader, optimizer, DEVICE,
                                 contrastive_weight=CONTRASTIVE_WEIGHT, temperature=TEMPERATURE)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, DEVICE,
                            contrastive_weight=CONTRASTIVE_WEIGHT, temperature=TEMPERATURE)
        val_losses.append(val_loss)

        print(f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pt')
            print("✓ Best model saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹ Early stopping at epoch {epoch + 1}")
                break

    elapsed = datetime.now() - start_time
    torch.save(model.state_dict(), OUTPUT_DIR / 'final_model.pt')

    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump({
            'train_losses':  train_losses,
            'val_losses':    val_losses,
            'best_epoch':    best_epoch + 1,
            'best_val_loss': float(best_val_loss),
            'training_time': str(elapsed),
        }, f, indent=2)

    print(f"\n✅ Done — {elapsed}  best epoch {best_epoch + 1}  val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
