import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes user identity, explicit features, and interaction history into a fixed-size embedding.

    History is encoded with additive attention directly over item embeddings —
    no LSTM.  This is fully parallelizable (no sequential dependency), making
    it significantly faster than LSTM while retaining the ability to focus on
    the most relevant history items per prediction.
    """

    def __init__(self, num_users, user_feature_dim, embedding_dim=64):
        super().__init__()
        self.user_embedding   = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.user_feature_net = nn.Sequential(
            nn.Linear(user_feature_dim, 64), nn.ReLU(), nn.Linear(64, embedding_dim)
        )
        # Additive attention over raw item embeddings — single linear scoring
        # layer replaces the entire LSTM.  Padding tokens are masked to -inf
        # before softmax so they contribute zero weight.
        self.history_attention = nn.Linear(embedding_dim, 1)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, user_ids, user_features, item_embeddings, hist_lengths=None):
        uid_emb  = self.user_embedding(user_ids)
        feat_emb = self.user_feature_net(user_features)

        # Attention directly on item embeddings — fully parallel, no LSTM.
        # Clamp hist_lengths to min=1 so users with zero history always expose
        # at least one position to softmax (avoids all-inf → NaN).
        attn_scores = self.history_attention(item_embeddings).squeeze(-1)  # [B, T]
        if hist_lengths is not None:
            eff_lengths = hist_lengths.clamp(min=1).to(item_embeddings.device)
            mask = torch.arange(item_embeddings.size(1), device=item_embeddings.device).unsqueeze(0) >= eff_lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)         # [B, T, 1]
        hist_emb = (attn_weights * item_embeddings).sum(dim=1)             # [B, D]

        return self.fusion(torch.cat([uid_emb, feat_emb, hist_emb], dim=1))


class ItemTower(nn.Module):
    """Encodes recipe identity and nutritional features into a fixed-size embedding."""

    def __init__(self, num_items, item_feature_dim, embedding_dim=64):
        super().__init__()
        self.item_embedding   = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.item_feature_net = nn.Sequential(
            nn.Linear(item_feature_dim, 64), nn.ReLU(), nn.Linear(64, embedding_dim)
        )
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, item_ids, item_features):
        id_emb   = self.item_embedding(item_ids)
        feat_emb = self.item_feature_net(item_features)
        return self.fusion(torch.cat([id_emb, feat_emb], dim=1)), id_emb


class TwoTowerModel(nn.Module):
    """Two-tower recommendation model with ranking and rating heads.

    Interaction between user and item representations is captured via both
    concatenation and element-wise product (Neural Collaborative Filtering
    trick), giving the prediction heads additive AND multiplicative signals.
    """

    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, embedding_dim=64):
        super().__init__()
        self.num_users     = num_users
        self.num_items     = num_items
        self.embedding_dim = embedding_dim
        self.user_tower    = UserTower(num_users, user_feature_dim, embedding_dim)
        self.item_tower    = ItemTower(num_items, item_feature_dim, embedding_dim)
        # Input: [user_repr, item_repr, user_repr * item_repr] = 3 × embedding_dim
        self.ranking_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.rating_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, user_ids, user_features, user_history, item_ids, item_features, positions):
        hist_emb     = self.item_tower.item_embedding(user_history)
        hist_lengths = (user_history != 0).sum(dim=1)
        user_repr    = self.user_tower(user_ids, user_features, hist_emb, hist_lengths)
        item_repr, _ = self.item_tower(item_ids, item_features)
        # Element-wise product captures multiplicative user-item interactions
        interaction  = user_repr * item_repr
        combined     = torch.cat([user_repr, item_repr, interaction], dim=1)
        return {
            'ranking':   self.ranking_head(combined).squeeze(-1),
            'rating':    self.rating_head(combined).squeeze(-1),
            'user_repr': user_repr,
            'item_repr': item_repr,
        }


# Keep SimpleTwoTowerModel as an alias so existing checkpoints and
# any code that still references the old name loads without errors.
SimpleTwoTowerModel = TwoTowerModel
