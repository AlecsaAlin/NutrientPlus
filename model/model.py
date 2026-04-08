import torch
import torch.nn as nn


class UserTower(nn.Module):
    """Encodes user identity, explicit features, and interaction history into a fixed-size embedding."""

    def __init__(self, num_users, user_feature_dim, embedding_dim=64):
        super().__init__()
        self.user_embedding   = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.user_feature_net = nn.Sequential(
            nn.Linear(user_feature_dim, 64), nn.ReLU(), nn.Linear(64, embedding_dim)
        )
        self.history_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.fusion       = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, user_ids, user_features, item_embeddings, hist_lengths=None):
        uid_emb  = self.user_embedding(user_ids)
        feat_emb = self.user_feature_net(user_features)

        if hist_lengths is not None:     
            lengths_cpu = hist_lengths.clamp(min=1).cpu()
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                item_embeddings, lengths_cpu, batch_first=True, enforce_sorted=False,
            )
            _, (hn, _) = self.history_lstm(packed)
            hist_emb = hn.squeeze(0)               
        else:
            lstm_out, _ = self.history_lstm(item_embeddings)
            hist_emb = lstm_out[:, -1, :]

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
    """Two-tower recommendation model with ranking, rating, and contrastive loss heads."""

    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim, embedding_dim=64):
        super().__init__()
        self.num_users     = num_users
        self.num_items     = num_items
        self.embedding_dim = embedding_dim
        self.user_tower    = UserTower(num_users, user_feature_dim, embedding_dim)
        self.item_tower    = ItemTower(num_items, item_feature_dim, embedding_dim)
        self.ranking_head  = nn.Sequential(nn.Linear(embedding_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.rating_head   = nn.Sequential(nn.Linear(embedding_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, user_ids, user_features, user_history, item_ids, item_features, positions):

        hist_emb     = self.item_tower.item_embedding(user_history)
        hist_lengths = (user_history != 0).sum(dim=1)
        user_repr    = self.user_tower(user_ids, user_features, hist_emb, hist_lengths)
        item_repr, _ = self.item_tower(item_ids, item_features)
        combined     = torch.cat([user_repr, item_repr], dim=1)
        return {
            'ranking':   self.ranking_head(combined).squeeze(-1),
            'rating':    self.rating_head(combined).squeeze(-1),
            'user_repr': user_repr,
            'item_repr': item_repr,
        }

SimpleTwoTowerModel = TwoTowerModel
