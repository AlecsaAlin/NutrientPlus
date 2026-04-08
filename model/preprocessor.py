import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from collections import defaultdict
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


class FoodDataPreprocessor:
    """Preprocesses food recommendation data for two-tower model training."""

    def __init__(
            self,
            reviews_path: str,
            history_path: str,
            recipes_path: str,
            max_history_len: int = 20,
            sample_size: int = None,
            test_size: float = 0.2
    ):
        """
        Args:
            reviews_path: Path to reviews CSV (ReviewId, RecipeId, AuthorId, Rating, Review, DateSubmitted)
            history_path: Path to history CSV (user_id, history)
            recipes_path: Path to recipes CSV (RecipeId, Name, AuthorId, CookTime, etc.)
            max_history_len: Maximum length of user history sequence
            sample_size: If set, randomly sample this many reviews for faster testing
            test_size: Proportion of data to use for testing (default: 0.2 for 20%)
        """
        self.max_history_len = max_history_len
        self.sample_size = sample_size
        self.test_size = test_size

        print("Loading datasets...")

        if sample_size:
            print(f"  Sampling {sample_size} reviews for testing...")
            self.reviews_df = pd.read_csv(reviews_path, nrows=sample_size, encoding='utf-8', encoding_errors='ignore')
        else:
            self.reviews_df = pd.read_csv(reviews_path, encoding='utf-8', encoding_errors='ignore')

        self.history_df = pd.read_csv(history_path, usecols=[0, 1], names=['user_id', 'history'],
                                       header=0, encoding='utf-8', encoding_errors='ignore')
        self.recipes_df = pd.read_csv(recipes_path, encoding='utf-8', encoding_errors='ignore')

        print(f"Loaded {len(self.reviews_df)} reviews")
        print(f"Loaded {len(self.history_df)} user histories")
        print(f"Loaded {len(self.recipes_df)} recipes")

        self._preprocess_data()
        self._create_mappings()
        self._prepare_training_data()

    def _preprocess_data(self):
        """Clean and preprocess the raw data."""
        print("\nPreprocessing data...")

        # Clean reviews
        self.reviews_df['Rating'] = pd.to_numeric(self.reviews_df['Rating'], errors='coerce')
        self.reviews_df = self.reviews_df.dropna(subset=['Rating', 'RecipeId'])

        # Rename columns for consistency
        self.reviews_df.rename(columns={
            'AuthorId': 'UserId',
            'RecipeId': 'FoodId'
        }, inplace=True)

        def parse_history(x):
            if pd.isna(x) or x == '[]' or x == '':
                return []
            x = str(x).strip()
            x = x.strip('[]')
            x = x.strip('"\'')
            x = x.replace(' ', '')
            if not x:
                return []
            try:
                return [int(i) for i in x.split(',') if i]
            except ValueError:
                return []

        self.history_df['parsed_history'] = self.history_df['history'].apply(parse_history)
        self.history_df.rename(columns={'user_id': 'UserId'}, inplace=True, errors='ignore')

        # Clean recipes — handle both possible column names from different dataset versions
        if 'RecipeId' in self.recipes_df.columns:
            self.recipes_df.rename(columns={'RecipeId': 'FoodId'}, inplace=True)
        elif 'recipe_id' in self.recipes_df.columns:
            self.recipes_df.rename(columns={'recipe_id': 'FoodId'}, inplace=True)

        self.available_nutrition_cols = []
        possible_cols = [
            'Calories', 'ProteinContent', 'FatContent', 'SugarContent',
            'FiberContent', 'CarbohydrateContent', 'CholesterolContent',
            'SodiumContent', 'CategoryID',
        ]

        _fallback_defaults = {
            'Calories': 200,
            'ProteinContent': 10,
            'FatContent': 8,
            'CarbohydrateContent': 25,
            'SugarContent': 5,
            'FiberContent': 3,
        }

        self.max_values = {}

        for col in possible_cols:
            if col not in self.recipes_df.columns:
                continue

            self.available_nutrition_cols.append(col)
            self.recipes_df[col] = pd.to_numeric(self.recipes_df[col], errors='coerce')

            default_val = _fallback_defaults.get(col, 0)
            median_val  = self.recipes_df[col].median()
            fill_val    = median_val if not pd.isna(median_val) else default_val

            self.recipes_df.loc[:, col] = self.recipes_df[col].fillna(fill_val)

            p99 = float(self.recipes_df[col].quantile(0.99))
            self.max_values[col] = p99 if p99 > 0 else max(float(self.recipes_df[col].max()), 1.0)

            self.recipes_df.loc[self.recipes_df[col] > self.max_values[col], col] = self.max_values[col]
            self.recipes_df.loc[self.recipes_df[col] < 0, col] = default_val

        print(f"Available nutritional columns: {self.available_nutrition_cols}")

        def parse_time(time_str):
            if pd.isna(time_str):
                return 0
            try:
                time_str = str(time_str)
                if 'PT' in time_str:
                    time_str = time_str.replace('PT', '')
                    minutes = 0
                    if 'H' in time_str:
                        hours = int(time_str.split('H')[0])
                        minutes += hours * 60
                        time_str = time_str.split('H')[1] if 'H' in time_str else ''
                    if 'M' in time_str:
                        mins = time_str.replace('M', '')
                        if mins:
                            minutes += int(mins)
                    return minutes
                return 0
            except Exception:
                return 0

        if 'CookTime' in self.recipes_df.columns:
            self.recipes_df['CookTimeMinutes'] = self.recipes_df['CookTime'].apply(parse_time)
        if 'PrepTime' in self.recipes_df.columns:
            self.recipes_df['PrepTimeMinutes'] = self.recipes_df['PrepTime'].apply(parse_time)
        if 'TotalTime' in self.recipes_df.columns:
            self.recipes_df['TotalTimeMinutes'] = self.recipes_df['TotalTime'].apply(parse_time)
            self.recipes_df.loc[self.recipes_df['TotalTimeMinutes'] > 1440, 'TotalTimeMinutes'] = 1440

        if 'TotalTimeMinutes' in self.recipes_df.columns:
            p99_time = float(self.recipes_df['TotalTimeMinutes'].quantile(0.99))
            self.max_values['TotalTimeMinutes'] = p99_time if p99_time > 0 else 480.0

    def _create_mappings(self):
        """Create ID mappings for users and foods."""
        print("Creating ID mappings...")

        all_users = set(self.reviews_df['UserId'].unique())
        all_users.update(self.history_df['UserId'].unique())

        all_foods = set(self.reviews_df['FoodId'].unique())
        all_foods.update(self.recipes_df['FoodId'].unique())

        print(f"Found {len(all_users)} unique users and {len(all_foods)} unique foods")

        self.user_id_map = {uid: idx + 1 for idx, uid in enumerate(sorted(all_users))}
        self.food_id_map = {fid: idx + 1 for idx, fid in enumerate(sorted(all_foods))}

        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_food_map = {v: k for k, v in self.food_id_map.items()}

        print(f"Created mappings for {len(self.user_id_map)} users and {len(self.food_id_map)} foods")

        print("Creating user history dictionary...")
        self.user_history_dict = {}

        for row in self.history_df.itertuples(index=False):
            user_id = row.UserId
            history = row.parsed_history[:self.max_history_len]
            mapped  = [self.food_id_map[fid] for fid in history if fid in self.food_id_map]
            self.user_history_dict[user_id] = mapped

        print(f"✓ User history dictionary created for {len(self.user_history_dict)} users")

    def _prepare_training_data(self):
        """Merge reviews with recipe features and split into train/test."""
        full_df = self.reviews_df.merge(self.recipes_df, on='FoodId', how='left')

        full_df['HighRating'] = (full_df['Rating'] >= 4).astype(float)

        print(f"\nSplitting data into train ({int((1-self.test_size)*100)}%) "
              f"and test ({int(self.test_size*100)}%) sets (stratified by user)...")

        user_counts  = full_df['UserId'].value_counts()
        stratify_mask = full_df['UserId'].isin(user_counts[user_counts >= 2].index)

        df_stratify = full_df[stratify_mask]
        df_solo     = full_df[~stratify_mask]  # single-review users → always train

        train_parts, test_parts = [], []

        if len(df_stratify) > 0:
            tr, te = train_test_split(
                df_stratify,
                test_size=self.test_size,
                random_state=42,
                stratify=df_stratify['UserId'],
            )
            train_parts.append(tr)
            test_parts.append(te)

        if len(df_solo) > 0:
            train_parts.append(df_solo)

        self.train_df = pd.concat(train_parts).reset_index(drop=True)
        self.test_df  = (pd.concat(test_parts).reset_index(drop=True)
                         if test_parts else pd.DataFrame(columns=full_df.columns))

        print("Generating Position based on Rating (per split)...")
        def _assign_positions(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df = df.sort_values('Rating', ascending=False)
            df['Position'] = df.groupby('UserId').cumcount().clip(upper=9)
            return df.reset_index(drop=True)

        self.train_df = _assign_positions(self.train_df)
        self.test_df  = _assign_positions(self.test_df) if len(self.test_df) > 0 else self.test_df

        self.training_df = full_df

        train_user_stats = (
            self.train_df.groupby('UserId', sort=False)
            .agg(AvgRating=('Rating', 'mean'), NumReviews=('Rating', 'count'))
            .reset_index()
        )
        if 'TotalTimeMinutes' in self.train_df.columns:
            time_stats = (
                self.train_df.groupby('UserId', sort=False)['TotalTimeMinutes']
                .mean()
                .rename('AvgTimePreference')
                .reset_index()
            )
            train_user_stats = train_user_stats.merge(time_stats, on='UserId', how='left')
            train_user_stats['AvgTimePreference'] = train_user_stats['AvgTimePreference'].fillna(30.0)
        else:
            train_user_stats['AvgTimePreference'] = 30.0
        self.user_stats = train_user_stats

        print(f"✓ Total samples:    {len(full_df):,}")
        print(f"✓ Training samples: {len(self.train_df):,}")
        print(f"✓ Testing samples:  {len(self.test_df):,}")
        print(f"\nPosition distribution in training set:\n"
              f"{self.train_df['Position'].value_counts().sort_index()}")
        print(f"\nPosition distribution in testing set:\n"
              f"{self.test_df['Position'].value_counts().sort_index()}")

    def get_dataset_info(self) -> Dict:
        """Returns information needed to initialize the model."""
        return {
            'num_users':       len(self.user_id_map),
            'num_foods':       len(self.food_id_map),
            'user_feature_dim': self._get_user_feature_dim(),
            'food_feature_dim': self._get_food_feature_dim(),
            'max_history_len': self.max_history_len,
        }

    def save_preprocessed(self, output_dir: str):
        """
        Save preprocessed data for fast loading later.
        Creates separate 'train' and 'test' subdirectories.
        """
        train_dir = os.path.join(output_dir, 'train')
        test_dir  = os.path.join(output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        print(f"\nSaving preprocessed data to {output_dir}...")

        # Train split
        self.train_df.to_pickle(os.path.join(train_dir, 'training_df.pkl'))
        self.history_df.to_pickle(os.path.join(train_dir, 'history_df.pkl'))
        self.recipes_df.to_pickle(os.path.join(train_dir, 'recipes_df.pkl'))

        # Test split
        self.test_df.to_pickle(os.path.join(test_dir, 'training_df.pkl'))
        self.history_df.to_pickle(os.path.join(test_dir, 'history_df.pkl'))
        self.recipes_df.to_pickle(os.path.join(test_dir, 'recipes_df.pkl'))

        self.user_stats.to_pickle(os.path.join(train_dir, 'user_stats.pkl'))
        self.user_stats.to_pickle(os.path.join(test_dir,  'user_stats.pkl'))

        metadata = {
            'user_id_map':             self.user_id_map,
            'food_id_map':             self.food_id_map,
            'reverse_user_map':        self.reverse_user_map,
            'reverse_food_map':        self.reverse_food_map,
            'user_history_dict':       self.user_history_dict,
            'available_nutrition_cols': self.available_nutrition_cols,
            'max_history_len':         self.max_history_len,
            'test_size':               self.test_size,
            'max_values':              self.max_values,
        }

        for d in (train_dir, test_dir):
            with open(os.path.join(d, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)

        print(f"\n✓ TRAIN folder: {len(self.train_df):,} samples")
        print(f"✓ TEST  folder: {len(self.test_df):,} samples")
        print(f"✓ Preprocessing complete! Saved to {output_dir}")
        print(f"  Use '{train_dir}' for training")
        print(f"  Use '{test_dir}'  for testing/evaluation")

    @classmethod
    def load_preprocessed(cls, input_dir: str):
        """Load preprocessed data quickly without reprocessing."""
        print(f"Loading preprocessed data from {input_dir}...")

        instance = cls.__new__(cls)

        instance.training_df = pd.read_pickle(os.path.join(input_dir, 'training_df.pkl'))
        instance.test_df     = instance.training_df
        instance.history_df  = pd.read_pickle(os.path.join(input_dir, 'history_df.pkl'))
        instance.recipes_df  = pd.read_pickle(os.path.join(input_dir, 'recipes_df.pkl'))

        with open(os.path.join(input_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        instance.user_id_map             = metadata['user_id_map']
        instance.food_id_map             = metadata['food_id_map']
        instance.reverse_user_map        = metadata['reverse_user_map']
        instance.reverse_food_map        = metadata['reverse_food_map']
        instance.user_history_dict       = metadata['user_history_dict']
        instance.available_nutrition_cols = metadata['available_nutrition_cols']
        instance.max_history_len         = metadata['max_history_len']
        instance.test_size               = metadata.get('test_size', 0.2)
        instance.max_values              = metadata.get('max_values', {})

        instance.reviews_df = instance.training_df[['UserId', 'FoodId', 'Rating', 'Review']].copy()

        user_stats_path = os.path.join(input_dir, 'user_stats.pkl')
        if os.path.exists(user_stats_path):
            instance.user_stats = pd.read_pickle(user_stats_path)
        else:
            instance.user_stats = None

        print(f"✓ Loaded {len(instance.training_df):,} samples")
        print(f"✓ Loaded {len(instance.user_history_dict):,} user histories")
        print(f"✓ Loaded {len(instance.recipes_df):,} recipes")

        return instance

    def _get_user_feature_dim(self) -> int:
        return 5

    def _get_food_feature_dim(self) -> int:
        feature_count = len(self.available_nutrition_cols)
        if 'TotalTimeMinutes' in self.recipes_df.columns:
            feature_count += 1
        return max(feature_count, 1)


class FoodRecommendationDataset(Dataset):
    """
    PyTorch Dataset for food recommendations.

    All tensors are pre-computed as contiguous numpy arrays during __init__.

    """

    def __init__(self, preprocessor: 'FoodDataPreprocessor',
                 df: pd.DataFrame = None):
        self.preprocessor = preprocessor
        self.df           = df if df is not None else preprocessor.training_df

        max_h     = preprocessor.max_history_len
        uid_map   = preprocessor.user_id_map
        fid_map   = preprocessor.food_id_map
        hist_dict = preprocessor.user_history_dict
        nutr_cols = preprocessor.available_nutrition_cols
        max_vals  = preprocessor.max_values
        n         = len(self.df)

        print(f"  Pre-computing tensors for {n:,} samples ...")

        self._user_ids = np.array(
            [uid_map.get(u, 1) for u in self.df['UserId']], dtype=np.int64
        )

        if getattr(preprocessor, 'user_stats', None) is not None:
            user_stats = preprocessor.user_stats
        else:
            user_stats = (
                self.df.groupby('UserId', sort=False)
                .agg(AvgRating=('Rating', 'mean'), NumReviews=('Rating', 'count'))
                .reset_index()
            )
            if 'TotalTimeMinutes' in self.df.columns:
                time_stats = (
                    self.df.groupby('UserId', sort=False)['TotalTimeMinutes']
                    .mean()
                    .rename('AvgTimePreference')
                    .reset_index()
                )
                user_stats = user_stats.merge(time_stats, on='UserId', how='left')
                user_stats['AvgTimePreference'] = user_stats['AvgTimePreference'].fillna(30.0)
            else:
                user_stats['AvgTimePreference'] = 30.0

        per_sample = self.df[['UserId']].merge(user_stats, on='UserId', how='left')

        uf = np.zeros((n, 5), dtype=np.float32)
        uf[:, 0] = np.clip(per_sample['AvgRating'].fillna(0).values        / 5.0,   0.0, 1.0)
        uf[:, 1] = np.clip(per_sample['NumReviews'].fillna(0).values        / 100.0, 0.0, 1.0)
        uf[:, 2] = np.clip(per_sample['AvgTimePreference'].fillna(30).values / 480.0, 0.0, 1.0)
        self._user_features = uf

        hist_arr = np.zeros((n, max_h), dtype=np.int64)
        for i, uid in enumerate(self.df['UserId']):
            h = hist_dict.get(uid, [])[:max_h]
            hist_arr[i, :len(h)] = h
        self._user_history = hist_arr

        self._item_ids = np.array(
            [fid_map.get(f, 0) for f in self.df['FoodId']], dtype=np.int64
        )

        feat_cols = list(nutr_cols)
        if 'TotalTimeMinutes' in self.df.columns:
            feat_cols.append('TotalTimeMinutes')

        n_feat = max(len(feat_cols), 1)
        item_feat = np.zeros((n, n_feat), dtype=np.float32)
        for fi, col in enumerate(feat_cols):
            if col in self.df.columns:
                vals = self.df[col].fillna(0.0).values.astype(np.float32)
                mv   = float(max_vals.get(col, 1000))
                if mv > 0:
                    item_feat[:, fi] = np.clip(vals / mv, 0.0, 1.0)
        self._item_features = item_feat

        # ── 6. Positions ───────────────────────────────────────────────────
        self._positions = self.df['Position'].fillna(0).values.astype(np.int64)

        # ── 7. Labels ──────────────────────────────────────────────────────
        ratings = self.df['Rating'].fillna(2.5).values.astype(np.float32)
        self._labels = np.stack([
            (ratings >= 4.0).astype(np.float32),
            np.clip(ratings / 5.0, 0.0, 1.0),
        ], axis=1)
        nan_arrays = [self._user_features, self._item_features, self._labels]
        if any(np.isnan(a).any() or np.isinf(a).any() for a in nan_arrays):
            raise ValueError(
                "NaN or Inf detected in pre-computed feature arrays. "
                "Check the nutritional columns and max_values."
            )

        print(f"  ✓ Pre-computation complete. No NaN/Inf detected.")

    def __len__(self):
        return len(self._user_ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        """O(1) array index — no pandas, no loops."""
        return (
            torch.from_numpy(self._user_ids     [idx:idx+1]).squeeze(0),
            torch.from_numpy(self._user_features [idx]),
            torch.from_numpy(self._user_history  [idx]),
            torch.from_numpy(self._item_ids      [idx:idx+1]).squeeze(0),
            torch.from_numpy(self._item_features [idx]),
            torch.tensor(self._positions[idx],   dtype=torch.long),
            torch.from_numpy(self._labels        [idx]),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Food Recommendation Data Preprocessor")
    parser.add_argument('--reviews', type=str, default='reviews.csv')
    parser.add_argument('--history', type=str, default='user_history.csv')
    parser.add_argument('--recipes', type=str, default='recipes.csv')
    parser.add_argument('--sample',  type=int, default=None)
    parser.add_argument('--save',    type=str, default=None)
    parser.add_argument('--load',    type=str, default=None)

    QUICK_TEST = True

    if QUICK_TEST:
        _here     = Path(__file__).parent.absolute()
        _root     = _here.parent
        _datasets = _root / 'data' / 'raw'
        _prepdata = _root / 'data' / 'preprocessed'
        args = type('Args', (), {
            'reviews': str(_datasets / 'reviews.csv'),
            'history': str(_datasets / 'user_history.csv'),
            'recipes': str(_datasets / 'recipes.csv'),
            'sample':  None,
            'save':    str(_prepdata),
            'load':    None
        })()
    else:
        args = parser.parse_args()

    print("\n" + "=" * 70)
    print("FOOD RECOMMENDATION DATA LOADER")
    print("=" * 70)

    if args.load:
        preprocessor = FoodDataPreprocessor.load_preprocessed(args.load)
    else:
        preprocessor = FoodDataPreprocessor(
            reviews_path=args.reviews,
            history_path=args.history,
            recipes_path=args.recipes,
            max_history_len=20,
            sample_size=args.sample,
        )
        if args.save:
            preprocessor.save_preprocessed(args.save)

    info = preprocessor.get_dataset_info()
    print("\n" + "=" * 70)
    print("DATASET INFO")
    print("=" * 70)
    print(f"  Number of users:    {info['num_users']:,}")
    print(f"  Number of foods:    {info['num_foods']:,}")
    print(f"  User feature dim:   {info['user_feature_dim']}")
    print(f"  Food feature dim:   {info['food_feature_dim']}")
    print(f"  Max history length: {info['max_history_len']}")

    print("\n" + "=" * 70)
    print("CREATING PYTORCH DATASET (train split only)")
    print("=" * 70)
    dataset = FoodRecommendationDataset(preprocessor, df=preprocessor.train_df)
    print(f"✓ Train dataset: {len(dataset):,} samples")

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    for batch in dataloader:
        user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch
        print(f"  user_ids shape:      {user_ids.shape}")
        print(f"  user_features shape: {user_features.shape}")
        print(f"  user_history shape:  {user_history.shape}")
        print(f"  item_ids shape:      {item_ids.shape}")
        print(f"  item_features shape: {item_features.shape}")
        print(f"  positions shape:     {positions.shape}")
        print(f"  labels shape:        {labels.shape}")
        print("\n✓ Batch loaded successfully!")
        break

    print("=" * 70)
    print("✓ DATASET READY FOR TRAINING!")
    print("=" * 70)
