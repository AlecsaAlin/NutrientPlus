import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from collections import defaultdict


class FoodDataPreprocessor:
    """Preprocesses food recommendation data for two-tower model training."""

    def __init__(
            self,
            reviews_path: str,
            history_path: str,
            recipes_path: str,
            max_history_len: int = 20
    ):
        """
        Args:
            reviews_path: Path to reviews CSV (ReviewId, RecipeId, AuthorId, Rating, Review, DateSubmitted)
            history_path: Path to history CSV (user_id, history)
            recipes_path: Path to recipes CSV (RecipeId, Name, AuthorId, CookTime, etc.)
            max_history_len: Maximum length of user history sequence
        """
        self.max_history_len = max_history_len

        print("Loading datasets...")
        self.reviews_df = pd.read_csv(reviews_path)
        # Use low_memory=False to avoid dtype warnings on large files
        self.history_df = pd.read_csv(history_path, low_memory=False)
        self.recipes_df = pd.read_csv(recipes_path)

        print(f"Loaded {len(self.reviews_df)} reviews")
        print(f"Loaded {len(self.history_df)} user histories")
        print(f"Loaded {len(self.recipes_df)} recipes")

        self._preprocess_data()
        self._create_mappings()
        self._prepare_training_data()

    def _preprocess_data(self):
        """Clean and preprocess the raw data."""
        # Clean reviews
        self.reviews_df['Rating'] = pd.to_numeric(self.reviews_df['Rating'], errors='coerce')
        self.reviews_df = self.reviews_df.dropna(subset=['Rating', 'RecipeId'])

        # Rename columns for consistency
        self.reviews_df.rename(columns={
            'AuthorId': 'UserId',
            'RecipeId': 'FoodId'
        }, inplace=True)

        # Parse history - convert string representation to list
        def parse_history(x):
            if pd.isna(x) or x == '[]' or x == '':
                return []
            # Convert to string and clean
            x = str(x).strip()
            # Remove brackets if present
            x = x.strip('[]')
            # Remove quotes if present
            x = x.strip('"\'')
            # Remove all spaces
            x = x.replace(' ', '')
            if not x:
                return []
            # Split by comma and convert to int
            try:
                return [int(i) for i in x.split(',') if i]
            except ValueError:
                return []

        # Parse history column
        self.history_df['parsed_history'] = self.history_df['history'].apply(parse_history)

        # Rename user_id column for consistency
        self.history_df.rename(columns={'user_id': 'UserId'}, inplace=True)

        # Clean recipes
        self.recipes_df.rename(columns={'RecipeId': 'FoodId'}, inplace=True)

        # Identify available nutritional columns
        self.available_nutrition_cols = []
        possible_cols = ['Calories','ProteinContent']

        for col in possible_cols:
            if col in self.recipes_df.columns:
                self.available_nutrition_cols.append(col)
                # Convert to numeric
                self.recipes_df[col] = pd.to_numeric(self.recipes_df[col], errors='coerce')
                # Fill missing values using .loc to avoid warning
                median_val = self.recipes_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                self.recipes_df.loc[:, col] = self.recipes_df[col].fillna(median_val)

        print(f"Available nutritional columns: {self.available_nutrition_cols}")

        # Parse time columns (assuming format like "PT30M" for 30 minutes)
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
            except:
                return 0

        if 'CookTime' in self.recipes_df.columns:
            self.recipes_df['CookTimeMinutes'] = self.recipes_df['CookTime'].apply(parse_time)
        if 'PrepTime' in self.recipes_df.columns:
            self.recipes_df['PrepTimeMinutes'] = self.recipes_df['PrepTime'].apply(parse_time)
        if 'TotalTime' in self.recipes_df.columns:
            self.recipes_df['TotalTimeMinutes'] = self.recipes_df['TotalTime'].apply(parse_time)

    def _create_mappings(self):
        """Create ID mappings for users and foods."""
        # Get all unique users and foods
        all_users = set(self.reviews_df['UserId'].unique())
        all_users.update(self.history_df['UserId'].unique())

        all_foods = set(self.reviews_df['FoodId'].unique())
        all_foods.update(self.recipes_df['FoodId'].unique())

        # Add foods from history
        for hist in self.history_df['parsed_history']:
            all_foods.update(hist)

        # Create mappings (start from 1, reserve 0 for padding)
        self.user_id_map = {uid: idx + 1 for idx, uid in enumerate(sorted(all_users))}
        self.food_id_map = {fid: idx + 1 for idx, fid in enumerate(sorted(all_foods))}

        # Reverse mappings
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_food_map = {v: k for k, v in self.food_id_map.items()}

        print(f"Created mappings for {len(self.user_id_map)} users and {len(self.food_id_map)} foods")

        # Create user history dictionary
        self.user_history_dict = {}
        for _, row in self.history_df.iterrows():
            user_id = row['UserId']
            history = row['parsed_history'][:self.max_history_len]  # Take most recent
            # Map food IDs
            mapped_history = [self.food_id_map.get(fid, 0) for fid in history if fid in self.food_id_map]
            self.user_history_dict[user_id] = mapped_history

    def _prepare_training_data(self):
        """Prepare the training dataframe by merging reviews with recipe features."""
        # Merge reviews with recipe features
        self.training_df = self.reviews_df.merge(
            self.recipes_df,
            on='FoodId',
            how='left'
        )

        # Add position (simulate position in recommendation list)
        # In real scenario, you'd have actual position data
        self.training_df['Position'] = np.random.randint(0, 10, size=len(self.training_df))

        # Create binary engagement signals
        self.training_df['HighRating'] = (self.training_df['Rating'] >= 4).astype(float)
        self.training_df['HasReview'] = (~self.training_df['Review'].isna()).astype(float)

        print(f"Prepared {len(self.training_df)} training samples")

    def get_dataset_info(self) -> Dict:
        """Returns information needed to initialize the model."""
        return {
            'num_users': len(self.user_id_map),
            'num_foods': len(self.food_id_map),
            'user_feature_dim': self._get_user_feature_dim(),
            'food_feature_dim': self._get_food_feature_dim(),
            'max_history_len': self.max_history_len,
        }

    def _get_user_feature_dim(self) -> int:
        """Calculate user feature dimension (basic demographic features)."""
        return 5  # avg_rating, num_reviews, avg_cook_time_preference, etc.

    def _get_food_feature_dim(self) -> int:
        """Calculate food feature dimension based on available recipe columns."""
        # Count available features
        feature_count = len(self.available_nutrition_cols)

        # Add time feature if available
        if 'TotalTimeMinutes' in self.recipes_df.columns:
            feature_count += 1

        return max(feature_count, 1)  # At least 1 feature


class FoodRecommendationDataset(Dataset):
    """PyTorch Dataset for food recommendations."""

    def __init__(self, preprocessor: FoodDataPreprocessor):
        self.preprocessor = preprocessor
        self.df = preprocessor.training_df
        self.user_history_dict = preprocessor.user_history_dict
        self.user_id_map = preprocessor.user_id_map
        self.food_id_map = preprocessor.food_id_map
        self.max_history_len = preprocessor.max_history_len
        self.available_nutrition_cols = preprocessor.available_nutrition_cols

        # Compute user-level features
        self._compute_user_features()

    def _compute_user_features(self):
        """Compute aggregated features for each user."""
        user_stats = self.df.groupby('UserId').agg({
            'Rating': ['mean', 'count'],
        }).reset_index()

        user_stats.columns = ['UserId', 'AvgRating', 'NumReviews']

        # Add time preference if available
        if 'TotalTimeMinutes' in self.df.columns:
            time_stats = self.df.groupby('UserId')['TotalTimeMinutes'].mean().reset_index()
            time_stats.columns = ['UserId', 'AvgTimePreference']
            user_stats = user_stats.merge(time_stats, on='UserId', how='left')
        else:
            user_stats['AvgTimePreference'] = 30.0

        self.user_features_df = user_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        row = self.df.iloc[idx]

        # 1. User ID
        user_id = self.user_id_map[row['UserId']]
        user_id_tensor = torch.tensor(user_id, dtype=torch.long)

        # 2. User Features
        user_stats = self.user_features_df[
            self.user_features_df['UserId'] == row['UserId']
            ]

        if len(user_stats) > 0:
            user_features = torch.tensor([
                user_stats['AvgRating'].values[0] / 5.0,  # normalize to 0-1
                min(user_stats['NumReviews'].values[0] / 100.0, 1.0),  # cap at 100
                user_stats['AvgTimePreference'].values[0] / 180.0,  # normalize by 3 hours
                0.0,  # placeholder for dietary preference
                0.0,  # placeholder for another feature
            ], dtype=torch.float32)
        else:
            user_features = torch.zeros(5, dtype=torch.float32)

        # 3. User History
        history = self.user_history_dict.get(row['UserId'], [])
        history = history[:self.max_history_len]  # truncate if needed

        # Pad to max_history_len
        while len(history) < self.max_history_len:
            history.append(0)  # 0 is padding

        user_history = torch.tensor(history, dtype=torch.long)

        # 4. Food/Item ID
        item_id = self.food_id_map.get(row['FoodId'], 0)
        item_id_tensor = torch.tensor(item_id, dtype=torch.long)

        # 5. Food Features - dynamically build based on available columns
        item_features_list = []

        # Add nutritional features that are available
        for col in self.available_nutrition_cols:
            if col in row.index and not pd.isna(row[col]):
                # Normalize based on typical daily values
                if col == 'Calories':
                    item_features_list.append(float(row[col]) / 2000.0)
                elif col == 'ProteinContent':
                    item_features_list.append(float(row[col]) / 50.0)
                    # Generic normalization for other nutrients
                    item_features_list.append(float(row[col]) / 100.0)
            else:
                item_features_list.append(0.0)

        # Add time feature if available
        if 'TotalTimeMinutes' in row.index and not pd.isna(row['TotalTimeMinutes']):
            item_features_list.append(float(row['TotalTimeMinutes']) / 180.0)
        elif 'TotalTimeMinutes' in self.df.columns:
            item_features_list.append(0.0)

        # If no features available, use a dummy feature
        if not item_features_list:
            item_features_list = [0.0]

        item_features = torch.tensor(item_features_list, dtype=torch.float32)

        # 6. Position
        position = torch.tensor(row.get('Position', 0), dtype=torch.long)

        # 7. Labels (multi-task: [high_rating, normalized_rating, has_review])
        labels = torch.tensor([
            float(row['HighRating']),  # binary: rating >= 4
            row['Rating'] / 5.0,  # normalized rating
            float(row['HasReview']),  # binary: wrote review
        ], dtype=torch.float32)

        return (user_id_tensor, user_features, user_history,
                item_id_tensor, item_features, position, labels)


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = FoodDataPreprocessor(
        reviews_path="reviews.csv",
        history_path="user_history.csv",
        recipes_path="recipes.csv",
        max_history_len=20
    )

    # Get dataset info
    info = preprocessor.get_dataset_info()
    print("\nDataset Info:")
    print(f"  Number of users: {info['num_users']}")
    print(f"  Number of foods: {info['num_foods']}")
    print(f"  User feature dim: {info['user_feature_dim']}")
    print(f"  Food feature dim: {info['food_feature_dim']}")
    print(f"  Max history length: {info['max_history_len']}")

    # Create PyTorch dataset
    dataset = FoodRecommendationDataset(preprocessor)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows, >0 for Linux/Mac
    )

    # Test loading a batch
    print("\nTesting batch loading...")
    for batch in dataloader:
        user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch
        print(f"  User IDs shape: {user_ids.shape}")
        print(f"  User features shape: {user_features.shape}")
        print(f"  User history shape: {user_history.shape}")
        print(f"  Item IDs shape: {item_ids.shape}")
        print(f"  Item features shape: {item_features.shape}")
        print(f"  Positions shape: {positions.shape}")
        print(f"  Labels shape: {labels.shape}")
        break

    print("\nDataset ready for training!")