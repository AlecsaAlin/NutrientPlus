import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from collections import defaultdict
import pickle
import os
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

        # Load reviews (optionally sample for testing)
        if sample_size:
            print(f"  Sampling {sample_size} reviews for testing...")
            self.reviews_df = pd.read_csv(reviews_path, nrows=sample_size, encoding='utf-8', encoding_errors='ignore')
        else:
            self.reviews_df = pd.read_csv(reviews_path, encoding='utf-8', encoding_errors='ignore')

        # Use low_memory=False to avoid dtype warnings on large files
        self.history_df = pd.read_csv(history_path, low_memory=False, encoding='utf-8', encoding_errors='ignore')
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

        self.max_values = {
            'Calories': 19823.5,
            'ProteinContent': 1878.3,
            'FatContent': 2191.0,
            'CarbohydrateContent': 5030.7,
            'SugarContent': 4570.9,
            'FiberContent': 835.7,
            'TotalTimeMinutes': 480,
            'CholesterolContent':9167.2,
            'SodiumContent':1246921.1,
            'CategoryID':312
        }

        # Identify available nutritional columns
        self.available_nutrition_cols = []
        possible_cols = ['Calories', 'ProteinContent','FatContent','SugarContent','FiberContent',
                         'CarbohydrateContent','CholesterolContent','SodiumContent','CategoryID']


        for col in possible_cols:
            if col in self.recipes_df.columns:
                self.available_nutrition_cols.append(col)
                # Convert to numeric
                self.recipes_df[col] = pd.to_numeric(self.recipes_df[col], errors='coerce')

                # ✅ BETTER DEFAULT VALUES
                if col == 'Calories':
                    default_val = 200  # Reasonable default for a recipe
                elif col == 'ProteinContent':
                    default_val = 10
                elif col == 'FatContent':
                    default_val = 8
                elif col == 'CarbohydrateContent':
                    default_val = 25
                elif col == 'SugarContent':
                    default_val = 5
                elif col == 'FiberContent':
                    default_val = 3
                else:
                    default_val = 0

                median_val = self.recipes_df[col].median()
                fill_val = median_val if not pd.isna(median_val) else default_val

                self.recipes_df.loc[:, col] = self.recipes_df[col].fillna(fill_val)

                # ✅ CAP AT MAX VALUE
                max_val = self.max_values.get(col, 1000)
                self.recipes_df.loc[self.recipes_df[col] > max_val, col] = max_val
                self.recipes_df.loc[self.recipes_df[col] < 0, col] = default_val

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
            # ✅ CAP EXTREME TIME VALUES
            self.recipes_df.loc[self.recipes_df['TotalTimeMinutes'] > 1440, 'TotalTimeMinutes'] = 1440  # Max 24 hours

    def _create_mappings(self):
        """Create ID mappings for users and foods."""
        print("Creating ID mappings...")

        # Get all unique users and foods efficiently
        all_users = set(self.reviews_df['UserId'].unique())
        all_users.update(self.history_df['UserId'].unique())

        all_foods = set(self.reviews_df['FoodId'].unique())
        all_foods.update(self.recipes_df['FoodId'].unique())

        print(f"Found {len(all_users)} unique users and {len(all_foods)} unique foods")

        # Create mappings (start from 1, reserve 0 for padding)
        self.user_id_map = {uid: idx + 1 for idx, uid in enumerate(sorted(all_users))}
        self.food_id_map = {fid: idx + 1 for idx, fid in enumerate(sorted(all_foods))}

        # Reverse mappings
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_food_map = {v: k for k, v in self.food_id_map.items()}

        print(f"Created mappings for {len(self.user_id_map)} users and {len(self.food_id_map)} foods")

        # Create user history dictionary efficiently
        print("Creating user history dictionary...")
        self.user_history_dict = {}

        for idx in range(len(self.history_df)):
            user_id = self.history_df.iloc[idx]['UserId']
            history = self.history_df.iloc[idx]['parsed_history'][:self.max_history_len]
            # Map food IDs
            mapped_history = [self.food_id_map.get(fid, 0) for fid in history if fid in self.food_id_map]
            self.user_history_dict[user_id] = mapped_history

        print(f"✓ User history dictionary created for {len(self.user_history_dict)} users")

    def _prepare_training_data(self):
        """Prepare the training dataframe by merging reviews with recipe features and split into train/test."""
        # Merge reviews with recipe features
        full_df = self.reviews_df.merge(
            self.recipes_df,
            on='FoodId',
            how='left'
        )

        print("Generating Position based on Rating...")

        # Sort by rating in descending order (best ratings first)
        full_df = full_df.sort_values('Rating', ascending=False).reset_index(drop=True)

        # Assign position within each user's history
        full_df['Position'] = full_df.groupby('UserId').cumcount()

        # Cap positions at 9 (top 10 items)
        full_df['Position'] = full_df['Position'].clip(upper=9)

        # Create binary engagement signals
        full_df['HighRating'] = (full_df['Rating'] >= 4).astype(float)
        full_df['HasReview'] = (~full_df['Review'].isna()).astype(float)

        # Split into train and test sets
        print(f"\nSplitting data into train ({int((1-self.test_size)*100)}%) and test ({int(self.test_size*100)}%) sets...")
        self.train_df, self.test_df = train_test_split(
            full_df,
            test_size=self.test_size,
            random_state=None  # No fixed seed - different split each time
        )

        # Reset indices
        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)

        # Store full dataframe for backward compatibility
        self.training_df = full_df

        print(f"✓ Total samples: {len(full_df):,}")
        print(f"✓ Training samples: {len(self.train_df):,}")
        print(f"✓ Testing samples: {len(self.test_df):,}")
        print(f"\nPosition distribution in training set:\n{self.train_df['Position'].value_counts().sort_index()}")
        print(f"\nPosition distribution in testing set:\n{self.test_df['Position'].value_counts().sort_index()}")

    def get_dataset_info(self) -> Dict:
        """Returns information needed to initialize the model."""
        return {
            'num_users': len(self.user_id_map),
            'num_foods': len(self.food_id_map),
            'user_feature_dim': self._get_user_feature_dim(),
            'food_feature_dim': self._get_food_feature_dim(),
            'max_history_len': self.max_history_len,
        }

    def save_preprocessed(self, output_dir: str):
        """
        Save preprocessed data for fast loading later.
        Creates separate 'train' and 'test' subdirectories.

        Args:
            output_dir: Base directory to save preprocessed files
        """
        # Create train and test directories
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        print(f"\nSaving preprocessed data to {output_dir}...")
        print(f"  Train folder: {train_dir}")
        print(f"  Test folder: {test_dir}")

        # Save TRAIN dataframes
        self.train_df.to_pickle(os.path.join(train_dir, 'training_df.pkl'))
        self.history_df.to_pickle(os.path.join(train_dir, 'history_df.pkl'))
        self.recipes_df.to_pickle(os.path.join(train_dir, 'recipes_df.pkl'))

        # Save TEST dataframes
        self.test_df.to_pickle(os.path.join(test_dir, 'training_df.pkl'))
        self.history_df.to_pickle(os.path.join(test_dir, 'history_df.pkl'))
        self.recipes_df.to_pickle(os.path.join(test_dir, 'recipes_df.pkl'))

        # Save mappings and metadata (same for both)
        metadata = {
            'user_id_map': self.user_id_map,
            'food_id_map': self.food_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_food_map': self.reverse_food_map,
            'user_history_dict': self.user_history_dict,
            'available_nutrition_cols': self.available_nutrition_cols,
            'max_history_len': self.max_history_len,
            'test_size': self.test_size,
            'max_values': self.max_values,
        }

        with open(os.path.join(train_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        with open(os.path.join(test_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"\n✓ TRAIN folder:")
        print(f"  - Saved training_df.pkl ({len(self.train_df):,} samples)")
        print(f"  - Saved history_df.pkl ({len(self.history_df):,} users)")
        print(f"  - Saved recipes_df.pkl ({len(self.recipes_df):,} recipes)")
        print(f"  - Saved metadata.pkl")
        
        print(f"\n✓ TEST folder:")
        print(f"  - Saved training_df.pkl ({len(self.test_df):,} samples)")
        print(f"  - Saved history_df.pkl ({len(self.history_df):,} users)")
        print(f"  - Saved recipes_df.pkl ({len(self.recipes_df):,} recipes)")
        print(f"  - Saved metadata.pkl")
        
        print(f"\n✓ Preprocessing complete! Saved to {output_dir}")
        print(f"  Use '{train_dir}' for training")
        print(f"  Use '{test_dir}' for testing/evaluation")

    @classmethod
    def load_preprocessed(cls, input_dir: str):
        """
        Load preprocessed data quickly without reprocessing.

        Args:
            input_dir: Directory containing preprocessed files

        Returns:
            FoodDataPreprocessor instance with loaded data
        """
        print(f"Loading preprocessed data from {input_dir}...")

        # Create empty instance
        instance = cls.__new__(cls)

        # Load dataframes
        instance.training_df = pd.read_pickle(os.path.join(input_dir, 'training_df.pkl'))
        instance.history_df = pd.read_pickle(os.path.join(input_dir, 'history_df.pkl'))
        instance.recipes_df = pd.read_pickle(os.path.join(input_dir, 'recipes_df.pkl'))

        # Load metadata
        with open(os.path.join(input_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        instance.user_id_map = metadata['user_id_map']
        instance.food_id_map = metadata['food_id_map']
        instance.reverse_user_map = metadata['reverse_user_map']
        instance.reverse_food_map = metadata['reverse_food_map']
        instance.user_history_dict = metadata['user_history_dict']
        instance.available_nutrition_cols = metadata['available_nutrition_cols']
        instance.max_history_len = metadata['max_history_len']
        instance.test_size = metadata.get('test_size', 0.2)
        instance.max_values = metadata.get('max_values', {})

        # Also need reviews_df for dataset creation
        instance.reviews_df = instance.training_df[['UserId', 'FoodId', 'Rating', 'Review']].copy()

        print(f"✓ Loaded {len(instance.training_df):,} training samples")
        print(f"✓ Loaded {len(instance.user_history_dict):,} user histories")
        print(f"✓ Loaded {len(instance.recipes_df):,} recipes")

        return instance

    def _get_user_feature_dim(self) -> int:
        """Calculate user feature dimension (basic demographic features)."""
        return 5  # avg_rating, num_reviews, avg_cook_time_preference, placeholder, placeholder

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
            # Fill NaN time preferences with default
            user_stats['AvgTimePreference'] = user_stats['AvgTimePreference'].fillna(30.0)
        else:
            user_stats['AvgTimePreference'] = 30.0

        self.user_features_df = user_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        row = self.df.iloc[idx]

        # 1. User ID
        user_id = self.user_id_map.get(row['UserId'], 1)
        user_id_tensor = torch.tensor(user_id, dtype=torch.long)

        # 2. User Features (with safe normalization!)
        user_stats = self.user_features_df[
            self.user_features_df['UserId'] == row['UserId']
            ]

        if len(user_stats) > 0:
            avg_rating = user_stats['AvgRating'].values[0]
            num_reviews = user_stats['NumReviews'].values[0]
            avg_time = user_stats['AvgTimePreference'].values[0]

            # ✅ SAFE NORMALIZATION WITH CLIPPING
            user_features = torch.tensor([
                np.clip(avg_rating / 5.0, 0, 1),  # Clip to [0, 1]
                np.clip(num_reviews / 100.0, 0, 1),  # Clip to [0, 1]
                np.clip(avg_time / 480.0, 0, 1),  # Allow up to 540 min, then clip
                0.0,
                0.0,
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

        # 5. Food Features - dynamically build based on available columns (with safe normalization!)
        item_features_list = []

        # Add nutritional features that are available
        for col in self.available_nutrition_cols:
            if col in row.index and not pd.isna(row[col]):
                val = float(row[col])
                max_val = self.preprocessor.max_values.get(col, 1000)
                normalized_val = val / max_val
                normalized_val = np.clip(normalized_val, 0.0, 1.0)
                item_features_list.append(normalized_val)
            else:
                item_features_list.append(0.0)

        # Add time feature if available
        if 'TotalTimeMinutes' in row.index and not pd.isna(row['TotalTimeMinutes']):
            time_val = float(row['TotalTimeMinutes'])
            max_time = self.preprocessor.max_values.get('TotalTimeMinutes', 480)
            normalized_time = np.clip(time_val / max_time, 0.0, 1.0)
            item_features_list.append(normalized_time)
        elif 'TotalTimeMinutes' in self.df.columns:
            item_features_list.append(0.0)

        # If no features available, use a dummy feature
        if not item_features_list:
            item_features_list = [0.0]

        item_features = torch.tensor(item_features_list, dtype=torch.float32)

        # ✅ SAFETY CHECK FOR NaN/Inf BEFORE RETURNING
        if torch.isnan(item_features).any() or torch.isinf(item_features).any():
            print(f"⚠️ Warning: NaN/Inf in item_features at idx {idx}, replacing with zeros")
            item_features = torch.nan_to_num(item_features, nan=0.0, posinf=1.0, neginf=0.0)

        if torch.isnan(user_features).any() or torch.isinf(user_features).any():
            print(f"⚠️ Warning: NaN/Inf in user_features at idx {idx}, replacing with defaults")
            user_features = torch.nan_to_num(user_features, nan=0.0, posinf=1.0, neginf=0.0)

        # 6. Position (from rating-based ranking)
        position = torch.tensor(row.get('Position', 0), dtype=torch.long)

        # 7. Labels (multi-task: [high_rating, normalized_rating, has_review])
        rating = row['Rating']
        if pd.isna(rating):
            rating = 2.5  # Default to middle rating if missing

        labels = torch.tensor([
            float(rating >= 4.0),  # binary: rating >= 4
            np.clip(rating / 5.0, 0, 1),  # ✅ normalized rating with clip
            float(row['HasReview']) if not pd.isna(row['HasReview']) else 0.0,  # binary: wrote review
        ], dtype=torch.float32)

        return (user_id_tensor, user_features, user_history,
                item_id_tensor, item_features, position, labels)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Food Recommendation Data Preprocessor")
    parser.add_argument('--reviews', type=str, default='reviews.csv',
                        help='Path to reviews CSV file')
    parser.add_argument('--history', type=str, default='user_history.csv',
                        help='Path to user history CSV file')
    parser.add_argument('--recipes', type=str, default='recipes.csv',
                        help='Path to recipes CSV file')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N reviews for quick testing')
    parser.add_argument('--save', type=str, default=None,
                        help='Save preprocessed data to this directory')
    parser.add_argument('--load', type=str, default=None,
                        help='Load preprocessed data from this directory')

    # Quick test mode - uncomment and modify paths for your setup
    QUICK_TEST = True  # Set to True to enable quick test

    if QUICK_TEST:
        args = type('Args', (), {
            'reviews': '../../Datasets/reviews.csv',
            'history': '../../Datasets/user_history.csv',
            'recipes': '../../Datasets/recipes.csv',
            'sample': None,
            'save': '../../preprocessed_data',
            'load': None
        })()
    else:
        args = parser.parse_args()

    # Initialize preprocessor
    print("\n" + "=" * 70)
    print("FOOD RECOMMENDATION DATA LOADER ")
    print("=" * 70)

    if args.load:
        # Load preprocessed data (much faster!)
        preprocessor = FoodDataPreprocessor.load_preprocessed(args.load)
    else:
        # Process from CSV files
        preprocessor = FoodDataPreprocessor(
            reviews_path=args.reviews,
            history_path=args.history,
            recipes_path=args.recipes,
            max_history_len=20,
            sample_size=args.sample
        )

        # Save if requested
        if args.save:
            preprocessor.save_preprocessed(args.save)

    # Get dataset info
    info = preprocessor.get_dataset_info()
    print("\n" + "=" * 70)
    print("DATASET INFO")
    print("=" * 70)
    print(f"  Number of users: {info['num_users']:,}")
    print(f"  Number of foods: {info['num_foods']:,}")
    print(f"  User feature dim: {info['user_feature_dim']}")
    print(f"  Food feature dim: {info['food_feature_dim']}")
    print(f"  Max history length: {info['max_history_len']}")

    # Create PyTorch dataset
    print("\n" + "=" * 70)
    print("CREATING PYTORCH DATASET")
    print("=" * 70)
    dataset = FoodRecommendationDataset(preprocessor)
    print(f"✓ Dataset created with {len(dataset):,} samples")

    # ✅ VALIDATE DATA FOR NaN/Inf
    print("\n" + "=" * 70)
    print("VALIDATING DATA FOR NaN/Inf")
    print("=" * 70)

    nan_count = 0
    inf_count = 0
    check_samples = 10000

    for i in range(check_samples):
        sample = dataset[i]
        user_ids, user_features, user_history, item_ids, item_features, positions, labels = sample

        if torch.isnan(user_features).any():
            nan_count += 1
        if torch.isnan(item_features).any():
            nan_count += 1
        if torch.isnan(labels).any():
            nan_count += 1

        if torch.isinf(user_features).any():
            inf_count += 1
        if torch.isinf(item_features).any():
            inf_count += 1

    if nan_count == 0 and inf_count == 0:
        print(f"✅ No NaN/Inf found in first {check_samples} samples!")
    else:
        print(f"⚠️ Found {nan_count} NaN and {inf_count} Inf issues in first {check_samples} samples")

    # Create dataloader
    print("\n" + "=" * 70)
    print("TESTING BATCH LOADING")
    print("=" * 70)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows, >0 for Linux/Mac
    )

    # Test loading a batch
    for batch in dataloader:
        user_ids, user_features, user_history, item_ids, item_features, positions, labels = batch
        print(f"  User IDs shape: {user_ids.shape}")
        print(f"  User features shape: {user_features.shape}")
        print(f"  User history shape: {user_history.shape}")
        print(f"  Item IDs shape: {item_ids.shape}")
        print(f"  Item features shape: {item_features.shape}")
        print(f"  Positions shape: {positions.shape}")
        print(f"  Labels shape: {labels.shape}")
        print("\n✓ Batch loaded successfully!")
        break

    print("=" * 70)
    print("✓ DATASET READY FOR TRAINING!")
    print("=" * 70)