import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from tqdm import tqdm
import argparse
import pickle
import itertools
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import sys
from joblib import Parallel, delayed, parallel_backend

# Suppress warnings
warnings.filterwarnings("ignore")

# Check for XGBoost
try:
    import xgboost as xgb
    HAVE_XGB = True
    
    # Check XGBoost version
    xgb_version = tuple(map(int, xgb.__version__.split('.')))
    XGB_VERSION_2_PLUS = xgb_version[0] >= 2
    
    # Check if GPU is available for XGBoost
    try:
        if XGB_VERSION_2_PLUS:
            # XGBoost 2.0+ syntax
            gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
        else:
            # XGBoost pre-2.0 syntax
            gpu_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            
        test_model = xgb.XGBClassifier(**gpu_params)
        XGB_GPU = True
        print(f"XGBoost GPU acceleration is available (version {xgb.__version__})")
        
        if XGB_VERSION_2_PLUS:
            print("Using XGBoost 2.0+ GPU syntax: tree_method='hist', device='cuda'")
        else:
            print("Using XGBoost pre-2.0 GPU syntax: tree_method='gpu_hist'")
            
    except Exception as e:
        XGB_GPU = False
        print(f"XGBoost GPU acceleration not available, using CPU. Error: {e}")
except ImportError:
    HAVE_XGB = False
    XGB_GPU = False
    XGB_VERSION_2_PLUS = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import umap
    HAVE_UMAP = True
    
    # Check if CUDA is available for UMAP
    try:
        import cuml
        HAVE_CUML = True
        print("UMAP GPU acceleration (cuML) is available")
    except ImportError:
        HAVE_CUML = False
        print("UMAP GPU acceleration not available, using CPU")
except ImportError:
    HAVE_UMAP = False
    HAVE_CUML = False
    print("UMAP not available. Install with: pip install umap-learn")

try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
    
    # Check if GPU is available for pytorch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"PyTorch GPU acceleration available: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch GPU acceleration not available, using CPU")
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    CUDA_AVAILABLE = False
    print("Sentence Transformers not available. Install with: pip install sentence-transformers")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    HAVE_NLTK = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    HAVE_NLTK = False
    print("NLTK not available. Install with: pip install nltk")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    HAVE_VADER = True
except ImportError:
    HAVE_VADER = False
    print("VADER not available. Install with: pip install vaderSentiment")
    
    # Create a mock sentiment analyzer
    class MockSentimentAnalyzer:
        def polarity_scores(self, text):
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    
    sentiment_analyzer = MockSentimentAnalyzer()

# Define relation class mapping
RELATION_MAP = {
    -1.0: "Destructive Disagreement",
    -0.5: "Destructive Agreement",
    0.0: "Rephrase",
    0.5: "Constructive Agreement",
    1.0: "Constructive Disagreement"
}

# Simplified relation map for tables
SIMPLE_RELATION_MAP = {
    -1.0: "destructive_attack",
    -0.5: "destructive_agreement",
    0.0: "rephrase",
    0.5: "constructive_agreement",
    1.0: "constructive_attack"
}

# Define chain categories
CHAIN_CATEGORIES = {
    "Highly Destructive": (-1.0, -0.75),
    "Moderately Destructive": (-0.75, -0.25),
    "Slightly Destructive-Constructive_Neutral": (-0.25, 0.25),
    "Constructive": (0.25, 1.0)
}

class ProgressEstimator:
    """Class to track progress and estimate completion time."""
    def __init__(self, total_tasks, description="Progress"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, increment=1):
        """Update progress and print estimated time remaining."""
        self.completed_tasks += increment
        
        # Calculate progress percentage
        progress = self.completed_tasks / self.total_tasks
        
        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - self.start_time
        if progress > 0:
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            
            # Format times for display
            elapsed_str = str(timedelta(seconds=int(elapsed_time)))
            remaining_str = str(timedelta(seconds=int(remaining_time)))
            
            # Print progress bar and time estimates
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            sys.stdout.write(f"\r{self.description}: [{bar}] {progress*100:.1f}% | "
                            f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            sys.stdout.flush()
            
            if self.completed_tasks == self.total_tasks:
                print()  # Add a newline when complete
        else:
            sys.stdout.write(f"\r{self.description}: [{'░' * bar_length}] 0.0% | "
                            f"Elapsed: {str(timedelta(seconds=int(elapsed_time)))} | Remaining: unknown")
            sys.stdout.flush()

def load_json_data(file_path):
    """Load JSON data with error handling."""
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def extract_comments_from_chains(data):
    """Extract individual comments from the chain data."""
    comments = []

    for chain_id, chain_data in data['discussions'].items():
        for message in chain_data['messages']:
            # Skip root comments and comments without divisiveness scores
            if message.get('Level', 0) > 0 and 'divisiveness_score' in message:
                # Add chain_id to the message for later reference
                message['chain_id'] = chain_id
                comments.append(message)

    print(f"Extracted {len(comments)} comments from chains")
    return comments

def comments_to_dataframe(comments):
    """Convert comments list to DataFrame."""
    df = pd.DataFrame(comments)
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")

    # Check for missing divisiveness scores
    missing = df['divisiveness_score'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} comments missing divisiveness scores")
        df = df.dropna(subset=['divisiveness_score'])
        print(f"Dropped comments with missing scores, {len(df)} remain")

    return df

def extract_text_features(df):
    """Extract linguistic features from comment text."""
    print("Extracting text features...")
    features = pd.DataFrame(index=df.index)

    # Check if required columns exist
    if 'CommentText' not in df.columns or 'ParentCommentText' not in df.columns:
        print("Warning: Comment text columns not found")
        return features

    # Fill NaN values with empty strings
    df['CommentText'] = df['CommentText'].fillna('')
    df['ParentCommentText'] = df['ParentCommentText'].fillna('')

    # Define simple functions that don't require NLTK
    def count_words_simple(text):
        return len(text.split())

    def count_words(text):
        if HAVE_NLTK:
            tokens = word_tokenize(text.lower())
            return len(tokens)
        else:
            return count_words_simple(text)

    def capitals_ratio(text):
        if not text or len(text) == 0:
            return 0
        capitals = sum(1 for c in text if c.isupper())
        return capitals / len(text)

    def count_punctuation(text, punct):
        return text.count(punct) / (len(text) + 1)

    # Extract features from child comments
    progress = tqdm(total=7, desc="Text Features")
    
    print("Processing child comments...")
    features['word_count'] = df['CommentText'].apply(count_words)
    progress.update(1)
    
    features['capitals_ratio'] = df['CommentText'].apply(capitals_ratio)
    progress.update(1)
    
    features['question_marks'] = df['CommentText'].apply(lambda x: count_punctuation(x, '?'))
    progress.update(1)
    
    features['exclamation_marks'] = df['CommentText'].apply(lambda x: count_punctuation(x, '!'))
    progress.update(1)

    # Extract sentiment features
    print("Calculating sentiment scores...")
    sentiment_features = df['CommentText'].apply(lambda x: pd.Series(sentiment_analyzer.polarity_scores(x)))
    features['sentiment_neg'] = sentiment_features['neg']
    features['sentiment_pos'] = sentiment_features['pos']
    features['sentiment_neu'] = sentiment_features['neu']
    features['sentiment_compound'] = sentiment_features['compound']
    progress.update(1)

    # Extract some parent text features for comparison
    print("Processing parent comments...")
    features['parent_word_count'] = df['ParentCommentText'].apply(count_words)
    features['parent_capitals_ratio'] = df['ParentCommentText'].apply(capitals_ratio)
    progress.update(1)

    # Calculate some derived features
    features['word_count_diff'] = features['word_count'] - features['parent_word_count']
    features['word_count_ratio'] = features['word_count'] / (features['parent_word_count'] + 1)
    progress.update(1)

    progress.close()
    print(f"Created {len(features.columns)} text features")
    return features

def extract_text_embeddings(df, model_name='roberta-base-nli-stsb-mean-tokens', batch_size=32):
    """Extract text embeddings using pre-trained transformer models."""
    print(f"Extracting text embeddings using {model_name}...")

    if not HAVE_SENTENCE_TRANSFORMERS:
        print("Warning: Sentence Transformers not available. Returning empty DataFrame.")
        return pd.DataFrame(index=df.index)

    # Check if required column exists
    if 'CommentText' not in df.columns:
        print("Warning: CommentText column not found")
        return pd.DataFrame(index=df.index)

    # Load model with GPU if available
    try:
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        model = SentenceTransformer(model_name, device=device)
        print(f"Loaded embedding model: {model_name} on {device}")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return pd.DataFrame(index=df.index)

    # Fill NaN values with empty strings
    texts = df['CommentText'].fillna('').tolist()

    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} comments using batch size {batch_size}...")
    start_time = time.time()
    try:
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return pd.DataFrame(index=df.index)

    # Create DataFrame with proper column names
    embedding_df = pd.DataFrame(
        embeddings,
        index=df.index,
        columns=[f'emb_{i}' for i in range(embeddings.shape[1])]
    )

    elapsed_time = time.time() - start_time
    print(f"Embedding extraction completed in {elapsed_time:.2f} seconds")
    print(f"Created {embedding_df.shape[1]} embedding features")

    return embedding_df

def reduce_embedding_dimensions(embedding_features, method='pca', n_components=50):
    """Apply dimension reduction to embeddings."""
    print(f"Applying {method.upper()} to reduce embeddings from {embedding_features.shape[1]} to {n_components} dimensions")
    
    # First standardize
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embedding_features)
    
    if method.lower() == 'pca':
        # Apply PCA
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(scaled_embeddings)
        
        # Report variance explained
        explained_variance = sum(reducer.explained_variance_ratio_)
        print(f"PCA: {n_components} components explain {explained_variance:.2%} of variance")
        
        # Create DataFrame
        reduced_df = pd.DataFrame(
            reduced_data,
            index=embedding_features.index,
            columns=[f'pca_emb_{i}' for i in range(n_components)]
        )
        
    elif method.lower() == 'umap' and HAVE_UMAP:
        # Apply UMAP with GPU if available
        if HAVE_CUML:
            from cuml.manifold import UMAP as cuUMAP
            reducer = cuUMAP(n_components=n_components, random_state=42)
        else:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            
        reduced_data = reducer.fit_transform(scaled_embeddings)
        
        # Create DataFrame
        reduced_df = pd.DataFrame(
            reduced_data,
            index=embedding_features.index,
            columns=[f'umap_emb_{i}' for i in range(n_components)]
        )
        
    else:
        print(f"Method {method} not available. Using PCA.")
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(scaled_embeddings)
        
        reduced_df = pd.DataFrame(
            reduced_data,
            index=embedding_features.index,
            columns=[f'pca_emb_{i}' for i in range(n_components)]
        )
    
    print(f"Reduced embeddings to {reduced_df.shape[1]} dimensions")
    return reduced_df, reducer

def create_propaganda_features(df):
    """Create features from the propaganda techniques."""
    print("Processing propaganda techniques...")
    features = pd.DataFrame(index=df.index)

    # Initialize with empty lists
    techniques_list = pd.Series([[] for _ in range(len(df))], index=df.index)

    # Check if Techniques column exists and has data
    if 'Techniques' in df.columns:
        # Process techniques with progress bar
        progress = tqdm(total=len(df), desc="Processing Techniques")
        
        for idx, row in df.iterrows():
            tech = row.get('Techniques')

            # Skip None or NaN values
            if tech is None or (isinstance(tech, float) and pd.isna(tech)):
                progress.update(1)
                continue

            # Process different formats
            if isinstance(tech, list):
                techniques_list.loc[idx] = tech
            elif isinstance(tech, str):
                if tech.startswith('['):
                    try:
                        techniques_list.loc[idx] = eval(tech)
                    except:
                        techniques_list.loc[idx] = [tech] if tech else []
                elif tech and ',' in tech:
                    techniques_list.loc[idx] = [t.strip() for t in tech.split(',')]
                elif tech:
                    techniques_list.loc[idx] = [tech]
            
            progress.update(1)
        
        progress.close()
    else:
        print("Warning: 'Techniques' column not found in dataframe")

    # Count of propaganda techniques
    features['technique_count'] = techniques_list.apply(len)

    # Has propaganda techniques or not (binary)
    features['has_techniques'] = features['technique_count'] > 0

    # Create one-hot encoding for specific techniques
    all_techniques = set()
    for techniques in techniques_list:
        if isinstance(techniques, list):
            for technique in techniques:
                if technique and isinstance(technique, str):
                    # Clean up technique names to be valid column names
                    all_techniques.add(technique.replace(',', '').replace(' ', '_'))

    print(f"Found {len(all_techniques)} unique propaganda techniques")

    # Create one-hot encoding features, but only for techniques that appear multiple times
    progress = tqdm(total=len(all_techniques), desc="Creating Technique Features")
    
    for technique in all_techniques:
        if technique:  # Skip empty techniques
            # Create standardized column name
            technique_name = f"tech_{technique}"

            # Count occurrences of this technique
            features[technique_name] = 0  # Initialize with zeros

            # Set to 1 where technique exists
            for idx, techs in techniques_list.items():
                cleaned_techs = [t.replace(',', '').replace(' ', '_')
                                for t in techs if isinstance(t, str)]
                if technique in cleaned_techs:
                    features.loc[idx, technique_name] = 1
            
            progress.update(1)
    
    progress.close()
    print(f"Created {len(features.columns)} propaganda features")
    return features

def create_stance_features(df):
    """Create features based on stance information."""
    print("Creating stance features...")
    features = pd.DataFrame(index=df.index)

    # Extract stances safely with fallbacks for missing values
    try:
        # Check if the columns exist
        if 'parent_comment_stance' in df.columns:
            features['parent_stance'] = df['parent_comment_stance'].fillna(0)
        else:
            print("Warning: 'parent_comment_stance' not found, using zeros")
            features['parent_stance'] = 0

        if 'Stance_Label' in df.columns:
            features['comment_stance'] = df['Stance_Label'].fillna(0)
        else:
            print("Warning: 'Stance_Label' not found, using zeros")
            features['comment_stance'] = 0

        # Stance relationship features
        features['stance_diff'] = features['comment_stance'] - features['parent_stance']
        features['abs_stance_diff'] = abs(features['stance_diff'])
        features['same_stance'] = features['stance_diff'] == 0

        # Agreement vs. disagreement (based on stance)
        features['is_agreement'] = (
                ((features['parent_stance'] > 0) & (features['comment_stance'] > 0)) |
                ((features['parent_stance'] < 0) & (features['comment_stance'] < 0)) |
                ((features['parent_stance'] == 0) & (features['comment_stance'] == 0))
        )
    except Exception as e:
        print(f"Error creating stance features: {e}")
        # Provide basic fallback features if something goes wrong
        features['parent_stance'] = 0
        features['comment_stance'] = 0
        features['stance_diff'] = 0
        features['abs_stance_diff'] = 0
        features['same_stance'] = True
        features['is_agreement'] = True

    print(f"Created {len(features.columns)} stance features")
    return features

def create_feature_combinations(text_features, stance_features, propaganda_features, embedding_features=None, reduced_embeddings=None):
    """Create different feature combinations for testing."""
    combinations = {
        "Base(Text)": text_features,
        "Base+Stance": pd.concat([text_features, stance_features], axis=1),
        "Base+Tech": pd.concat([text_features, propaganda_features], axis=1),
        "Base+Tech+Stance": pd.concat([text_features, stance_features, propaganda_features], axis=1)
    }

    # Add embedding combinations if provided
    if embedding_features is not None and not embedding_features.empty:
        combinations["Base+Embedding"] = pd.concat([text_features, embedding_features], axis=1)
        combinations["Base+Tech+Stance+Embedding"] = pd.concat([
            text_features, stance_features, propaganda_features, embedding_features
        ], axis=1)
    
    # Add reduced embedding combinations if provided
    if reduced_embeddings is not None and not reduced_embeddings.empty:
        combinations["Base+ReducedEmbedding"] = pd.concat([text_features, reduced_embeddings], axis=1)
        combinations["Base+Tech+Stance+ReducedEmbedding"] = pd.concat([
            text_features, stance_features, propaganda_features, reduced_embeddings
        ], axis=1)

    for name, features in combinations.items():
        print(f"Created {name} with {features.shape[1]} features")
        
    return combinations

def plot_learning_curve(estimator, X, y, cv=5, title='Learning Curve', filename='learning_curve.png'):
    """Plot learning curve to visualize potential overfitting."""
    print(f"Generating learning curve...")

    train_sizes = np.linspace(0.1, 1.0, 5)  # Reduced number of points for faster computation

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, scoring='f1_macro',
            train_sizes=train_sizes, n_jobs=-1,
        )

        # Calculate mean and std for train and test scores
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"Learning curve saved to {filename}")
    except Exception as e:
        print(f"Error generating learning curve: {e}")

def train_model_with_tuning(model_name, base_model, features, target, class_values, param_grid, k_neighbors_values, cv=3, output_dir=None):
    """Train a model with hyperparameter tuning and test different SMOTE k_neighbors values."""
    print(f"\nTraining and tuning {model_name}...")
    
    # Get unique class values and create mapping
    class_to_index = {val: i for i, val in enumerate(class_values)}
    index_to_class = {i: val for val, i in class_to_index.items()}

    # Map target to integers for model compatibility
    target_indices = target.map(class_to_index)
    
    best_score = -1
    best_model = None
    best_params = None
    best_k_neighbors = None
    
    # Calculate total iterations for progress tracking
    total_iterations = len(k_neighbors_values)
    progress = ProgressEstimator(total_iterations, f"Tuning {model_name}")
    
    # Test each k_neighbors value
    for k_idx, k_neighbors in enumerate(k_neighbors_values):
        print(f"Testing SMOTE with k_neighbors={k_neighbors}")
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(k_neighbors=k_neighbors, random_state=42)),
            ('classifier', base_model)
        ])
        
        # Set up grid search
        with parallel_backend('loky', n_jobs=-1):  
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring='f1_macro',
                verbose=1
            )
            
            # Fit grid search
            print(f"Starting grid search for k_neighbors={k_neighbors}...")
            start_time = time.time()
            grid_search.fit(features, target_indices)
            elapsed_time = time.time() - start_time
            print(f"Grid search completed in {elapsed_time:.2f} seconds")
        
        # Check if this k_neighbors value gave better results
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_k_neighbors = k_neighbors
            
            # If output_dir is provided, save the current best model
            if output_dir:
                model_dir = os.path.join(output_dir, "intermediate_models")
                os.makedirs(model_dir, exist_ok=True)
                
                model_path = os.path.join(model_dir, f"{model_name}_k{k_neighbors}_best.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(grid_search.best_estimator_, f)
                print(f"Saved current best model to {model_path}")
        
        progress.update(1)
    
    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best k_neighbors: {best_k_neighbors}")
    print(f"Best CV F1: {best_score:.4f}")
    
    # Compute class-specific metrics on cross-validation
    class_metrics = {}
    
    # Use the best model for final evaluation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    class_f1_scores = {label: [] for label in class_values}
    
    print("Evaluating best model with cross-validation...")
    cv_progress = tqdm(total=cv, desc=f"{model_name} CV Evaluation")
    
    for train_idx, val_idx in cv_splitter.split(features, target_indices):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = target_indices.iloc[train_idx], target_indices.iloc[val_idx]
        
        # Clone the best model to avoid fitting issues
        from sklearn.base import clone
        model_clone = clone(best_model)
        model_clone.fit(X_train, y_train)
        
        y_pred = model_clone.predict(X_val)
        
        # Calculate metrics for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, y_pred, labels=range(len(class_values)), average=None, zero_division=0
        )
        
        # Store F1 scores for each class
        for i, class_val in enumerate(class_values):
            class_f1_scores[class_val].append(f1[i])
            
        cv_progress.update(1)
    
    cv_progress.close()
    
    # Average metrics across folds
    for class_val in class_values:
        class_metrics[class_val] = np.mean(class_f1_scores[class_val])
    
    # Train the best model on the full dataset
    print("Training final model on full dataset...")
    best_model.fit(features, target_indices)
    
    # Generate learning curve for best model
    if output_dir:
        curve_dir = os.path.join(output_dir, "learning_curves")
        os.makedirs(curve_dir, exist_ok=True)
        plot_learning_curve(
            best_model, features, target_indices, 
            cv=3, title=f"Learning Curve - {model_name}",
            filename=os.path.join(curve_dir, f"learning_curve_{model_name.lower().replace(' ', '_')}.png")
        )
    
    return best_model, best_score, class_metrics, best_k_neighbors, best_params

def evaluate_comment_predictions(model, features, true_labels, class_values, index_to_class, output_dir=None):
    """Evaluate model on individual comment divisiveness prediction."""
    print("\nEvaluating individual comment predictions...")

    # Map true labels to indices
    class_to_index = {val: i for i, val in enumerate(class_values)}
    true_indices = true_labels.map(class_to_index)

    # Make predictions
    pred_indices = model.predict(features)

    # Map back to original values
    pred_values = pd.Series([index_to_class[idx] for idx in pred_indices], index=true_labels.index)

    # Calculate metrics
    f1 = f1_score(true_indices, pred_indices, average='macro')
    print(f"Macro F1 Score: {f1:.4f}")

    # Get class-specific F1 scores
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        true_indices, pred_indices, labels=range(len(class_values)), average=None, zero_division=0
    )
    
    # Create dictionary mapping class values to their F1 scores
    class_f1_dict = {class_values[i]: f1_per_class[i] for i in range(len(class_values))}

    # Classification report with meaningful class names
    target_names = [RELATION_MAP[val] for val in class_values]
    class_report = classification_report(true_indices, pred_indices, target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(class_report)

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_indices, pred_indices)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_dir:
        # Save confusion matrix
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Save classification report
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(class_report)
    else:
        plt.savefig('confusion_matrix.png')
    
    plt.close()

    return pred_values, f1, class_f1_dict

def evaluate_chain_predictions(pred_values, test_data, test_df, class_values, output_dir=None):
    """Ultra-optimized chain-level predictions using direct dictionary lookups."""
    print("\n=== Experiment 2: Chain-Level Category Prediction ===")
    print("\nEvaluating chain-level predictions...")
    start_time = time.time()
    
    # Convert Series to fast lookup dictionaries
    comment_ids = test_df['CommentID'].values
    true_values = test_df['divisiveness_score'].values
    pred_values_array = pred_values.values
    
    # Create fast lookup dictionaries
    true_dict = {}
    pred_dict = {}
    
    # Handle duplicates during dictionary creation
    seen_ids = set()
    for i, comment_id in enumerate(comment_ids):
        if comment_id not in seen_ids:
            true_dict[comment_id] = float(true_values[i])
            pred_dict[comment_id] = float(pred_values_array[i])
            seen_ids.add(comment_id)
    
    if len(seen_ids) < len(comment_ids):
        print(f"Warning: Found {len(comment_ids) - len(seen_ids)} duplicate comment IDs. Using first occurrence only.")
    
    # Pre-process chain data for faster access
    chain_comments = {}
    for chain_id, chain_data in test_data['discussions'].items():
        valid_comments = []
        for msg in chain_data['messages']:
            if msg.get('Level', 0) > 0 and 'divisiveness_score' in msg:
                comment_id = msg['CommentID']
                if comment_id in true_dict and comment_id in pred_dict:
                    valid_comments.append(comment_id)
        if valid_comments:
            chain_comments[chain_id] = valid_comments
    
    print(f"Processing {len(chain_comments)} chains with {sum(len(c) for c in chain_comments.values())} comments...")
    
    # Define function to get category (optimize with direct comparison)
    def get_category(score):
        if score <= -0.75:
            return "Highly Destructive"
        elif score <= -0.25:
            return "Moderately Destructive"
        elif score <= 0.25:
            return "Slightly Destructive-Constructive_Neutral"
        else:
            return "Constructive"
    
    # Batch process all chains
    chain_results = {}
    categories = list(CHAIN_CATEGORIES.keys())
    
    # Initialize confusion matrix and metrics
    confusion_matrix = np.zeros((len(categories), len(categories)), dtype=int)
    category_metrics = {cat: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for cat in categories}
    
    # Process all chains (no tqdm to maximize speed)
    for chain_id, comment_ids in chain_comments.items():
        # Direct dictionary lookups for maximum speed
        true_scores = [true_dict[cid] for cid in comment_ids]
        pred_scores = [pred_dict[cid] for cid in comment_ids]
        
        # Simple averaging
        if true_scores and pred_scores:
            true_avg = sum(true_scores) / len(true_scores)
            pred_avg = sum(pred_scores) / len(pred_scores)
            
            # Get categories
            true_category = get_category(true_avg)
            pred_category = get_category(pred_avg)
            
            # Store results
            chain_results[chain_id] = {
                'true_avg': true_avg,
                'pred_avg': pred_avg,
                'true_category': true_category,
                'pred_category': pred_category,
                'correct': true_category == pred_category,
                'num_comments': len(true_scores)
            }
            
            # Update confusion matrix
            true_idx = categories.index(true_category)
            pred_idx = categories.index(pred_category)
            confusion_matrix[true_idx, pred_idx] += 1
            
            # Update metrics directly (faster than later calculation)
            for category in categories:
                is_true_positive = (true_category == category and pred_category == category)
                is_false_positive = (true_category != category and pred_category == category)
                is_false_negative = (true_category == category and pred_category != category)
                is_true_negative = (true_category != category and pred_category != category)
                
                if is_true_positive:
                    category_metrics[category]['tp'] += 1
                elif is_false_positive:
                    category_metrics[category]['fp'] += 1
                elif is_false_negative:
                    category_metrics[category]['fn'] += 1
                elif is_true_negative:
                    category_metrics[category]['tn'] += 1
    
    # Calculate accuracy
    accuracy = sum(1 for r in chain_results.values() if r['correct']) / len(chain_results) if chain_results else 0
    print(f"Chain Category Prediction Accuracy: {accuracy:.4f}")
    
    # Calculate metrics for reporting
    chain_f1_scores = {}
    chain_report = f"{'Category':<30} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n"
    chain_report += "-" * 70 + "\n"
    
    for i, category in enumerate(categories):
        # Get precalculated metrics
        tp = category_metrics[category]['tp']
        fp = category_metrics[category]['fp']
        fn = category_metrics[category]['fn']
        support = sum(confusion_matrix[i, :])
        
        # Calculate precision, recall, F1 with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        chain_f1_scores[category] = f1
        chain_report += f"{category:<30} {precision:.4f}    {recall:.4f}    {f1:.4f}    {support}\n"
    
    chain_report += "-" * 70 + "\n"
    
    # Calculate macro averages
    macro_precision = sum(tp / (tp + fp) if (tp + fp) > 0 else 0 
                        for category, metrics in category_metrics.items() 
                        for tp, fp in [(metrics['tp'], metrics['fp'])]) / len(categories)
    
    macro_recall = sum(tp / (tp + fn) if (tp + fn) > 0 else 0 
                     for category, metrics in category_metrics.items() 
                     for tp, fn in [(metrics['tp'], metrics['fn'])]) / len(categories)
    
    macro_f1 = sum(chain_f1_scores.values()) / len(categories)
    total_support = sum(sum(confusion_matrix[i, :]) for i in range(len(categories)))
    
    chain_report += f"{'Macro Average':<30} {macro_precision:.4f}    {macro_recall:.4f}    {macro_f1:.4f}    {total_support}\n"
    
    # Print report
    print("\nChain Category Metrics:")
    print(chain_report)
    
    # Create visualization and save results if output_dir is specified
    if output_dir:
        # Save results to CSV
        chain_results_df = pd.DataFrame([
            {
                'chain_id': chain_id,
                'true_avg': data['true_avg'],
                'pred_avg': data['pred_avg'],
                'true_category': data['true_category'],
                'pred_category': data['pred_category'],
                'correct': data['correct'],
                'num_comments': data['num_comments']
            }
            for chain_id, data in chain_results.items()
        ])
        chain_results_df.to_csv(os.path.join(output_dir, 'chain_prediction_results.csv'), index=False)
        
        # Save evaluation report
        with open(os.path.join(output_dir, 'chain_evaluation_report.txt'), 'w') as f:
            f.write(chain_report)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories, yticklabels=categories)
        plt.title('Chain Category Confusion Matrix')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'chain_category_confusion_matrix.png'))
        plt.close()
        
        # Plot chain accuracy by size
        plot_chain_accuracy_by_size(chain_results, output_dir)
    
    elapsed = time.time() - start_time
    print(f"Chain evaluation completed in {elapsed:.2f} seconds")
    
    return {
        'macro_f1': macro_f1,
        'category_f1': chain_f1_scores,
        'accuracy': accuracy,
        'chain_results': chain_results
    }

def plot_chain_accuracy_by_size(chain_results, output_dir=None):
    """Plot accuracy by chain size."""
    # Group chains by size
    size_groups = defaultdict(list)

    for chain_id, data in chain_results.items():
        size = data['num_comments']
        # Group sizes to avoid too many categories
        group = 1 if size == 1 else 2 if size == 2 else 3 if size == 3 else 4 if size <= 5 else 6 if size <= 10 else 11
        size_groups[group].append(data['correct'])

    # Calculate accuracy for each size group
    size_accuracy = {}
    for size, results in size_groups.items():
        accuracy = sum(results) / len(results)
        size_accuracy[size] = {
            'accuracy': accuracy,
            'count': len(results)
        }

    # Sort by size
    sorted_sizes = sorted(size_accuracy.keys())

    # Plot
    plt.figure(figsize=(10, 6))
    accuracies = [size_accuracy[s]['accuracy'] for s in sorted_sizes]
    counts = [size_accuracy[s]['count'] for s in sorted_sizes]

    # Create x-axis labels
    x_labels = ['1 comment' if s == 1 else f'{s} comments' if s <= 5
    else '6-10 comments' if s == 6 else '11+ comments'
                for s in sorted_sizes]

    # Plot bars
    plt.bar(x_labels, accuracies, color='skyblue')

    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, accuracies[i] + 0.02, f'n={count}', ha='center')

    plt.ylim(0, 1.0)
    plt.title('Chain Category Prediction Accuracy by Chain Size')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Comments in Chain')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'chain_accuracy_by_size.png'))
    else:
        plt.savefig('chain_accuracy_by_size.png')
        
    plt.close()

def run_experiments(train_data, test_data, output_path, pca_components_range, k_neighbors_range):
    """Run all experiments and generate table results with parameter tuning."""
    print("======= Running Divisiveness Prediction Experiments with Parameter Tuning =======")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_path, f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Extract and process comments
    train_comments = extract_comments_from_chains(train_data)
    test_comments = extract_comments_from_chains(test_data)
    
    train_df = comments_to_dataframe(train_comments)
    test_df = comments_to_dataframe(test_comments)
    
    # Extract features
    print("\n=== Extracting Training Features ===")
    train_text = extract_text_features(train_df)
    train_propaganda = create_propaganda_features(train_df)
    train_stance = create_stance_features(train_df)
    train_embeddings = extract_text_embeddings(train_df) if HAVE_SENTENCE_TRANSFORMERS else None
    
    print("\n=== Extracting Testing Features ===")
    test_text = extract_text_features(test_df)
    test_propaganda = create_propaganda_features(test_df)
    test_stance = create_stance_features(test_df)
    test_embeddings = extract_text_embeddings(test_df) if HAVE_SENTENCE_TRANSFORMERS else None
    
    # Get target values
    train_target = train_df['divisiveness_score']
    test_target = test_df['divisiveness_score']
    
    # Unique class values
    class_values = sorted(train_target.unique())
    print(f"Class values: {class_values}")
    
    # Define hyperparameter search spaces
    param_grids = {
        "Random Forest": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        },
        "SVM": {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
            'classifier__kernel': ['rbf', 'linear']
        },
        "Log_Reg": {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__max_iter': [1000, 2000]
        }
    }
    
    if HAVE_XGB:
        if XGB_VERSION_2_PLUS:
            param_grids["XGBoost"] = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 6, 9]
            }
        else:
            param_grids["XGBoost"] = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 6, 9]
            }
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
        "Log_Reg": LogisticRegression(class_weight='balanced', random_state=42)
    }
    
    # Add XGBoost if available
    if HAVE_XGB:
        if XGB_VERSION_2_PLUS:
            xgb_params = {
                'objective': 'multi:softmax',
                'num_class': len(class_values),
                'random_state': 42
            }
            
            if XGB_GPU:
                xgb_params.update({
                    'tree_method': 'hist',
                    'device': 'cuda'
                })
        else:
            xgb_params = {
                'objective': 'multi:softmax',
                'num_class': len(class_values),
                'random_state': 42
            }
            
            if XGB_GPU:
                xgb_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0
                })
        
        models["XGBoost"] = xgb.XGBClassifier(**xgb_params)
    
    # Track best configurations
    best_configs = {}
    
    # Results storage
    all_results = {}
    
    # Create results tables
    comment_results = {
        'macro_f1': pd.DataFrame(index=["Base(Text)", "Base+Stance", "Base+Tech", "Base+Tech+Stance", 
                                       "Base+Embedding", "Base+Tech+Stance+Embedding"], 
                                columns=list(models.keys()))
    }
    
    # Create tables for each class
    for class_val in class_values:
        comment_results[SIMPLE_RELATION_MAP[class_val]] = pd.DataFrame(
            index=["Base(Text)", "Base+Stance", "Base+Tech", "Base+Tech+Stance", 
                  "Base+Embedding", "Base+Tech+Stance+Embedding"], 
            columns=list(models.keys())
        )
    
    # Chain results
    chain_results = {
        'macro_f1': pd.DataFrame(
            index=["Base(Text)", "Base+Stance", "Base+Tech", "Base+Tech+Stance", 
                  "Base+Embedding", "Base+Tech+Stance+Embedding"], 
            columns=list(models.keys())
        )
    }
    
    # Create tables for each chain category
    for category in CHAIN_CATEGORIES.keys():
        chain_results[category] = pd.DataFrame(
            index=["Base(Text)", "Base+Stance", "Base+Tech", "Base+Tech+Stance", 
                  "Base+Embedding", "Base+Tech+Stance+Embedding"], 
            columns=list(models.keys())
        )
    
    # Storage for parameter configurations
    param_configs = {}
    
    # Run experiments for base feature combinations (without embeddings)
    base_combinations = create_feature_combinations(train_text, train_stance, train_propaganda)
    test_base_combinations = create_feature_combinations(test_text, test_stance, test_propaganda)
    
    # Process each feature combination
    for feature_name, train_features in base_combinations.items():
        # Create output directory for this feature combination
        feature_dir = os.path.join(output_path, feature_name.replace('(', '').replace(')', '').replace('+', '_'))
        os.makedirs(feature_dir, exist_ok=True)
        
        # Ensure test features match training features
        test_features = test_base_combinations[feature_name]
        test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
        
        # Track best model and results for this feature combination
        feature_best_results = {
            'comment_f1': -1,
            'chain_f1': -1,
            'model_name': '',
            'model': None
        }
        
        # Train and evaluate each model
        for model_name, model in models.items():
            # Create output directory for this model
            model_dir = os.path.join(feature_dir, model_name.replace(' ', '_'))
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"\n=== Training and evaluating: {model_name} on {feature_name} ===")
            
            # Train with hyperparameter tuning
            best_model, cv_score, class_metrics, best_k, best_params = train_model_with_tuning(
                model_name, model, train_features, train_target, class_values, 
                param_grids[model_name], k_neighbors_range, output_dir=model_dir
            )
            
            # Store best parameters
            param_configs[f"{feature_name}_{model_name}"] = {
                'model_params': best_params,
                'k_neighbors': best_k
            }

            # Evaluate on test set for comment predictions
            test_comment_dir = os.path.join(model_dir, "comment_evaluation")
            os.makedirs(test_comment_dir, exist_ok=True)
            
            # Get prediction values for comments (to be reused in chain evaluation)
            pred_values, test_f1, test_class_f1 = evaluate_comment_predictions(
                best_model, test_features, test_target, class_values, 
                {i: val for i, val in enumerate(class_values)},
                output_dir=test_comment_dir
            )
            
            # Store metrics for each class in the result tables
            comment_results['macro_f1'].loc[feature_name, model_name] = test_f1
            
            for class_val, f1_score in test_class_f1.items():
                class_name = SIMPLE_RELATION_MAP[class_val]
                comment_results[class_name].loc[feature_name, model_name] = f1_score
            
            # Evaluate on chain level using the same prediction values
            test_chain_dir = os.path.join(model_dir, "chain_evaluation")
            os.makedirs(test_chain_dir, exist_ok=True)
            
            # Use predictions directly without recomputing them
            chain_metrics = evaluate_chain_predictions(
                pred_values, test_data, test_df, class_values, 
                output_dir=test_chain_dir
            )
            
            # Store chain metrics
            chain_results['macro_f1'].loc[feature_name, model_name] = chain_metrics['macro_f1']
            
            for category, f1_score in chain_metrics['category_f1'].items():
                chain_results[category].loc[feature_name, model_name] = f1_score
            
            # Save model
            model_path = os.path.join(model_dir, f"{feature_name}_{model_name}_best_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Check if this is the best model for this feature combination
            if cv_score > feature_best_results['comment_f1']:
                feature_best_results['comment_f1'] = cv_score
                feature_best_results['chain_f1'] = chain_metrics['macro_f1']
                feature_best_results['model_name'] = model_name
                feature_best_results['model'] = best_model
        
        # Save the best model for this feature combination
        best_configs[feature_name] = feature_best_results
    
    # Now process embedding-based feature combinations with dimension reduction
    if train_embeddings is not None and not train_embeddings.empty:
        embedding_dir = os.path.join(output_path, "embedding_experiments")
        os.makedirs(embedding_dir, exist_ok=True)
        
        for n_components in pca_components_range:
            pca_dir = os.path.join(embedding_dir, f"pca_{n_components}")
            os.makedirs(pca_dir, exist_ok=True)
            
            print(f"\n=== Testing PCA with {n_components} components ===")
            
            # Apply dimension reduction to embeddings
            train_reduced_embeddings, reducer = reduce_embedding_dimensions(
                train_embeddings, method='pca', n_components=n_components
            )
            
            # Apply same transformation to test embeddings
            test_reduced_embeddings = pd.DataFrame(
                reducer.transform(StandardScaler().fit_transform(test_embeddings)),
                index=test_embeddings.index,
                columns=[f'pca_emb_{i}' for i in range(n_components)]
            )
            
            # Create feature combinations with reduced embeddings
            train_combinations = create_feature_combinations(
                train_text, train_stance, train_propaganda, None, train_reduced_embeddings
            )
            
            test_combinations = create_feature_combinations(
                test_text, test_stance, test_propaganda, None, test_reduced_embeddings
            )
            
            # Process only embedding-based combinations with reduced embeddings
            for feature_name in ["Base+ReducedEmbedding", "Base+Tech+Stance+ReducedEmbedding"]:
                if feature_name in train_combinations:
                    # Create directory for this feature combination
                    feature_dir = os.path.join(pca_dir, feature_name.replace('+', '_'))
                    os.makedirs(feature_dir, exist_ok=True)
                    
                    train_features = train_combinations[feature_name]
                    test_features = test_combinations[feature_name]
                    
                    # Ensure test features match training features
                    test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
                    
                    # Use a modified name that includes component count
                    display_name = f"{feature_name.replace('Reduced', '')} (PCA-{n_components})"
                    
                    # Add this index to result DataFrames if not already present
                    if display_name not in comment_results['macro_f1'].index:
                        for result_table in comment_results.values():
                            result_table.loc[display_name] = np.nan
                        
                        for result_table in chain_results.values():
                            result_table.loc[display_name] = np.nan
                    
                    # Track best model and results for this feature combination
                    feature_best_results = {
                        'comment_f1': -1,
                        'chain_f1': -1,
                        'model_name': '',
                        'model': None
                    }
                    
                    # Train and evaluate each model
                    for model_name, model in models.items():
                        # Create output directory for this model
                        model_dir = os.path.join(feature_dir, model_name.replace(' ', '_'))
                        os.makedirs(model_dir, exist_ok=True)
                        
                        print(f"\n=== Training and evaluating: {model_name} on {display_name} ===")
                        
                        # Train with hyperparameter tuning
                        best_model, cv_score, class_metrics, best_k, best_params = train_model_with_tuning(
                            model_name, model, train_features, train_target, class_values, 
                            param_grids[model_name], k_neighbors_range, output_dir=model_dir
                        )
                        
                        # Store best parameters
                        param_configs[f"{display_name}_{model_name}"] = {
                            'model_params': best_params,
                            'k_neighbors': best_k,
                            'pca_components': n_components
                        }
                        
                        # Evaluate on test set for comment predictions
                        test_comment_dir = os.path.join(model_dir, "comment_evaluation")
                        os.makedirs(test_comment_dir, exist_ok=True)
                        
                        # Get prediction values for comments (to be reused in chain evaluation)
                        pred_values, test_f1, test_class_f1 = evaluate_comment_predictions(
                            best_model, test_features, test_target, class_values, 
                            {i: val for i, val in enumerate(class_values)},
                            output_dir=test_comment_dir
                        )
                        
                        # Store metrics for each class in the result tables
                        comment_results['macro_f1'].loc[display_name, model_name] = test_f1
                        
                        for class_val, f1_score in test_class_f1.items():
                            class_name = SIMPLE_RELATION_MAP[class_val]
                            comment_results[class_name].loc[display_name, model_name] = f1_score
                        
                        # Evaluate on chain level using the same prediction values
                        test_chain_dir = os.path.join(model_dir, "chain_evaluation")
                        os.makedirs(test_chain_dir, exist_ok=True)
                        
                        # Use predictions directly without recomputing them
                        chain_metrics = evaluate_chain_predictions(
                            pred_values, test_data, test_df, class_values, 
                            output_dir=test_chain_dir
                        )
                        
                        # Store chain metrics
                        chain_results['macro_f1'].loc[display_name, model_name] = chain_metrics['macro_f1']
                        
                        for category, f1_score in chain_metrics['category_f1'].items():
                            chain_results[category].loc[display_name, model_name] = f1_score
                        
                        # Save model
                        model_path = os.path.join(model_dir, f"{display_name}_{model_name}_best_model.pkl")
                        with open(model_path, 'wb') as f:
                            pickle.dump(best_model, f)
                        
                        # Check if this is the best model for this feature combination
                        if cv_score > feature_best_results['comment_f1']:
                            feature_best_results['comment_f1'] = cv_score
                            feature_best_results['chain_f1'] = chain_metrics['macro_f1']
                            feature_best_results['model_name'] = model_name
                            feature_best_results['model'] = best_model
                    
                    # Save the best model for this feature combination
                    best_configs[display_name] = feature_best_results
    
    # Test regular embeddings without dimension reduction
    if train_embeddings is not None and not train_embeddings.empty:
        full_embedding_dir = os.path.join(output_path, "full_embedding_experiments")
        os.makedirs(full_embedding_dir, exist_ok=True)
        
        # Create feature combinations with full embeddings
        train_combinations = create_feature_combinations(
            train_text, train_stance, train_propaganda, train_embeddings
        )
        
        test_combinations = create_feature_combinations(
            test_text, test_stance, test_propaganda, test_embeddings
        )
        
        # Process only embedding-based combinations
        for feature_name in ["Base+Embedding", "Base+Tech+Stance+Embedding"]:
            # Create directory for this feature combination
            feature_dir = os.path.join(full_embedding_dir, feature_name.replace('+', '_'))
            os.makedirs(feature_dir, exist_ok=True)
            
            train_features = train_combinations[feature_name]
            test_features = test_combinations[feature_name]
            
            # Ensure test features match training features
            test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
            
            # Track best model and results for this feature combination
            feature_best_results = {
                'comment_f1': -1,
                'chain_f1': -1,
                'model_name': '',
                'model': None
            }
            
            # Train and evaluate each model
            for model_name, model in models.items():
                # Create output directory for this model
                model_dir = os.path.join(feature_dir, model_name.replace(' ', '_'))
                os.makedirs(model_dir, exist_ok=True)
                
                print(f"\n=== Training and evaluating: {model_name} on {feature_name} ===")
                
                # Train with hyperparameter tuning
                best_model, cv_score, class_metrics, best_k, best_params = train_model_with_tuning(
                    model_name, model, train_features, train_target, class_values, 
                    param_grids[model_name], k_neighbors_range, output_dir=model_dir
                )
                
                # Store best parameters
                param_configs[f"{feature_name}_{model_name}"] = {
                    'model_params': best_params,
                    'k_neighbors': best_k
                }
                
                # Evaluate on test set for comment predictions
                test_comment_dir = os.path.join(model_dir, "comment_evaluation")
                os.makedirs(test_comment_dir, exist_ok=True)
                
                # Get prediction values for comments (to be reused in chain evaluation)
                pred_values, test_f1, test_class_f1 = evaluate_comment_predictions(
                    best_model, test_features, test_target, class_values, 
                    {i: val for i, val in enumerate(class_values)},
                    output_dir=test_comment_dir
                )
                
                # Store metrics for each class in the result tables
                comment_results['macro_f1'].loc[feature_name, model_name] = test_f1
                
                for class_val, f1_score in test_class_f1.items():
                    class_name = SIMPLE_RELATION_MAP[class_val]
                    comment_results[class_name].loc[feature_name, model_name] = f1_score
                
                # Evaluate on chain level using the same prediction values
                test_chain_dir = os.path.join(model_dir, "chain_evaluation")
                os.makedirs(test_chain_dir, exist_ok=True)
                
                # Use predictions directly without recomputing them
                chain_metrics = evaluate_chain_predictions(
                    pred_values, test_data, test_df, class_values, 
                    output_dir=test_chain_dir
                )
                
                # Store chain metrics
                chain_results['macro_f1'].loc[feature_name, model_name] = chain_metrics['macro_f1']
                
                for category, f1_score in chain_metrics['category_f1'].items():
                    chain_results[category].loc[feature_name, model_name] = f1_score
                
                # Save model
                model_path = os.path.join(model_dir, f"{feature_name}_{model_name}_best_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                
                # Check if this is the best model for this feature combination
                if cv_score > feature_best_results['comment_f1']:
                    feature_best_results['comment_f1'] = cv_score
                    feature_best_results['chain_f1'] = chain_metrics['macro_f1']
                    feature_best_results['model_name'] = model_name
                    feature_best_results['model'] = best_model
            
            # Save the best model for this feature combination
            best_configs[feature_name] = feature_best_results
    
    # Clean up any NaN values in the results
    for result_table in comment_results.values():
        result_table.fillna(0, inplace=True)
    
    for result_table in chain_results.values():
        result_table.fillna(0, inplace=True)
    
    # Save results to tables directory
    tables_dir = os.path.join(output_path, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # Save comment-level results
    for metric_name, table in comment_results.items():
        table.to_csv(f"{tables_dir}/comment_{metric_name}.csv")
        print(f"Saved {metric_name} table to {tables_dir}/comment_{metric_name}.csv")
    
    # Save chain-level results
    for metric_name, table in chain_results.items():
        table.to_csv(f"{tables_dir}/chain_{metric_name}.csv")
        print(f"Saved {metric_name} table to {tables_dir}/chain_{metric_name}.csv")
    
    # Create a consolidated Excel file with all tables
    with pd.ExcelWriter(f"{tables_dir}/all_results.xlsx") as writer:
        # Write comment-level sheets
        for metric_name, table in comment_results.items():
            table.to_excel(writer, sheet_name=f"comment_{metric_name}")
        
        # Write chain-level sheets
        for metric_name, table in chain_results.items():
            table.to_excel(writer, sheet_name=f"chain_{metric_name}")
        
        # Write parameter configurations
        pd.DataFrame.from_dict(param_configs, orient='index').to_excel(writer, sheet_name="parameters")
    
    print(f"Saved all results to {tables_dir}/all_results.xlsx")
    
    # Save best configurations and models
    with open(f"{output_path}/best_configs.json", 'w') as f:
        # Convert to serializable format
        serializable_configs = {}
        for feature_name, config in best_configs.items():
            serializable_configs[feature_name] = {
                'comment_f1': float(config['comment_f1']),
                'chain_f1': float(config['chain_f1']),
                'model_name': config['model_name']
            }
        json.dump(serializable_configs, f, indent=2)
    
    print(f"Saved best configurations to {output_path}/best_configs.json")
    
    # Find overall best model
    best_overall_f1 = -1
    best_overall_config = None
    best_overall_feature = None
    
    for feature_name, config in best_configs.items():
        if config['comment_f1'] > best_overall_f1:
            best_overall_f1 = config['comment_f1']
            best_overall_config = config
            best_overall_feature = feature_name
    
    print("\n=== Best Overall Model ===")
    print(f"Feature Set: {best_overall_feature}")
    print(f"Model: {best_overall_config['model_name']}")
    print(f"Comment F1: {best_overall_config['comment_f1']:.4f}")
    print(f"Chain F1: {best_overall_config['chain_f1']:.4f}")
    
    # Return all results for further analysis
    return {
        'comment_results': comment_results,
        'chain_results': chain_results,
        'best_configs': best_configs,
        'param_configs': param_configs,
        'best_overall': {
            'feature': best_overall_feature,
            'model': best_overall_config['model_name'],
            'comment_f1': best_overall_config['comment_f1'],
            'chain_f1': best_overall_config['chain_f1']
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run divisiveness prediction experiments with parameter tuning")
    parser.add_argument('--train_path', default="Add Path", help="Path to training dataset")
    parser.add_argument('--test_path', default="Add Path", help="Path to test dataset")
    parser.add_argument('--output_path', default="Add Path", help="Directory to save results")
    parser.add_argument('--pca_components', default="20,25,30,35,40,45,50", help="PCA components to test (comma-separated)")
    parser.add_argument('--k_neighbors', default="5,8,10,12", help="SMOTE k_neighbors values to test (comma-separated)")
    
    args = parser.parse_args()
    
    # Parse PCA components and k_neighbors ranges
    pca_components_range = [int(x) for x in args.pca_components.split(',')]
    k_neighbors_range = [int(x) for x in args.k_neighbors.split(',')]
    
    print(f"Testing PCA components: {pca_components_range}")
    print(f"Testing SMOTE k_neighbors: {k_neighbors_range}")
    
    # Load data
    train_data = load_json_data(args.train_path)
    test_data = load_json_data(args.test_path)
    
    if not train_data or not test_data:
        print("Failed to load data. Exiting.")
        exit(1)
    
    # Run experiments
    run_experiments(train_data, test_data, args.output_path, pca_components_range, k_neighbors_range)
