import json
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import time
from tqdm import tqdm
import re

# Set the path to your model
MODEL_PATH = "Add Path"

# Set the path to your test data
TEST_DATA_PATH = "Add Path"
OUTPUT_DIR = "./ablation_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to load model
def load_model(model_path):
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to load test data
def load_json_data(file_path):
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Helper function to extract comments from chains
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

# Convert comments to DataFrame
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

# Function to extract text features (reusing from your script)
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
        return count_words_simple(text)

    def capitals_ratio(text):
        if not text or len(text) == 0:
            return 0
        capitals = sum(1 for c in text if c.isupper())
        return capitals / len(text)

    def count_punctuation(text, punct):
        return text.count(punct) / (len(text) + 1)

    # Extract features from child comments
    print("Processing child comments...")
    features['word_count'] = df['CommentText'].apply(count_words)
    features['capitals_ratio'] = df['CommentText'].apply(capitals_ratio)
    features['question_marks'] = df['CommentText'].apply(lambda x: count_punctuation(x, '?'))
    features['exclamation_marks'] = df['CommentText'].apply(lambda x: count_punctuation(x, '!'))

    # Try to use sentiment if available
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sentiment_analyzer = SentimentIntensityAnalyzer()
        print("Calculating sentiment scores...")
        sentiment_features = df['CommentText'].apply(lambda x: pd.Series(sentiment_analyzer.polarity_scores(x)))
        features['sentiment_neg'] = sentiment_features['neg']
        features['sentiment_pos'] = sentiment_features['pos']
        features['sentiment_neu'] = sentiment_features['neu']
        features['sentiment_compound'] = sentiment_features['compound']
    except ImportError:
        print("VADER not available, skipping sentiment features")
        # Create dummy sentiment features
        features['sentiment_neg'] = 0
        features['sentiment_pos'] = 0
        features['sentiment_neu'] = 0
        features['sentiment_compound'] = 0

    # Extract some parent text features for comparison
    print("Processing parent comments...")
    features['parent_word_count'] = df['ParentCommentText'].apply(count_words)
    features['parent_capitals_ratio'] = df['ParentCommentText'].apply(capitals_ratio)

    # Calculate some derived features
    features['word_count_diff'] = features['word_count'] - features['parent_word_count']
    features['word_count_ratio'] = features['word_count'] / (features['parent_word_count'] + 1)

    print(f"Created {len(features.columns)} text features")
    return features

# Function to extract propaganda features
def create_propaganda_features(df):
    """Create features from the propaganda techniques."""
    print("Processing propaganda techniques...")
    features = pd.DataFrame(index=df.index)

    # Initialize with empty lists
    techniques_list = pd.Series([[] for _ in range(len(df))], index=df.index)

    # Check if Techniques column exists and has data
    if 'Techniques' in df.columns:
        # Process techniques
        for idx, row in df.iterrows():
            tech = row.get('Techniques')

            # Skip None or NaN values
            if tech is None or (isinstance(tech, float) and pd.isna(tech)):
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

    # Create one-hot encoding features
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

    print(f"Created {len(features.columns)} propaganda features")
    return features

# Function to extract stance features
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

# Function to extract text embeddings and apply PCA (if model was trained with embeddings)
def extract_embeddings_with_pca(df, pca_components=40):
    """Extract text embeddings and apply PCA."""
    print(f"Extracting embeddings with PCA (components={pca_components})...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'roberta-base-nli-stsb-mean-tokens'  # Use the same model as in training
        
        # Load model
        model = SentenceTransformer(model_name, device=device)
        print(f"Loaded embedding model: {model_name} on {device}")
        
        # Generate embeddings
        texts = df['CommentText'].fillna('').tolist()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Apply standardization and PCA
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # Apply PCA
        pca = PCA(n_components=pca_components)
        reduced_data = pca.fit_transform(scaled_embeddings)
        
        # Create DataFrame with column names matching those expected by the model
        embedding_df = pd.DataFrame(
            reduced_data,
            index=df.index,
            columns=[f'pca_emb_{i}' for i in range(pca_components)]
        )
        
        print(f"Generated {pca_components} PCA-reduced embedding features")
        return embedding_df, pca
        
    except ImportError as e:
        print(f"Warning: Could not load embedding libraries: {e}")
        print("Creating empty embeddings DataFrame")
        
        # Create empty DataFrame with expected column names
        embedding_df = pd.DataFrame(
            0,  # Fill with zeros
            index=df.index,
            columns=[f'pca_emb_{i}' for i in range(pca_components)]
        )
        
        return embedding_df, None

# Function to analyze built-in feature importance from XGBoost
def analyze_feature_importance(model, feature_names):
    """Extract and visualize feature importance from the XGBoost model."""
    print("Analyzing built-in XGBoost feature importance...")
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
        # If it's a pipeline, get feature importances from the classifier
        importances = model.steps[-1][1].feature_importances_
    else:
        print("Could not extract feature importances directly from model")
        return None
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 30 features
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(30))
    plt.title('Top 30 Features by Importance')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance_top30.png")
    
    # Save full importance table
    importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    
    # Group features by category and calculate aggregate importance
    feature_groups = {}
    
    # Define patterns to identify feature groups
    patterns = {
        'Text': r'^(word_count|capitals_ratio|question_marks|exclamation_marks|sentiment|parent_word)',
        'Propaganda': r'^(technique_count|has_techniques|tech_)',
        'Stance': r'^(parent_stance|comment_stance|stance_diff|abs_stance_diff|same_stance|is_agreement)',
        'Embeddings': r'^(pca_emb_)'
    }
    
    # Categorize features
    for feature in importance_df['Feature']:
        assigned = False
        for group, pattern in patterns.items():
            if re.match(pattern, feature):
                if group not in feature_groups:
                    feature_groups[group] = 0
                feature_groups[group] += float(importance_df[importance_df['Feature'] == feature]['Importance'])
                assigned = True
                break
                
        if not assigned:
            if 'Other' not in feature_groups:
                feature_groups['Other'] = 0
            feature_groups['Other'] += float(importance_df[importance_df['Feature'] == feature]['Importance'])
    
    # Create DataFrame for group importance
    group_importance = pd.DataFrame({
        'Feature Group': list(feature_groups.keys()),
        'Importance': list(feature_groups.values())
    })
    
    # Sort by importance
    group_importance = group_importance.sort_values('Importance', ascending=False)
    
    # Plot feature group importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature Group', data=group_importance)
    plt.title('Feature Group Importance')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_group_importance.png")
    
    # Save group importance
    group_importance.to_csv(f"{OUTPUT_DIR}/feature_group_importance.csv", index=False)
    
    return importance_df, group_importance

# Function to perform ablation study by removing feature groups
def perform_group_ablation(model, X, y, feature_groups):
    """Remove each feature group and measure the impact on performance."""
    print("Performing feature group ablation study...")
    
    # Get base performance with all features
    y_pred = model.predict(X)
    base_f1 = f1_score(y, y_pred, average='macro')
    print(f"Base F1 score with all features: {base_f1:.4f}")
    
    results = {'All Features': base_f1}
    
    # Test each feature group by removing it
    for group_name, group_cols in feature_groups.items():
        print(f"Testing without {group_name} features...")
        
        # Create copy without this feature group
        X_ablated = X.copy()
        for col in group_cols:
            if col in X_ablated.columns:
                # Set to mean or 0 to neutralize the feature
                X_ablated[col] = 0
        
        # Predict and evaluate
        try:
            y_pred = model.predict(X_ablated)
            ablated_f1 = f1_score(y, y_pred, average='macro')
            
            # Calculate impact (decrease in performance)
            impact = base_f1 - ablated_f1
            
            results[f"Without {group_name}"] = ablated_f1
            print(f"F1 without {group_name}: {ablated_f1:.4f} (Impact: {impact:.4f})")
            
        except Exception as e:
            print(f"Error evaluating without {group_name}: {e}")
            results[f"Without {group_name}"] = None
    
    # Create a DataFrame for visualization
    impact_df = pd.DataFrame({
        'Configuration': list(results.keys()),
        'F1 Score': list(results.values())
    })
    
    # Calculate impact
    impact_df['Impact'] = base_f1 - impact_df['F1 Score']
    impact_df = impact_df.sort_values('Impact', ascending=False)
    
    # Plot feature group ablation results
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Impact', y='Configuration', data=impact_df[impact_df['Configuration'] != 'All Features'])
    plt.title('Feature Group Ablation Impact (Higher = More Important)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_group_ablation.png")
    
    # Save results
    impact_df.to_csv(f"{OUTPUT_DIR}/feature_group_ablation.csv", index=False)
    
    return impact_df

# Function to perform permutation importance
def perform_permutation_importance(model, X, y, n_repeats=10):
    """Calculate permutation importance for all features."""
    print(f"Calculating permutation importance (n_repeats={n_repeats})...")
    
    # Calculate permutation importance
    start_time = time.time()
    perm_importance = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring='f1_macro'
    )
    elapsed_time = time.time() - start_time
    print(f"Permutation importance calculated in {elapsed_time:.2f} seconds")
    
    # Create DataFrame for visualization
    perm_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    })
    
    # Sort by importance
    perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 30 features
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=perm_importance_df.head(30),
        xerr=perm_importance_df.head(30)['Std']
    )
    plt.title('Top 30 Features by Permutation Importance')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/permutation_importance_top30.png")
    
    # Save full importance table
    perm_importance_df.to_csv(f"{OUTPUT_DIR}/permutation_importance.csv", index=False)
    
    return perm_importance_df

# Main function to run the ablation study
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Load test data
    test_data = load_json_data(TEST_DATA_PATH)
    test_comments = extract_comments_from_chains(test_data)
    test_df = comments_to_dataframe(test_comments)
    
    # Extract features
    text_features = extract_text_features(test_df)
    propaganda_features = create_propaganda_features(test_df)
    stance_features = create_stance_features(test_df)
    
    # Use the same PCA components as in training (40)
    embedding_features, pca = extract_embeddings_with_pca(test_df, pca_components=40)
    
    # Combine all features
    all_features = pd.concat([text_features, propaganda_features, stance_features, embedding_features], axis=1)
    
    # Get target values
    target = test_df['divisiveness_score']
    
    # Create feature groups for ablation study
    feature_groups = {
        'Text': text_features.columns.tolist(),
        'Propaganda': propaganda_features.columns.tolist(),
        'Stance': stance_features.columns.tolist(),
        'Embeddings': embedding_features.columns.tolist()
    }
    
    # Print feature counts by group
    print("\nFeature counts by group:")
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)} features")
    
    # Ensure features match model's expected features
    try:
        # Get feature names used during training (if available in model)
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_names_in_'):
            model_features = model.steps[-1][1].feature_names_in_
        else:
            print("Could not extract feature names from model, using all available features")
            model_features = all_features.columns.tolist()
            
        # Subset features to match model's expectations
        print(f"Model expects {len(model_features)} features")
        
        # Check which features are missing
        missing_features = [f for f in model_features if f not in all_features.columns]
        extra_features = [f for f in all_features.columns if f not in model_features]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features expected by model are missing")
            print(f"First few missing features: {missing_features[:5]}")
            
            # Add missing features with zeros
            for feature in missing_features:
                all_features[feature] = 0
                
        if extra_features:
            print(f"Warning: {len(extra_features)} extra features not used by the model")
            print(f"First few extra features: {extra_features[:5]}")
        
        # Reorder columns to match model's expectations
        all_features = all_features[model_features]
        
    except Exception as e:
        print(f"Error matching features: {e}")
        print("Proceeding with all available features")
    
    # Ensure target is in the correct format for model
    # Map target values to indices if needed
    unique_values = sorted(target.unique())
    class_to_index = {val: i for i, val in enumerate(unique_values)}
    target_indices = target.map(class_to_index)
    
    print(f"Target values mapped to indices: {class_to_index}")
    
    # Save feature information
    with open(f"{OUTPUT_DIR}/feature_info.txt", 'w') as f:
        f.write(f"Total features: {len(all_features.columns)}\n\n")
        
        for group, features in feature_groups.items():
            f.write(f"{group} features ({len(features)}):\n")
            for feature in features:
                f.write(f"  {feature}\n")
            f.write("\n")
    
    # 1. XGBoost Built-in Feature Importance
    feature_importance, group_importance = analyze_feature_importance(model, all_features.columns)
    
    # 2. Feature Group Ablation
    group_ablation_results = perform_group_ablation(model, all_features, target_indices, feature_groups)
    
    # 3. Permutation Importance (for top features)
    perm_importance = perform_permutation_importance(model, all_features, target_indices, n_repeats=5)
    
    # Compare results from different importance methods
    print("\nTop 10 features by built-in importance:")
    print(feature_importance.head(10))
    
    print("\nTop 10 features by permutation importance:")
    print(perm_importance.head(10))
    
    print("\nFeature group importance:")
    print(group_importance)
    
    print("\nFeature group ablation impact:")
    print(group_ablation_results)
    
    # Create a comprehensive report
    with open(f"{OUTPUT_DIR}/ablation_study_report.txt", 'w') as f:
        f.write("XGBoost Feature Importance Ablation Study\n")
        f.write("=======================================\n\n")
        
        f.write("Model: XGBoost on All Features, PCA 40\n")
        f.write(f"Model path: {MODEL_PATH}\n\n")
        
        f.write(f"Total features analyzed: {len(all_features.columns)}\n\n")
        
        f.write("Feature Group Summary:\n")
        for group, features in feature_groups.items():
            f.write(f"  {group}: {len(features)} features\n")
        f.write("\n")
        
        f.write("Feature Group Importance (Built-in):\n")
        for _, row in group_importance.iterrows():
            f.write(f"  {row['Feature Group']}: {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("Feature Group Ablation Impact:\n")
        for _, row in group_ablation_results.iterrows():
            if row['Configuration'] != 'All Features':
                f.write(f"  {row['Configuration']}: {row['Impact']:.4f}\n")
        f.write("\n")
        
        f.write("Top 20 Features by Built-in Importance:\n")
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            f.write(f"  {i}. {row['Feature']}: {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("Top 20 Features by Permutation Importance:\n")
        for i, (_, row) in enumerate(perm_importance.head(20).iterrows(), 1):
            f.write(f"  {i}. {row['Feature']}: {row['Importance']:.4f} (±{row['Std']:.4f})\n")
        f.write("\n")
        
        f.write("Conclusions:\n")
        f.write("  1. The most important feature group appears to be: " + 
                group_importance.iloc[0]['Feature Group'] + "\n")
        
        f.write("  2. The top individual feature is: " + 
                feature_importance.iloc[0]['Feature'] + "\n")
        
        f.write("  3. The feature with highest permutation importance is: " + 
                perm_importance.iloc[0]['Feature'] + "\n")
        
        f.write("\nPlots and detailed results saved to: " + OUTPUT_DIR + "\n")
    
    print(f"\nAblation study completed. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
