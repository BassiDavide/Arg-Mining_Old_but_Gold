import os
import json
import argparse
import yaml
import time
import numpy as np
import requests
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DivisivenessDetection:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.setup_deepseek()
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Define relation and chain category mappings
        self.RELATION_MAP = {
            -1.0: "Destructive Disagreement",
            -0.5: "Destructive Agreement",
            0.0: "Rephrase",
            0.5: "Constructive Agreement",
            1.0: "Constructive Disagreement"
        }
        
        self.CHAIN_CATEGORIES = {
            "Highly Destructive": (-1.0, -0.75),
            "Moderately Destructive": (-0.75, -0.25),
            "Slightly Destructive-Constructive/Neutral": (-0.25, 0.25),
            "Constructive": (0.25, 1.0)
        }

    def setup_deepseek(self) -> None:
        """Setup DeepSeek API credentials with error checking"""
        try:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DeepSeek API key not found in environment variables")

            # DeepSeek API endpoint
            self.api_url = "https://api.deepseek.com/v1/chat/completions"
        except Exception as e:
            print(f"Error setting up DeepSeek credentials: {str(e)}")
            raise

    def load_config(self) -> None:
        """Load configuration with error checking"""
        try:
            with open(self.config_path, 'r') as stream:
                self.model_config = yaml.safe_load(stream)
            required_fields = ['model_name', 'prompt_type', 'input_data_path', 'output_path']
            for field in required_fields:
                if field not in self.model_config:
                    raise ValueError(f"Missing required field in config: {field}")

            # Validate prompt_type
            valid_prompt_types = ['text_only', 'text_stance', 'text_techniques', 'text_stance_techniques']
            if self.model_config['prompt_type'] not in valid_prompt_types:
                raise ValueError(
                    f"Invalid prompt_type: {self.model_config['prompt_type']}. Must be one of {valid_prompt_types}")

            # Convert OpenAI model name to DeepSeek model name if needed
            if self.model_config['model_name'].startswith('gpt-'):
                print(f"Converting OpenAI model name '{self.model_config['model_name']}' to DeepSeek model...")
                # Map common OpenAI models to DeepSeek equivalents
                model_mapping = {
                    'gpt-4o': 'deepseek-chat',
                    'gpt-4': 'deepseek-chat',
                    'gpt-3.5-turbo': 'deepseek-chat'
                }
                self.model_config['model_name'] = model_mapping.get(
                    self.model_config['model_name'], 'deepseek-chat'
                )
                print(f"Using DeepSeek model: {self.model_config['model_name']}")

        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise

    def prompt_gen(self, comment_pair: Dict) -> Tuple[str, str]:
        """Generate prompt based on specified prompt type"""
        parent_text = comment_pair.get("ParentCommentText", "")
        child_text = comment_pair.get("CommentText", "")
        child_stance = comment_pair.get("Stance_Label", None)
        parent_stance = comment_pair.get("parent_comment_stance", None)
        techniques = comment_pair.get("Techniques", [])

        # Base system message with detailed divisiveness guidelines
        system_message = """
        You are an analyzer assessing how online comments affect conversation dynamics. Focus on HOW content is communicated, not WHAT beliefs are expressed.

        Analyze the following parent-child comment pair and determine the divisiveness score based on this scale:

        Constructive Disagreement/Attack: Messages that express disagreement while maintaining the conditions for mutual understanding. They challenge views by engaging with specific validity claims that can be rationally discussed, providing reasons and/or examples that others can accept or challenge, and treating other participants as equal contributors to the dialogue. These messages demonstrate perspective-taking and articulated interests even while disagreeing.

        Constructive Agreement/Support: Messages that strengthen mutual understanding while agreeing. They contribute to rational dialogue by providing specific reasons for agreement, and advancing the conversation by adding substance that others can engage with. These messages enhance relations while supporting previous points.

        Neutral/Unrelated: These comments serve the discussion process rather than advancing or challenging arguments. They often function as bridges between perspectives by seeking information, adding context, or maintaining the conversation space. Questions seeking additional information, factual additions that extend without supporting or attacking, topic shifts that redirect conversation, and procedural comments all fall into this category. Unlike agreement or disagreement comments, neutral messages maintain conversational neutrality, neither intensifying nor reducing tensions between different positions.

        Destructive Agreement/Support: Messages that strengthen divisions while agreeing by treating opposing views as invalid or contemptible. They build group solidarity through shared rejection of others rather than through constructive dialogue. These messages exhibit self-focused orientation, harm-directed behavior, and in-group loyalty while supporting previous comments.

        Destructive Disagreement/Attack: Comments that hinder productive dialogue through hostile language or mockery. Rather than engaging with different views, they aim to dominate the conversation by delegitimizing others and their perspectives. These messages aim to shut down dialogue through inflammatory language, hostile tone, or aggressive rhetoric, regardless of the view being expressed. They prioritize defeating the interlocutor over maintaining dialogue conditions.
        """

        # Generate user prompt based on prompt type
        prompt_type = self.model_config['prompt_type']

        if prompt_type == "text_only":
            user_prompt = f"""
            Parent Comment: {parent_text}

            Child Comment (responding to parent): {child_text}

            Based on the parent-child comment pair above, determine the divisiveness score. Focus on HOW the child comment engages with the parent comment, not on what opinions are expressed.

            Respond with only one of these exact labels:
            Constructive_Disagreement/Attack
            Constructive_Agreement/Support
            Rephrase/Other
            Destructive_Agreement/Support
            Destructive_Disagreement/Attack
            """

        elif prompt_type == "text_stance":
            # Map stance values to descriptive labels
            stance_mapping = {0: "against", 1: "for", 2: "neutral"}
            parent_stance_text = stance_mapping.get(parent_stance,"unknown") if parent_stance is not None else "unknown"

            child_stance_text = stance_mapping.get(child_stance, "unknown") if child_stance is not None else "unknown"

            user_prompt = f"""
            Parent Comment: {parent_text}
            Parent Stance on Immigration: {parent_stance_text}

            Child Comment (responding to parent): {child_text}
            Child Stance on Immigration: {child_stance_text}

            Based on the parent-child comment pair above, determine the divisiveness score. Remember that divisiveness is about HOW the conversation occurs, not about whether the stances agree or disagree.

            Respond with only one of these exact labels:
            Constructive_Disagreement/Attack
            Constructive_Agreement/Support
            Rephrase/Other
            Destructive_Agreement/Support
            Destructive_Disagreement/Attack
            """

        elif prompt_type == "text_techniques":
            techniques_text = ", ".join(techniques) if techniques else "None detected"

            user_prompt = f"""
            Parent Comment: {parent_text}

            Child Comment (responding to parent): {child_text}
            Propaganda Techniques in Child Comment: {techniques_text}

            Based on the parent-child comment pair above, determine the divisiveness score. Consider how the detected propaganda techniques might affect the quality of dialogue.

            Respond with only one of these exact labels:
            Constructive_Disagreement/Attack
            Constructive_Agreement/Support
            Rephrase/Other
            Destructive_Agreement/Support
            Destructive_Disagreement/Attack
            """

        elif prompt_type == "text_stance_techniques":
            # Map stance values to descriptive labels
            stance_mapping = {0: "against", 1: "for", 2: "neutral"}
            parent_stance_text = stance_mapping.get(parent_stance,
                                                    "unknown") if parent_stance is not None else "unknown"
            child_stance_text = stance_mapping.get(child_stance, "unknown") if child_stance is not None else "unknown"
            techniques_text = ", ".join(techniques) if techniques else "None detected"

            user_prompt = f"""
            Parent Comment: {parent_text}
            Parent Stance on Immigration: {parent_stance_text}

            Child Comment (responding to parent): {child_text}
            Child Stance on Immigration: {child_stance_text}
            Propaganda Techniques in Child Comment: {techniques_text}

            Based on the parent-child comment pair above, determine the divisiveness score. Consider both the stance alignment and the propaganda techniques, but focus on HOW the conversation occurs.

            Respond with only one of these exact labels:
            Constructive_Disagreement/Attack
            Constructive_Agreement/Support
            Rephrase/Other
            Destructive_Agreement/Support
            Destructive_Disagreement/Attack
            """

        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}")

        return system_message, user_prompt

    def inference(self, system_message: str, user_prompt: str) -> str:
        """Make API call to DeepSeek with robust error handling and retries"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model_config['model_name'],
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 100,  # Short response sufficient for score
            "temperature": 0.1  # Low temperature for more consistent output
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses

                response_data = response.json()

                # Extract the assistant's message content from DeepSeek response
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    if 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                        return response_data['choices'][0]['message']['content']

                print(f"Unexpected response format: {response_data}")
                return "0"  # Default to neutral if response format is unexpected

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.error_count += 1
                    print(f"All retries failed. Total errors: {self.error_count}")
                    return "0"  # Default to neutral if all retries fail

    def process_output(self, output: str) -> float:
        """Process model output to extract score"""
        # Clean and normalize the output
        output = output.strip()

        # Try to find the score in the output
        valid_scores = {
            "Constructive_Disagreement/Attack": 1.0,
            "Constructive_Agreement/Support": 0.5,
            "Rephrase/Other": 0.0,
            "Destructive_Agreement/Support": -0.5,
            "Destructive_Disagreement/Attack": -1.0
        }

        if output in valid_scores:
            return valid_scores[output]

        # Create lowercase versions for comparison
        output_lower = output.lower()
        valid_scores_lower = {k.lower(): v for k, v in valid_scores.items()}

        # Try exact match with lowercase
        if output_lower in valid_scores_lower:
            return valid_scores_lower[output_lower]

        # If no exact match, try substring matches with lowercase
        for score_str, score_val in valid_scores_lower.items():
            if score_str in output_lower:
                return score_val

        # Fallback to more flexible matching
        if "constructive" in output_lower and ("disagree" in output_lower or "attack" in output_lower):
            return 1.0
        elif "constructive" in output_lower and ("agree" in output_lower or "support" in output_lower):
            return 0.5
        elif "rephrase" in output_lower or "other" in output_lower or "neutral" in output_lower:
            return 0.0
        elif "destructive" in output_lower and ("agree" in output_lower or "support" in output_lower):
            return -0.5
        elif "destructive" in output_lower and ("disagree" in output_lower or "attack" in output_lower):
            return -1.0

        # Default to neutral if no clear indication
        print(f"Could not determine score from output: '{output}', defaulting to 0")
        return 0.0

    def calculate_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict:
        """Calculate performance metrics"""
        # Convert float scores to string labels for clarity in reporting
        score_to_label = {
            -1.0: "destructive_attack",
            -0.5: "destructive_agreement",
            0.0: "neutral",
            0.5: "constructive_agreement",
            1.0: "constructive_attack"
        }

        # Convert scores to categorical classes for metrics calculation
        # Map the float values to integers 0-4 for metrics calculation
        score_to_index = {
            -1.0: 0,
            -0.5: 1,
            0.0: 2,
            0.5: 3,
            1.0: 4
        }

        y_true_indices = [score_to_index[score] for score in y_true]
        y_pred_indices = [score_to_index[score] for score in y_pred]

        # Calculate metrics using the integer indices
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
        )

        accuracy = accuracy_score(y_true_indices, y_pred_indices)
        f1_macro = f1_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)
        f1_micro = f1_score(y_true_indices, y_pred_indices, average='micro', zero_division=0)

        # Create metrics dictionary
        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "class_metrics": {}
        }

        # Add per-class metrics
        labels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for i, label in enumerate(labels):
            label_name = score_to_label[label]
            metrics["class_metrics"][label_name] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i])
            }

        return metrics

    def generate_confusion_matrix(self, y_true: List[float], y_pred: List[float]) -> np.ndarray:
        """Generate a confusion matrix for the results"""
        # Define all possible score values
        scores = [-1.0, -0.5, 0.0, 0.5, 1.0]
        n_scores = len(scores)

        # Initialize confusion matrix
        conf_matrix = np.zeros((n_scores, n_scores), dtype=int)

        # Fill confusion matrix
        for true, pred in zip(y_true, y_pred):
            try:
                true_idx = scores.index(true)
                pred_idx = scores.index(pred)
                conf_matrix[true_idx, pred_idx] += 1
            except ValueError:
                print(f"Warning: Unexpected value encountered: true={true}, pred={pred}")
                continue

        return conf_matrix
    
    # New methods for handling nested JSON and chain evaluation
    
    def load_json_data(self, file_path: str) -> Optional[Dict]:
        """Load nested JSON data with error handling."""
        print(f"Loading data from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def extract_comments_from_chains(self, data: Dict) -> List[Dict]:
        """Extract individual comments from the chain data."""
        comments = []

        for chain_id, chain_data in data['discussions'].items():
            for message in chain_data['messages']:
                # Skip root comments (level 0)
                if message.get('Level', 0) > 0:
                    # Add chain_id to the message for later reference
                    message['chain_id'] = chain_id
                    # Handle parent comment text
                    if 'ParentCommentID' in message and message['ParentCommentID']:
                        # Find parent comment and add its text
                        for parent_msg in chain_data['messages']:
                            if parent_msg.get('CommentID') == message['ParentCommentID']:
                                # Add the parent's stance if available
                                if 'Stance_Label' in parent_msg:
                                    message['parent_comment_stance'] = parent_msg['Stance_Label']
                                break
                    comments.append(message)

        print(f"Extracted {len(comments)} comments from chains")
        return comments
    
    def evaluate_chain_predictions(self, data: Dict, results: List[Dict]) -> Dict:
        """Evaluate chain-level predictions by aggregating comment scores."""
        print("\nEvaluating chain-level predictions...")
        
        # Create dictionary to organize results by chain
        chain_results = defaultdict(list)
        
        # Group results by chain
        for result in results:
            if 'chain_id' in result:
                chain_results[result['chain_id']].append(result)
        
        # Calculate chain-level metrics
        chain_metrics = {}
        
        for chain_id, comments in chain_results.items():
            if not comments:
                continue
                
            # Calculate average divisiveness score for the chain
            true_scores = [c['true_score'] for c in comments if 'true_score' in c]
            pred_scores = [c['predicted_score'] for c in comments if 'predicted_score' in c]
            
            if not true_scores or not pred_scores:
                continue
                
            true_avg = sum(true_scores) / len(true_scores)
            pred_avg = sum(pred_scores) / len(pred_scores)
            
            # Categorize chain based on average scores
            def get_category(score):
                for category, (min_val, max_val) in self.CHAIN_CATEGORIES.items():
                    if min_val <= score <= max_val:
                        return category
                return "Unknown"
            
            true_category = get_category(true_avg)
            pred_category = get_category(pred_avg)
            
            # Store chain metrics
            chain_metrics[chain_id] = {
                'true_avg': true_avg,
                'pred_avg': pred_avg,
                'true_category': true_category,
                'pred_category': pred_category,
                'correct': true_category == pred_category,
                'num_comments': len(comments)
            }
        
        # Calculate overall accuracy
        if chain_metrics:
            accuracy = sum(1 for m in chain_metrics.values() if m['correct']) / len(chain_metrics)
            print(f"Chain Category Prediction Accuracy: {accuracy:.4f}")
        else:
            accuracy = 0
            print("No chain metrics available for evaluation")
        
        # Create and save visualizations
        self.create_chain_visualizations(chain_metrics)
        
        return {
            'chain_metrics': chain_metrics,
            'chain_accuracy': accuracy
        }
    
    def create_chain_visualizations(self, chain_metrics: Dict) -> None:
        """Create visualizations for chain-level predictions."""
        if not chain_metrics:
            print("No chain metrics available for visualization")
            return
            
        # Create confusion matrix for chain categories
        categories = list(self.CHAIN_CATEGORIES.keys())
        category_cm = np.zeros((len(categories), len(categories)), dtype=int)
        
        for result in chain_metrics.values():
            try:
                true_idx = categories.index(result['true_category'])
                pred_idx = categories.index(result['pred_category'])
                category_cm[true_idx, pred_idx] += 1
            except (ValueError, KeyError):
                # Skip entries with invalid categories
                continue
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(category_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories, yticklabels=categories)
        plt.title('Chain Category Confusion Matrix')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.savefig('chain_category_confusion_matrix.png')
        plt.close()
        
        # Plot accuracy by chain size
        self.plot_chain_accuracy_by_size(chain_metrics)
        
        # Save detailed chain results to CSV
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
            for chain_id, data in chain_metrics.items()
        ])
        
        chain_results_df.to_csv('chain_prediction_results.csv', index=False)
        print("Saved detailed chain results to 'chain_prediction_results.csv'")
    
    def plot_chain_accuracy_by_size(self, chain_metrics: Dict) -> None:
        """Plot accuracy by chain size."""
        # Group chains by size
        size_groups = defaultdict(list)

        for chain_id, data in chain_metrics.items():
            size = data['num_comments']
            # Group sizes to avoid too many categories
            group = 1 if size == 1 else 2 if size == 2 else 3 if size == 3 else 4 if size <= 5 else 6 if size <= 10 else 11
            size_groups[group].append(data['correct'])

        # Calculate accuracy for each size group
        size_accuracy = {}
        for size, results in size_groups.items():
            if results:
                accuracy = sum(results) / len(results)
                size_accuracy[size] = {
                    'accuracy': accuracy,
                    'count': len(results)
                }

        if not size_accuracy:
            print("No size groups available for plotting")
            return

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
        plt.savefig('chain_accuracy_by_size.png')
        plt.close()

    def run_evaluation(self) -> Dict:
        """Run evaluation with the new JSON format and chain evaluation"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.model_config['output_path']), exist_ok=True)

            # Load nested JSON data
            data = self.load_json_data(self.model_config['input_data_path'])
            if not data:
                raise ValueError("Failed to load input data")

            # Extract comments from chains
            comments = self.extract_comments_from_chains(data)
            if not comments:
                raise ValueError("No comments extracted from chains")

            results = []
            true_scores = []
            pred_scores = []

            print(f"Processing {len(comments)} comment pairs...")
            print(f"Using prompt type: {self.model_config['prompt_type']}")
            print(f"Using DeepSeek model: {self.model_config['model_name']}")

            for comment in tqdm(comments):
                try:
                    # Skip comments without required fields
                    if 'CommentText' not in comment or 'ParentCommentText' not in comment:
                        print(f"Skipping comment missing required text fields: {comment.get('CommentID', 'unknown')}")
                        continue

                    # Generate prompt
                    system_message, user_prompt = self.prompt_gen(comment)

                    # Get model prediction
                    output = self.inference(system_message, user_prompt)

                    # Process output to get predicted score
                    pred_score = self.process_output(output)

                    # Get true score (if available) or set to None
                    true_score = float(comment['divisiveness_score']) if 'divisiveness_score' in comment else None

                    # Track scores if true score is available
                    if true_score is not None:
                        true_scores.append(true_score)
                        pred_scores.append(pred_score)

                    # Add to results
                    result = {
                        'CommentID': comment['CommentID'],
                        'chain_id': comment.get('chain_id', ''),
                        'ParentCommentText': comment.get('ParentCommentText', ''),
                        'CommentText': comment['CommentText'],
                        'predicted_score': pred_score,
                        'raw_model_output': output
                    }
                    
                    # Add true score if available
                    if true_score is not None:
                        result['true_score'] = true_score
                        result['correct'] = true_score == pred_score

                    results.append(result)

                except Exception as e:
                    print(f"Error processing comment {comment.get('CommentID', 'unknown')}: {str(e)}")
                    continue

            # Calculate comment-level metrics if true scores are available
            if true_scores and pred_scores:
                metrics = self.calculate_metrics(true_scores, pred_scores)
                
                # Generate confusion matrix
                conf_matrix = self.generate_confusion_matrix(true_scores, pred_scores)
                
                # Add confusion matrix to metrics
                score_labels = ["destructive_attack", "destructive_agreement", "neutral",
                                "constructive_agreement", "constructive_attack"]
                metrics["confusion_matrix"] = {
                    "matrix": conf_matrix.tolist(),
                    "labels": score_labels
                }
            else:
                metrics = {"message": "No ground truth scores available for evaluation"}
            
            # Add chain-level evaluation
            chain_evaluation = self.evaluate_chain_predictions(data, results)
            
            # Save results
            output_data = {
                "prompt_type": self.model_config['prompt_type'],
                "model_name": self.model_config['model_name'],
                "comment_metrics": metrics,
                "chain_evaluation": chain_evaluation,
                "results": results
            }

            try:
                with open(self.model_config['output_path'], 'w') as f:
                    json.dump(output_data, f, indent=2)
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                raise

            print(f"Results saved to {self.model_config['output_path']}")
            if self.error_count > 0:
                print(f"Total API errors encountered: {self.error_count}")

            # Print summary metrics if available
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                print("\nComment-Level Performance Summary:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"F1 Macro: {metrics['f1_macro']:.4f}")
                print(f"F1 Micro: {metrics['f1_micro']:.4f}")

                print("\nPer-Class Metrics:")
                for class_name, class_metrics in metrics["class_metrics"].items():
                    print(f"  {class_name}:")
                    print(f"    Precision: {class_metrics['precision']:.4f}")
                    print(f"    Recall: {class_metrics['recall']:.4f}")
                    print(f"    F1: {class_metrics['f1']:.4f}")
                    print(f"    Support: {class_metrics['support']}")

                # Print confusion matrix
                print("\nConfusion Matrix:")
                print("Predicted →")
                print("True ↓     | -1.0 | -0.5 |  0.0 |  0.5 |  1.0")
                print("-----------+------+------+------+------+------")

                for i, row in enumerate(conf_matrix):
                    score = [-1.0, -0.5, 0.0, 0.5, 1.0][i]
                    print(f"  {score:>5}  | {row[0]:4d} | {row[1]:4d} | {row[2]:4d} | {row[3]:4d} | {row[4]:4d}")

            # Print chain evaluation summary
            print("\nChain-Level Evaluation Summary:")
            print(f"Chain Category Prediction Accuracy: {chain_evaluation['chain_accuracy']:.4f}")
            print(f"Total Chains Evaluated: {len(chain_evaluation['chain_metrics'])}")
            
            return output_data

        except Exception as e:
            print(f"Critical error in run_evaluation: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()

    try:
        detector = DivisivenessDetection(args.config_path)
        detector.run_evaluation()
    except Exception as e:
        print(f"Program failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
