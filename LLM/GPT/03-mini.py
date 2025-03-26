import os
import json
import argparse
import yaml
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from openai import OpenAI


class OpenAIReasonerDivisiveness:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.setup_openai()
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def setup_openai(self) -> None:
        """Setup OpenAI API client with error checking"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")

            # Initialize the OpenAI client
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error setting up OpenAI client: {str(e)}")
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
            valid_prompt_types = ['reasoning_only', 'reasoning_stance', 'reasoning_techniques',
                                  'reasoning_stance_techniques']
            if self.model_config['prompt_type'] not in valid_prompt_types:
                raise ValueError(
                    f"Invalid prompt_type: {self.model_config['prompt_type']}. Must be one of {valid_prompt_types}")

            # Override model name to ensure we're using o3-mini
            if self.model_config['model_name'] != 'o3-mini':
                print(f"Overriding model name '{self.model_config['model_name']}' to 'o3-mini'")
                self.model_config['model_name'] = 'o3-mini'

        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise

    def prompt_gen(self, comment_pair: Dict) -> Tuple[str, str]:
        """Generate prompt based on the prompt type, focusing on reasoning steps"""
        parent_text = comment_pair.get("ParentCommentText", "")
        child_text = comment_pair.get("CommentText", "")
        child_stance = comment_pair.get("Stance_Label", None)
        parent_stance = comment_pair.get("parent_comment_stance", None)
        techniques = comment_pair.get("Techniques", [])

        # Base system message with divisiveness guidelines
        system_message = """
        You are an analyzer assessing how online comments affect conversation dynamics. 
        Focus on HOW content is communicated, not WHAT beliefs are expressed.

        Your goal is to analyze parent-child comment pairs and determine a divisiveness score 
        based on a specific scale. You will think through the analysis step by step before 
        providing a final score.

        The divisiveness scale is:

        Constructive Disagreement/Attack: Messages that express disagreement while maintaining 
        the conditions for mutual understanding. They challenge views by treating other participants 
        as equal contributors to the dialogue. These messages demonstrate perspective-taking and 
        articulated interests even while disagreeing.

        Constructive Agreement/Support: Messages that strengthen mutual understanding while 
        agreeing. They contribute to rational dialogue advancing the conversation by adding substance 
        that others can engage with. These messages enhance relations while supporting previous points.

        Neutral/Unrelated: These comments serve the discussion process rather than advancing 
        or challenging arguments. They often function as bridges between perspectives by seeking 
        information, adding context, or maintaining the conversation space. Questions seeking 
        additional information, factual additions that extend without supporting or attacking, 
        topic shifts that redirect conversation, and procedural comments all fall into this category. 
        Unlike agreement or disagreement comments, neutral messages maintain conversational neutrality, 
        neither intensifying nor reducing tensions between different positions.

        Destructive Agreement/Support: Messages that strengthen divisions while agreeing by 
        treating opposing views as invalid or contemptible. They build group solidarity through 
        shared rejection of others rather than through constructive dialogue. These messages exhibit 
        self-focused orientation, harm-directed behavior, and in-group loyalty while supporting 
        previous comments.

        Destructive Disagreement/Attack: Comments that hinder productive dialogue through 
        hostile language or mockery. Rather than engaging with different views, they aim to dominate 
        the conversation by delegitimizing others and their perspectives. These messages aim to shut 
        down dialogue through inflammatory language, hostile tone, or aggressive rhetoric, regardless 
        of the view being expressed. They prioritize defeating the interlocutor over maintaining 
        dialogue conditions.
        """

        prompt_type = self.model_config['prompt_type']

        base_reasoning_section = """
        Use the information I provided you to determine the divisiveness score. Specifically you should use them in relation to these questions:
        - Is the comment constructive or destructive to dialogue?
        - Does it engage with the parent's ideas or attack/dismiss them?
        - Does it contribute to mutual understanding or promote division?
        - Does it show perspective-taking and respectful engagement?

        Explain your reasoning step by step, then conclude with your final score selection.

            Respond with only one of these exact labels:
            Constructive_Disagreement/Attack
            Constructive_Agreement/Support
            Rephrase/Other
            Destructive_Agreement/Support
            Destructive_Disagreement/Attack
        """

        if prompt_type == "reasoning_only":
            user_prompt = f"""
            I'll provide you with a parent-child comment pair. 
            Your task is to analyze how the child comment responds to the parent, focusing on the conversational dynamics rather than the opinions expressed.
            Specifically, you should be first try to understand the relation between the 2 comments (attack, support or rephrase). 
            Then try to understand if the child comment is engaging with the parent in a way that promotes controversy/destructive-interaction or rather a discussion/constructive-interaction.

            Parent Comment: {parent_text}

            Child Comment (responding to parent): {child_text}

            {base_reasoning_section}
            """

        elif prompt_type == "reasoning_stance":
            stance_mapping = {0: "contra", 1: "neutral", 2: "pro"}
            parent_stance_text = stance_mapping.get(parent_stance, "unknown") if parent_stance is not None else "unknown"
            child_stance_text = stance_mapping.get(child_stance, "unknown") if child_stance is not None else "unknown"

            user_prompt = f"""
            I'll provide you with a parent-child comment pair discussing immigration. 
            Your task is to analyze how the child comment responds to the parent, focusing on the conversational dynamics rather than the opinions expressed.
            I'll provide you also as additional information the stance of each comment towards the topic of immigration.

            Parent Comment: {parent_text}
            This parent comment has a {parent_stance_text} stance towards immigration.

            Child Comment (responding to parent): {child_text}
            This child comment has a {child_stance_text} stance towards immigration. 

            {base_reasoning_section}
            """

        elif prompt_type == "reasoning_techniques":
            tech_mapping = {
                'Appeal_to_Authority': 'Appeal_to_Authority, which uses expert or authority claims to support an argument.',
                'Appeal_to_fear-prejudice': 'Appeal_to_fear-prejudice, which aims at creating anxiety or panic about potential consequences to gain support.',
                'Bandwagon,Reductio_ad_hitlerum': 'Bandwagon/Reductio_ad_hitlerum, which promotes ideas based on popularity or rejecting them by negative association.',
                'Black-and-White_Fallacy': 'Black-and-White_Fallacy, which presents complex issues as having only two possible outcomes, or one solution as the only possible one.',
                'Causal_Oversimplification': 'Causal_Oversimplification, which reduces complex issues to a single cause when multiple factors exist.',
                'Doubt': 'Doubt, which undermines credibility through questioning motives or expertise.',
                'Exaggeration,Minimisation': 'Exaggeration,Minimisation, which presents issues as either much worse or much less significant than reality.',
                'Flag-Waving': 'Flag-Waving, which exploits group identity (national, racial, gender, political or religious) to promote a position.',
                'Loaded_Language': 'Loaded_Language, which uses emotional words and phrases intended to influence interlocutor feelings and reactions.',
                'Name_Calling,Labeling': 'Name_Calling,Labeling, which attaches labels or names to discredit or praise without substantive argument.',
                'Repetition': 'Repetition, which uses restatements of the same message (or word) to reinforce acceptance.',
                'Slogans/Thought-terminating_Cliches': 'Slogans/Thought-terminating_Cliches, which uses ready-made phrases that use simplification and common-sense stereotypes to discourage critical thinking.',
                'Whataboutism,Straw_Men': 'Whataboutism,Straw_Men, which deflects criticism by pointing to opponent alleged hypocrisy.',
                'Appeal_to_Time': 'Appeal_to_Time, which uses deadlines or temporal arguments to create urgency or dismiss current concerns.'
            }
            if techniques:
                techniques_text = ""
                for tech in techniques:
                    techniques_text += f"- {tech_mapping.get(tech, tech)}\n"
            else:
                techniques_text = "None detected"

            user_prompt = f"""
            I'll provide you with a parent-child comment pair discussing immigration. 
            Your task is to analyze how the child comment responds to the parent, focusing on the conversational dynamics rather than the opinions expressed.
            I'll provide you also two additional information the list of divisive rhetorical techniques detected in the child text, if present. 

            Parent Comment: {parent_text}

            Child Comment (responding to parent): {child_text}

            The child comment text contains the following rhetorical techniques:
            {techniques_text}

            {base_reasoning_section}
            """


        elif prompt_type == "reasoning_stance_techniques":
            stance_mapping = {0: "contra", 1: "neutral", 2: "pro"}
            parent_stance_text = stance_mapping.get(parent_stance,
                                                    "unknown") if parent_stance is not None else "unknown"
            child_stance_text = stance_mapping.get(child_stance, "unknown") if child_stance is not None else "unknown"

            tech_mapping = {
                'Appeal_to_Authority': 'Appeal_to_Authority, which uses expert or authority claims to support an argument.',
                'Appeal_to_fear-prejudice': 'Appeal_to_fear-prejudice, which aims at creating anxiety or panic about potential consequences to gain support.',
                'Bandwagon,Reductio_ad_hitlerum': 'Bandwagon/Reductio_ad_hitlerum, which promotes ideas based on popularity or rejecting them by negative association.',
                'Black-and-White_Fallacy': 'Black-and-White_Fallacy, which presents complex issues as having only two possible outcomes, or one solution as the only possible one.',
                'Causal_Oversimplification': 'Causal_Oversimplification, which reduces complex issues to a single cause when multiple factors exist.',
                'Doubt': 'Doubt, which undermines credibility through questioning motives or expertise.',
                'Exaggeration,Minimisation': 'Exaggeration,Minimisation, which presents issues as either much worse or much less significant than reality.',
                'Flag-Waving': 'Flag-Waving, which exploits group identity (national, racial, gender, political or religious) to promote a position.',
                'Loaded_Language': 'Loaded_Language, which uses emotional words and phrases intended to influence interlocutor feelings and reactions.',
                'Name_Calling,Labeling': 'Name_Calling,Labeling, which attaches labels or names to discredit or praise without substantive argument.',
                'Repetition': 'Repetition, which uses restatements of the same message (or word) to reinforce acceptance.',
                'Slogans/Thought-terminating_Cliches': 'Slogans/Thought-terminating_Cliches, which uses ready-made phrases that use simplification and common-sense stereotypes to discourage critical thinking.',
                'Whataboutism,Straw_Men': 'Whataboutism,Straw_Men, which deflects criticism by pointing to opponent alleged hypocrisy.',
                'Appeal_to_Time': 'Appeal_to_Time, which uses deadlines or temporal arguments to create urgency or dismiss current concerns.'
            }

            if techniques:
                techniques_text = ""
                for tech in techniques:
                    techniques_text += f"- {tech_mapping.get(tech, tech)}\n"
            else:
                techniques_text = "None detected"

            user_prompt = f"""
            I'll provide you with a parent-child comment pair discussing immigration. 
            Your task is to analyze how the child comment responds to the parent, focusing on the conversational dynamics rather than the opinions expressed.
            I'll provide you also two additional information. 
            1) The stance of each comment towards the topic of immigration.
            2) If present, the list of divisive rhetorical techniques detected in the text. 

            Parent Comment: {parent_text}
            This parent comment has a {parent_stance_text} stance towards immigration.

            Child Comment (responding to parent): {child_text}
            This child comment has a {child_stance_text} stance towards immigration. 

            The child comment text contains the following rhetorical techniques:
            {techniques_text}

            {base_reasoning_section}
            """

        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}")

        return system_message, user_prompt

    def inference(self, system_message: str, user_prompt: str) -> str:
        """Make API call to OpenAI using the official client with robust error handling and retries"""
        for attempt in range(self.max_retries):
            try:
                # Using the OpenAI Python client to make the request
                response = self.client.chat.completions.create(
                    model=self.model_config['model_name'],
                    messages=[
                        {"role": "assistant", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=2000,
                    reasoning_effort="medium"
                )


                return response.choices[0].message.content

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
        # Clean the output but don't convert case yet
        output = output.strip()

        # Define label to score mapping with original capitalization
        valid_scores = {
            "Constructive_Disagreement/Attack": 1.0,
            "Constructive_Agreement/Support": 0.5,
            "Rephrase/Other": 0.0,
            "Destructive_Agreement/Support": -0.5,
            "Destructive_Disagreement/Attack": -1.0
        }

        # First, try exact match with original case
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

    def run_evaluation(self) -> Dict:
        """Run evaluation and save results"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.model_config['output_path']), exist_ok=True)

            # Read and validate input data
            try:
                with open(self.model_config['input_data_path'], 'r') as f:
                    comments = [json.loads(line) for line in f]
            except Exception as e:
                print(f"Error reading input file: {str(e)}")
                raise

            results = []
            true_scores = []
            pred_scores = []

            print(f"Processing {len(comments)} comment pairs...")
            print(f"Using prompt type: {self.model_config['prompt_type']}")
            print(f"Using OpenAI model: {self.model_config['model_name']}")

            for comment in tqdm(comments):
                try:
                    # Skip comments without divisiveness_score (ground truth)
                    if 'divisiveness_score' not in comment:
                        print(f"Skipping comment without divisiveness_score: {comment.get('CommentID', 'unknown')}")
                        continue

                    # Generate prompt
                    system_message, user_prompt = self.prompt_gen(comment)

                    # Get model prediction with reasoning
                    output = self.inference(system_message, user_prompt)

                    # Process output to get predicted score
                    pred_score = self.process_output(output)

                    # Add to results
                    true_score = float(comment['divisiveness_score'])
                    true_scores.append(true_score)
                    pred_scores.append(pred_score)

                    result = {
                        'CommentID': comment['CommentID'],
                        'ParentCommentText': comment.get('ParentCommentText', ''),
                        'CommentText': comment['CommentText'],
                        'true_score': true_score,
                        'predicted_score': pred_score,
                        'correct': true_score == pred_score,
                        'reasoning_output': output
                    }

                    results.append(result)

                except Exception as e:
                    print(f"Error processing comment {comment.get('CommentID', 'unknown')}: {str(e)}")
                    continue

            # Calculate metrics
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

            # Save results
            output_data = {
                "prompt_type": self.model_config['prompt_type'],
                "model_name": self.model_config['model_name'],
                "metrics": metrics,
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

            # Print summary metrics
            print("\nPerformance Summary:")
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

            return metrics

        except Exception as e:
            print(f"Critical error in run_evaluation: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()

    try:
        detector = OpenAIReasonerDivisiveness(args.config_path)
        detector.run_evaluation()
    except Exception as e:
        print(f"Program failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
