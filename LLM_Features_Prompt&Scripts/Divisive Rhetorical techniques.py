import openai
import os
import json
import argparse
import yaml
from tqdm import tqdm
import time
from typing import Dict, List, Optional


class YouTubePropagandaInference:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.setup_openai()
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def setup_openai(self) -> None:
        """Setup OpenAI credentials with error checking"""
        try:
            openai.organization = os.environ["OPENAI_ORGANIZATION"]
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not found in environment variables")
        except Exception as e:
            print(f"Error setting up OpenAI credentials: {str(e)}")
            raise

    def load_config(self) -> None:
        """Load configuration with error checking"""
        try:
            with open(self.config_path, 'r') as stream:
                self.model_config = yaml.safe_load(stream)
            required_fields = ['model_name', 'instruction', 'prompt_type',
                               'input_data_path', 'output_path']
            for field in required_fields:
                if field not in self.model_config:
                    raise ValueError(f"Missing required field in config: {field}")
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise

    def prompt_gen(self, input_text: str) -> str:
        """Generate prompt with the same propaganda techniques definitions"""
        prompt_instruction = """You are a multi-label text classifier identifying propaganda techniques within text. These are the propaganda techniques you classify with definitions and examples:
        Loaded_Language - Emotional words and phrases intended to influence audience feelings and reactions.
        Name_Calling,Labeling - Attaching labels or names to discredit or praise without substantive argument.
        Repetition - Multiple restatements of the same message (or word) to reinforce acceptance.
        Exaggeration,Minimisation - Presenting issues as either much worse or much less significant than reality.
        Appeal_to_fear-prejudice - Creating anxiety or panic about potential consequences to gain support.
        Flag-Waving - Exploiting group identity (national, racial, gender, political or religious) to promote a position.
        Causal_Oversimplification - Reducing complex issues to a single cause when multiple factors exist.
        Appeal_to_Authority - Using expert or authority claims to support an argument without additional evidence.
        Slogans/Thought-terminating_Cliches - Striking ready-made phrases that use simplification and common-sense stereotypes to discourage critical thinking.
        Whataboutism,Straw_Men - Deflecting criticism by pointing to opponent's alleged hypocrisy.
        Black-and-White_Fallacy - Presenting complex issues as having only two possible outcomes, or one solution as the only possible one.
        Bandwagon,Reductio_ad_hitlerum - Promoting ideas based on popularity or rejecting them by negative association.
        Doubt - Undermining credibility through questioning motives or expertise.
        Appeal_to_Time - Using deadlines or temporal arguments to create urgency or dismiss current concerns."""

        prompt_base = """
        For the given text please state which of the propaganda techniques are present. If no propaganda technique was identified return "no propaganda detected". An example output would list the propaganda techniques with each technique in a new line, e.g.:
        Loaded_Language
        Thought-terminating_Cliches
        Repetition
        Here is the text:"""

        return f'{prompt_instruction} {prompt_base} <{input_text}>'

    def inference(self, prompt: str) -> str:
        """Make API call with robust error handling and retries"""
        for attempt in range(self.max_retries):
            try:
                # Split the prompt into system and user messages
                prompt_parts = prompt.split("Here is the text:")
                system_message = prompt_parts[0].strip()
                user_text = prompt_parts[1].strip()

                completion = openai.ChatCompletion.create(
                    model=self.model_config['model_name'],
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_text}
                    ],
                    max_tokens=1000,
                    temperature=0.3,
                )

                if hasattr(completion.choices[0], 'message') and 'content' in completion.choices[0].message:
                    return completion.choices[0].message['content']
                else:
                    print(f"Unexpected response format: {completion}")
                    return "no propaganda detected"

            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.error_count += 1
                    print(f"All retries failed. Total errors: {self.error_count}")
                    return "no propaganda detected"

    def process_output(self, output: str) -> List[str]:
        """Process model output with validation"""
        if not output or output.lower().strip() == 'no propaganda detected':
            return []

        valid_techniques = {
            'Appeal_to_Authority', 'Appeal_to_fear-prejudice',
            'Bandwagon,Reductio_ad_hitlerum', 'Black-and-White_Fallacy',
            'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
            'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling',
            'Repetition', 'Slogans/Thought-terminating_Cliches',
            'Whataboutism,Straw_Men', 'Appeal_to_Time'
        }

        techniques = []
        for line in output.split('\n'):
            technique = line.strip()
            if technique in valid_techniques:
                techniques.append(technique)

        return techniques

    def save_results(self) -> None:
        """Process comments and save results with error handling"""
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
            print(f"Processing {len(comments)} comments...")

            for comment in tqdm(comments):
                try:
                    prompt = self.prompt_gen(comment['CommentText'])
                    output = self.inference(prompt)
                    techniques = self.process_output(output)

                    results.append({
                        'CommentID': comment['CommentID'],
                        'CommentText': comment['CommentText'],
                        'Techniques': techniques
                    })
                except Exception as e:
                    print(f"Error processing comment {comment.get('CommentID', 'unknown')}: {str(e)}")
                    continue

            # Save results
            try:
                with open(self.model_config['output_path'], 'w') as f:
                    for result in results:
                        json.dump(result, f)
                        f.write('\n')
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                raise

            print(f"Results saved to {self.model_config['output_path']}")
            if self.error_count > 0:
                print(f"Total API errors encountered: {self.error_count}")

        except Exception as e:
            print(f"Critical error in save_results: {str(e)}")
            raise

    def run_all(self) -> None:
        """Main execution method with error handling"""
        try:
            self.save_results()
        except Exception as e:
            print(f"Failed to complete execution: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()

    try:
        inference = YouTubePropagandaInference(args.config_path)
        inference.run_all()
    except Exception as e:
        print(f"Program failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
