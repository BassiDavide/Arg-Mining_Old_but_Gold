import os
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.environ.get("ENTER API KEY"))

SYSTEM_PROMPT = system_prompt_template = (
    """You are tasked with analyzing a pair of text entries, a "Comment" and its "Parent Comment", to determine the stance towards immigration.

    1. Read and understand the Parent Comment's immigration stance
    2. Analyze how the Response Comment relates to it, watching for:
       
    Classify as 0 (Against) if the comment:
    - references cultural/religious differences negatively
    - uses irony about integration
    - expresses nationalist solidarity against immigration
    - agrees with anti-immigration parent comments
    - disagrees with pro-immigration parent comments
    - implies immigrants should adapt or leave

    Classify as 1 (Neutral) if the comment:
    - makes factual statements without stance
    - asks genuine clarifying questions
    - discusses unrelated aspects
    - is too ambiguous to determine stance
    - cannot be clearly connected to immigration views
    
    Classify as 2 (Support) if the comment:
    - challenges anti-immigration views or disagree with anti-immigration parent comments
    - agrees with pro-immigration comments
    - highlights immigrant contributions
    - points out systemic issues
    - shares positive integration experiences
    - criticizes immigration restrictions
    - defends immigrant rights/cultures
    
    CRUCIAL:
    - Context determines meaning of short/ambiguous responses
    - Watch for irony and cultural/religious references
    - Personal experiences often reveal stance
    - National pride expressions and solidarity often signals stance
    
    CRUCIAL: Do not overthink. If you see ANY stance indicator, even subtle, immediately use it to classify as 0 or 2.
    CRUCIAL: Answer ONLY with the number (0, 1, or 2).
    
    The following is the content to analyze:
    """
)


def label_comments(data, batch_size=10):
    labeled_data = []

    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i + batch_size]
        batch_messages = []

        for item in batch:
            parent_comment = item.get('ParentCommentText', item.get('VideoID'))
            response_comment = item['CommentText']
            user_content = f"Parent Comment: {parent_comment}\nComment: {response_comment}"

            batch_messages.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "original_item": item
            })

        try:
            responses = [
                client.chat.completions.create(
                    messages=msg["messages"],
                    model="gpt-4o",
                    max_tokens=5,
                    temperature=0.1
                )
                for msg in batch_messages
            ]

            for response, msg in zip(responses, batch_messages):
                item = msg["original_item"].copy()
                try:
                    item['Stance_Label'] = int(response.choices[0].message.content.strip())
                    if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens_details'):
                        item['cached_tokens'] = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
                except (ValueError, AttributeError) as e:
                    print(f"Error processing response: {e}")
                    item['Stance_Label'] = None
                labeled_data.append(item)

        except Exception as e:
            print(f"Batch processing error: {e}")
            for msg in batch_messages:
                item = msg["original_item"].copy()
                item['Stance_Label'] = None
                labeled_data.append(item)

    return labeled_data


def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        data = [json.loads(line) for line in file]

    labeled_data = label_comments(data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        for entry in labeled_data:
            file.write(json.dumps(entry) + '\n')


def main():
    input_dir = 'ADD DIRECTORY'
    output_dir = 'ADD DIRECTORY'

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):

            output_filename = filename.replace('MAP_Precomments_', 'Label_')

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, output_filename)

            print(f"Processing {filename}...")
            process_file(input_path, output_path)
            print(f"Completed processing {filename}")


if __name__ == "__main__":
    main()
