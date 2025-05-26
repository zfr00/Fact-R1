import json
import re
from mathruler.grader import extract_boxed_content, grade_answer
from rouge import Rouge
from difflib import SequenceMatcher
from venus_api_base.http_client import HttpClient
from venus_api_base.config import Config
import Levenshtein

# API credentials for fakevv task
secret_id = ''
secret_key = ''
client = HttpClient(secret_id=secret_id, secret_key=secret_key, config=Config(read_timeout=500))
domain = 'http://v2.open.venus.oa.com'
header = {'Content-Type': 'application/json'}

def check_entity_hit(prediction, entity):
    prompt = f"""
    You are an AI assistant to help me evaluate whether the reasoning process in the prediction correctly identifies the labeled fake entity.

    Task Description:
    Determine if the following prediction mentions the fake entity: "{entity}".
    Prediction: "{prediction}"
    Output "yes" if the prediction reasoning process mentions and describes the entity; otherwise, output "no".

    Example Case:
    Fake Entity: "French"
    Prediction: "The news title, 'Storm Bert: drone footage shows extent of flooding at French holiday park,' suggests that the flooding occurred at a French holiday park, which is inconsistent with the video description indicating that the flooding occurred at an English holiday park. Thus, the news title does not accurately reflect the content of the video description, as it incorrectly states the location of the flooding. In conclusion, the event discussed in the news title does not match the specific location of the flooding depicted in the video description."
    Output: yes
    """
    
    body = {
        "appGroupId": 3808,
        'model': 'gpt-4o-mini',
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 4096
    }
    
    ret = client.post(f'{domain}/chat/single', header=header, body=json.dumps(body))
    if ret['code'] != 0 or ret['data']['status'] != 2:
        print('Failed: ', ret.get('message', ''), ret.get('traceId', ''))
        return 'no'
    else:
        return ret['data']['response'].strip().lower()

def entity_reward(predict_str: str, entity: str) -> float:
    hit = check_entity_hit(predict_str, entity)
    return 1.0 if "yes" in hit else 0.0

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r'(?:<think>(.*?)</think>\s*)?<answer>(.*?)</answer>', re.DOTALL)
    match = re.search(pattern, predict_str)
    return 1.0 if match else 0.0

def acc_reward(predict_str: str, ground_truth: str, task: str) -> float:
    reward = 0.0

    # Extract answer from content if it has think/answer tags
    content_match = re.search(r'<answer>(.*?)</answer>', predict_str)
    prediction = content_match.group(1).strip() if content_match else predict_str.strip()

    sol_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
    ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()

    try:
        if task in ['fakesv', 'fakevv','fakett']:
            if ground_truth.strip().lower() in predict_str.lower():
                return 1.0
            else:
                return 0.0

        elif task == 'caption':
            rouge = Rouge()
            scores = rouge.get_scores(prediction, ground_truth)
            reward = scores[0]['rouge-l']['f']

        elif task == 'ocr':
            edit_distance = Levenshtein.distance(prediction.lower(), ground_truth.lower())
            max_length = max(len(prediction), len(ground_truth))
            reward = 1 - edit_distance / max_length

    except Exception as e:
        print(e)

    return reward

def reasoning_steps_reward(predict_str: str) -> float:
    reasoning_keywords = [
        "however", "in conclusion", "therefore", "thus", "as a result",
        "consequently", "first", "second", "finally", "in summary",
        "on the other hand", "moreover", "furthermore", "additionally"
    ]
    
    lower_predict_str = predict_str.lower()
    keyword_count = sum(lower_predict_str.count(keyword) for keyword in reasoning_keywords)
    max_keywords = 2
    score = min(keyword_count / max_keywords, 1.0)
    
    return score

def multitask_compute_score(predict_str: str, ground_truth: str, task: str) -> float:
    if task == 'fakevv':
        entity = ground_truth.split('\n')[-1].strip()
        if not entity:
            entity = ground_truth
        # breakpoint()
        if "yes" in ground_truth.lower():
            ground_truth = "yes"
            entity_score = entity_reward(predict_str, entity)
            return 0.7 * acc_reward(predict_str, ground_truth, task) + 0.1 * format_reward(predict_str) + 0.1 * reasoning_steps_reward(predict_str) + 0.1 * entity_score
        else:
            ground_truth = "no"
            return 0.8 * acc_reward(predict_str, ground_truth, task) + 0.1 * format_reward(predict_str) + 0.1 * reasoning_steps_reward(predict_str)
    else:
        return 0.9 * acc_reward(predict_str, ground_truth, task) + 0.1 * format_reward(predict_str)

# Example usage
if __name__ == "__main__":
    tasks = ['fakevv']
    predict_str = "<think>Some reasoning</think><answer>Yes</answer>"
    ground_truth = "Yes\npeaceful rally"

    for task in tasks:
        score = multitask_compute_score(predict_str, ground_truth, task)
        print(f"Computed score for task '{task}': {score}")
