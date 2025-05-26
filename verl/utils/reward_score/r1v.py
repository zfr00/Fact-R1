import re

from mathruler.grader import grade_answer


def r1v_format_reward(predict_str: str) -> float:

    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, predict_str, re.DOTALL)
    return 1 if matches else 0.0
    #pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    # format_match = re.fullmatch(pattern, predict_str)
    # return 1.0 if format_match else 0.0


def r1v_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        ground_truth = ground_truth.strip()
        if '<answer>' in ground_truth and '</answer>' in ground_truth:
            content_match = re.search(r"<answer>(.*?)</answer>", ground_truth)
            ground_truth = content_match.group(1).strip() if content_match else ground_truth

        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if grade_answer(given_answer, ground_truth):
            return 1.0
    except Exception as e:
        print(e)
        pass

    return 0.0


def r1v_compute_score(predict_str: str, ground_truth: str, task: str) -> float:
    return 0.9 * r1v_accuracy_reward(predict_str, ground_truth) + 0.1 * r1v_format_reward(predict_str)

