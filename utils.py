import json
import re

import numpy as np
import torch
from datasets import load_dataset

import prompt_list.arc_c
import prompt_list.gsm8k
import prompt_list.math
import prompt_list.multiarith
import prompt_list.strategyqa


def load_benchmark(args):
    if args.benchmark == "gsm8k":
        prompts = prompt_list.gsm8k.prompts_gsm8k()
        prompt_template = prompts.get_PREAMBLE() + "\n\n" + prompts.get_SHORT_PROMPT() + "\n" + prompts.get_TEMPLATE()
        check_answer = Checker.check_answer_gsm8k
        dataset = load_dataset("gsm8k", "main")
        questions = dataset["test"]["question"]
        answers = dataset["test"]["answer"]
        if args.train_inference:
            questions = dataset["train"]["question"]
            answers = dataset["train"]["answer"]

        input_prompts = []
        for i in range(len(questions)):
            in_prompt = prompt_template.format(question=questions[i])
            input_prompts.append(in_prompt)

    elif args.benchmark == "strategyqa":
        prompts = prompt_list.strategyqa.prompts_strategyqa()
        prompt_template = prompts.get_PROMPT() + "\n" + prompts.get_TEMPLATE()
        check_answer = Checker.check_answer_strategyqa
        dataset = load_dataset("ChilleD/StrategyQA")
        questions = dataset["test"]["question"]
        answers = dataset["test"]["answer"]

        input_prompts = []
        for i in range(len(questions)):
            in_prompt = prompt_template.format(question=questions[i])
            input_prompts.append(in_prompt)

    elif args.benchmark == "multiarith":
        prompts = prompt_list.multiarith.prompts_multiarith()
        prompt_template = prompts.get_PREAMBLE() + "\n\n" + prompts.get_PROMPT() + "\n" + prompts.get_TEMPLATE()
        check_answer = Checker.check_answer_multiarith
        dataset = load_dataset("ChilleD/MultiArith")

        questions = dataset["train"]["question"]
        answers = dataset["train"]["final_ans"]

        input_prompts = []
        for i in range(len(questions)):
            in_prompt = prompt_template.format(question=questions[i])
            input_prompts.append(in_prompt)

    elif args.benchmark == "math":
        prompts = prompt_list.math.prompts_math()
        prompt_template = prompts.get_PROMPT() + "\n" + prompts.get_TEMPLATE()
        check_answer = Checker.check_answer_math
        dataset = load_dataset("lighteval/MATH", "all")

        input_prompts = []
        answers = []
        for i in range(len(dataset["test"])):
            question = dataset["test"]["problem"][i]
            in_prompt = prompt_template + question + "\n" + "Answer: Let's think step by step.\n"

            input_prompts.append(in_prompt)
            answers.append(dataset["test"]["solution"][i])

    elif args.benchmark == "arc_c":
        task_data = load_dataset("allenai/ai2_arc", "ARC-Challenge")

        prompts = prompt_list.arc_c.prompts_arc_c().get_PROMPT()
        # prompt_template = prompts.get_PROMPT() + "\n" + prompts.get_TEMPLATE()

        input_prompts = []
        answers = []
        for q_ in task_data["test"]:
            q = "Q: " + q_["question"] + "\n"
            for l, t in zip(q_["choices"]["label"], q_["choices"]["text"]):
                q += "(" + l + ") " + t + " "
            q += "\nA: Let's think step by step.\n"

            prompt_q = prompts + "\n\n" + q
            input_prompts.append(prompt_q)
            answers.append(q_["answerKey"])

        check_answer = Checker.check_answer_mmlu
        print("dataset size: ", len(input_prompts))

    elif args.benchmark == "arc_e":
        task_data = load_dataset("allenai/ai2_arc", "ARC-Easy")
        prompts = prompt_list.arc_c.prompts_arc_c().get_PROMPT()
        input_prompts = []
        answers = []
        for q_ in task_data["test"]:
            q = "Q: " + q_["question"] + "\n"
            for l, t in zip(q_["choices"]["label"], q_["choices"]["text"]):
                q += "(" + l + ") " + t + " "
            q += "\nA: Let's think step by step.\n"

            prompt_q = prompts + "\n\n" + q
            input_prompts.append(prompt_q)
            answers.append(q_["answerKey"])

        check_answer = Checker.check_answer_mmlu
        print("dataset size: ", len(input_prompts))

    elif args.benchmark == "svamp":
        task_data = load_dataset("ChilleD/SVAMP")
        prompts = (
            prompt_list.gsm8k.prompts_gsm8k().get_PREAMBLE() + "\n\n" + prompt_list.gsm8k.prompts_gsm8k().get_PROMPT()
        )
        input_prompts = []
        answers = []
        for k in ["train", "test"]:
            for q_ in task_data[k]:
                q = "Q: " + q_["Body"] + " " + q_["Question"] + "\n"
                q += "A: "

                prompt_q = prompts + "\n\n" + q
                input_prompts.append(prompt_q)
                answers.append(q_["Answer"])

        check_answer = Checker.check_answer_svamp
        print("dataset size: ", len(input_prompts))

    return check_answer, input_prompts, answers


def find_rank(tensor, new_id):
    sorted_tensor, indices = torch.sort(tensor, descending=True)
    index_of_new_id = (indices == new_id).nonzero().item()
    return int(index_of_new_id)


def find_numbers(x: str) -> list[str]:
    numbers = re.compile(
        r"-?[\d,]*\.?\d+",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers


def find_number(x: str, answer_delimiter: str = "The answer is") -> str:
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ""


def maybe_remove_comma(x: str) -> str:
    return x.replace(",", "")


def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]


PUNCTUATION_SET_TO_EXCLUDE = set("".join(["‘", "’", "´", "`", ".", ",", "-", '"']))


def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace("_", " ").lower()
        alias = "".join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else " " for c in alias)
        answers.append(" ".join(alias.split()).strip())
    return set(answers)


def find_answer(s):
    assert "boxed" in s
    ans = s.split("boxed")[-1]
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


class Checker:
    def check_answer_gsm8k(Ground_truth_answer, Generate_answer):
        all_responses = Generate_answer.split("Q:")[0]
        short_responses = maybe_remove_comma(find_number(all_responses))

        try:
            correct = float(maybe_remove_comma(find_number(Ground_truth_answer))) == float(short_responses)
        except:
            correct = maybe_remove_comma(find_number(Ground_truth_answer)) == maybe_remove_comma(
                find_number(short_responses)
            )
        return correct

    def check_answer_strategyqa(Ground_truth_answer, Generate_answer):

        answer_delimiter = "answer is"

        Generate_answer = Generate_answer.split("Q:")[0]
        if answer_delimiter in Generate_answer:
            Generate_answer = Generate_answer.split(answer_delimiter)[-1]

        if "yes" in Generate_answer:
            response = True
        elif "no" in Generate_answer:
            response = False
        else:
            return False

        return Ground_truth_answer == response

    def check_answer_multiarith(Ground_truth_answer, Generate_answer):

        all_responses = Generate_answer.split("Q:")[0]
        short_responses = maybe_remove_comma(find_number(all_responses))

        try:
            correct = float(maybe_remove_comma(find_number(Ground_truth_answer))) == float(short_responses)
        except:
            correct = maybe_remove_comma(find_number(Ground_truth_answer)) == maybe_remove_comma(
                find_number(short_responses)
            )
        return correct

    def check_answer_math(ans_str, pred_str):

        pred_str = pred_str.split("Question")[0]

        if "The answer is " in pred_str:
            pred = pred_str.split("The answer is ")[1].strip()
        else:
            pattern = "\d*\.?\d+"
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                # print(pred_str)
                pred = pred[-1]
            else:
                pred = ""

        gold = find_answer(ans_str)

        return pred == gold

    def check_answer_svamp(Ground_truth_answer, Generate_answer):
        all_responses = Generate_answer.split("Q:")[0]
        short_responses = maybe_remove_comma(find_number(all_responses))

        try:
            correct = Ground_truth_answer == float(short_responses)
        except:
            correct = Ground_truth_answer == float(maybe_remove_comma(find_number(short_responses)))
        return correct
