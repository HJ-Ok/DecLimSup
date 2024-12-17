import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from generate import generate, generate_with_reference
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hpu

torch._dynamo.config.suppress_errors = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="gsm8k", required=False)
    parser.add_argument("--N", type=int, default=1, required=False)
    parser.add_argument("--alpha", type=float, default=0.0, required=False)
    parser.add_argument("--generate_solo", type=bool, default=0, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=512, required=False)
    parser.add_argument("--reference_model", type=str, default="Qwen/Qwen1.5-1.8B-Chat", required=False)
    parser.add_argument("--generate_model", type=str, default="Qwen/Qwen1.5-0.5B-Chat", required=False)
    parser.add_argument("--wandb_name", type=str, default="Qwen7b_2_1.8b_alpha3_N1", required=False)
    parser.add_argument("--model_path", type=str, default="./save_models", required=False)
    parser.add_argument("--output_path", type=str, default="./outputs", required=False)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--multi_exec", action="store_true")
    parser.add_argument("--train_inference", action="store_true")
    parser.add_argument("--alpha_start", type=float, default=3.0, required=False)
    parser.add_argument("--alpha_end", type=float, default=-3.0, required=False)
    parser.add_argument("--alpha_step", type=float, default=-0.25, required=False)
    parser.add_argument("--ppl_threshold", type=float, default=1.8, required=False)
    parser.add_argument("--all_task", action="store_true")
    parser.add_argument("--task", type=str, default="", required=False)
    parser.add_argument("--ex_name", type=str, default="", required=False)
    parser.add_argument("--huggingface_token", type=str, default="", required=False)
    parser.add_argument("--use_hpu", action="store_true")
    return parser.parse_args()


def inference(args):
    if args.use_hpu:
        device = torch.device("hpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generate_tokenizer = AutoTokenizer.from_pretrained(
        args.generate_model, token=args.huggingface_token, cache_dir=args.model_path
    )
    if args.optimize:
        generate_model = AutoModelForCausalLM.from_pretrained(
            args.generate_model,
            token=args.huggingface_token,
            cache_dir=args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        generate_model = AutoModelForCausalLM.from_pretrained(
            args.generate_model, token=args.huggingface_token, cache_dir=args.model_path
        )
        generate_model = generate_model.to(device)

    generate_model.eval()

    if args.optimize:
        generate_model = torch.compile(generate_model)

    generate_model.generation_config.temperature = None
    generate_model.generation_config.top_p = None
    generate_model.generation_config.top_k = None

    generation_model_name = args.generate_model.split("/")[-1]

    task_list = [args.task]

    if args.generate_solo == 1:
        for task in task_list:
            args.task = task

            if task != "":
                dir_path = os.path.join(args.output_path, args.benchmark, task, generation_model_name)
            else:
                dir_path = os.path.join(args.output_path, args.benchmark, generation_model_name)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            output_path = os.path.join(dir_path, f"greedy_generation{args.ex_name}.csv")

            acc = generate(args, device, generate_model, generate_tokenizer, output_path)
            print(acc)
    else:
        reference_tokenizer = AutoTokenizer.from_pretrained(
            args.reference_model, token=args.huggingface_token, cache_dir=args.model_path
        )

        if args.optimize:
            reference_model = AutoModelForCausalLM.from_pretrained(
                args.reference_model,
                token=args.huggingface_token,
                cache_dir=args.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            reference_model = AutoModelForCausalLM.from_pretrained(
                args.reference_model, token=args.huggingface_token, cache_dir=args.model_path
            )
            reference_model = reference_model.to(device)

        reference_model.eval()

        if args.optimize:
            reference_model = torch.compile(reference_model)

        reference_model.generation_config.temperature = None
        reference_model.generation_config.top_p = None
        reference_model.generation_config.top_k = None

        reference_model_name = args.reference_model.split("/")[-1]

        if args.multi_exec:
            values = np.arange(args.alpha_start, args.alpha_end + args.alpha_step, args.alpha_step)
        else:
            values = [args.alpha]

        for task in task_list:
            args.task = task

            if task != "":
                dir_path = os.path.join(
                    args.output_path,
                    args.benchmark,
                    task,
                    reference_model_name + "_to_" + generation_model_name + args.ex_name,
                )
            else:
                dir_path = os.path.join(
                    args.output_path,
                    args.benchmark,
                    reference_model_name + "_to_" + generation_model_name + args.ex_name,
                )

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for a in values:
                args.alpha = a
                print("--------------------------------")
                print(args)
                print(f"Running with alpha={a}")
                output_path = os.path.join(
                    dir_path,
                    f"alpha{args.alpha}_with_{args.N}tokens.csv",
                )
                acc = generate_with_reference(
                    args, device, generate_model, generate_tokenizer, reference_model, reference_tokenizer, output_path
                )
                print("acc:", acc)


if __name__ == "__main__":
    args = parse_args()
    inference(args)
