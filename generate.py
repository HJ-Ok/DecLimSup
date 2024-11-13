import gc
import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

import prompt_list.gsm8k
import prompt_list.strategyqa
import wandb
from utils import Checker, find_number, find_rank, load_benchmark


def get_ppl(outputs):
    probabilities = F.softmax(outputs, dim=-1)
    max_prob = probabilities.max()
    ppl = 1 / max_prob

    return ppl


def get_entropy(outputs):
    probabilities = F.softmax(outputs, dim=-1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9))

    return entropy


def get_kl_loss(ori_outputs, ref_outputs):
    log_ori_outputs = torch.log(ori_outputs)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss_value = kl_loss(log_ori_outputs, ref_outputs)
    return kl_loss_value


def get_first_token(model, inputs):
    outputs = model(**inputs)
    return outputs.logits[:, -1, :]


def generate_with_reference(
    args,
    device,
    generate_model,
    generate_tokenizer,
    reference_model,
    reference_tokenizer,
    output_path,
):
    job_name = f"Wreference_{args.wandb_name}"
    wandb.init(project="LLM_decoding", config=args, name=job_name, mode="disabled" if args.test else "online")
    wandb.config.update(args)

    check_answer, input_prompts, answers = load_benchmark(args)
    if args.test:
        input_prompts = input_prompts[:5]
        answers = answers[:5]

    generate_model.generation_config.temperature = None
    generate_model.generation_config.top_p = None
    generate_model.generation_config.top_k = None
    generate_model.eval()

    terminators = [
        generate_tokenizer.eos_token_id,
        generate_tokenizer.convert_tokens_to_ids("Q"),
    ]

    if args.benchmark == "math":
        terminators.append(generate_tokenizer.convert_tokens_to_ids("Question"))

    prompt_df = pd.DataFrame()
    count = 0
    with torch.no_grad():
        for i in tqdm(range(len(input_prompts))):
            input_prompt = input_prompts[i]

            pass_generate = 0
            for tmp in range(args.N):
                if tmp == 0:
                    inputs = generate_tokenizer(input_prompt, return_tensors="pt")
                    ori_inputs = generate_model.prepare_inputs_for_generation(inputs.input_ids.to(device))
                    ori_output = get_first_token(generate_model, ori_inputs)

                    ori_entropy = get_entropy(ori_output)
                    ori_ppl = get_ppl(ori_output)

                    inputs_ref = reference_tokenizer(input_prompt, return_tensors="pt")
                    ref_inputs = reference_model.prepare_inputs_for_generation(inputs_ref.input_ids.to(device))
                    ref_output = get_first_token(reference_model, ref_inputs)

                    ref_entropy = get_entropy(ref_output)
                    ref_ppl = get_ppl(ref_output)

                    ori_output = F.softmax(ori_output, dim=-1)
                    ref_output = F.softmax(ref_output, dim=-1)

                    kl_loss = get_kl_loss(ori_output, ref_output)
                    new_output = ori_output + args.alpha * (ref_output - ori_output)

                    original_argmax = torch.argmax(ori_output)
                    new_argmax = torch.argmax(new_output)
                    ref_argmax = torch.argmax(ref_output)

                    tmp_inputs = torch.cat(
                        (inputs.input_ids.to(device), new_argmax.unsqueeze(0).unsqueeze(0)), dim=1
                    )
                else:
                    ori_inputs = generate_model.prepare_inputs_for_generation(tmp_inputs)
                    ori_output = get_first_token(generate_model, ori_inputs)

                    ref_inputs = reference_model.prepare_inputs_for_generation(tmp_inputs)
                    ref_output = get_first_token(reference_model, ref_inputs)

                    ori_output = F.softmax(ori_output, dim=-1)
                    ref_output = F.softmax(ref_output, dim=-1)

                    new_output = ori_output + args.alpha * (ref_output - ori_output)

                    new_argmax = torch.argmax(new_output)

                    tmp_inputs = torch.cat((tmp_inputs, new_argmax.unsqueeze(0).unsqueeze(0)), dim=1)

                    if new_argmax.item() in terminators:
                        generate_ids = tmp_inputs
                        pass_generate = 1
                        break
            if pass_generate == 0:
                for tmp in range(args.max_new_tokens):
                    ori_inputs = generate_model.prepare_inputs_for_generation(tmp_inputs)
                    tmp_output = get_first_token(generate_model, ori_inputs)
                    tmp_output = F.softmax(tmp_output, dim=-1)

                    tmp_argmax = torch.argmax(tmp_output)
                    tmp_inputs = torch.cat((tmp_inputs, tmp_argmax.unsqueeze(0).unsqueeze(0)), dim=1)

                    if tmp_argmax.item() in terminators:

                        generate_ids = tmp_inputs
                        pass_generate = 1
                        break

                generate_ids = tmp_inputs

            outputs = generate_tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0][len(input_prompt) :]

            original_word = generate_tokenizer.decode(
                original_argmax, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            new_word = generate_tokenizer.decode(
                new_argmax, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            ref_word = reference_tokenizer.decode(
                ref_argmax, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            prompt_df.loc[i, "input_prompt"] = input_prompt + "\n\n"
            prompt_df.loc[i, "result"] = outputs + "\n\n"

            Ground_truth_answer = answers[i]
            is_correct = check_answer(Ground_truth_answer, outputs)
            count += is_correct
            prompt_df.loc[i, "is_correct"] = is_correct
            prompt_df.loc[i, "length"] = int(generate_ids.shape[-1] - inputs.input_ids.shape[-1])
            prompt_df.loc[i, "GT_answer"] = Ground_truth_answer

            prompt_df.loc[i, "is_same"] = (original_argmax == new_argmax).item()
            prompt_df.loc[i, "original_word"] = original_word
            prompt_df.loc[i, "original_id"] = int(original_argmax.item())
            prompt_df.loc[i, "origninal_rank"] = find_rank(ori_output[0], original_argmax)
            prompt_df.loc[i, "ref_word"] = ref_word
            prompt_df.loc[i, "ref_id"] = int(ref_argmax.item())
            prompt_df.loc[i, "ref_rank"] = find_rank(ref_output[0], ref_argmax)
            prompt_df.loc[i, "new_word"] = new_word
            prompt_df.loc[i, "new_id"] = int(new_argmax.item())
            prompt_df.loc[i, "new_rank"] = find_rank(new_output[0], new_argmax)
            prompt_df.loc[i, "first_token_ori_ppl"] = ori_ppl.item()
            prompt_df.loc[i, "first_token_ori_entropy"] = ori_entropy.item()
            prompt_df.loc[i, "first_token_ref_ppl"] = ref_ppl.item()
            prompt_df.loc[i, "first_token_ref_entropy"] = ref_entropy.item()
            prompt_df.loc[i, "first_token_kl_loss"] = kl_loss.item()

            wandb.log({"count": i, "correct_count": count, "tmp_acc": count / (i + 1)})

        prompt_df.to_csv(output_path, index=False)
        acc = count / len(input_prompts)
        wandb.log({"accuracy": acc})
        wandb.finish()
    return acc


def generate(args, device, generate_model, generate_tokenizer, output_path):
    job_name = f"Greedy_{args.wandb_name}"
    wandb.init(project="LLM_decoding", config=args, name=job_name, mode="disabled" if args.test else "online")
    wandb.config.update(args)

    check_answer, input_prompts, answers = load_benchmark(args)
    if args.test:
        input_prompts = input_prompts[:5]
        answers = answers[:5]

    generate_model.generation_config.temperature = None
    generate_model.generation_config.top_p = None

    terminators = [
        generate_tokenizer.eos_token_id,
        generate_tokenizer.convert_tokens_to_ids("Q"),
    ]

    if args.benchmark == "math":
        terminators.append(generate_tokenizer.convert_tokens_to_ids("Question"))
    prompt_df = pd.DataFrame()
    count = 0
    with torch.no_grad():
        for i in tqdm(range(len(input_prompts))):
            input_prompt = input_prompts[i]
            inputs = generate_tokenizer(input_prompt, return_tensors="pt")

            tmp_inputs = inputs.input_ids.to(device)
            for tmp in range(args.max_new_tokens):
                ori_inputs = generate_model.prepare_inputs_for_generation(tmp_inputs)
                tmp_output = get_first_token(generate_model, ori_inputs)
                tmp_output = F.softmax(tmp_output, dim=-1)

                tmp_argmax = torch.argmax(tmp_output)
                tmp_inputs = torch.cat((tmp_inputs, tmp_argmax.unsqueeze(0).unsqueeze(0)), dim=1)

                if tmp_argmax.item() in terminators:
                    generate_ids = tmp_inputs
                    pass_generate = 1
                    break

            generate_ids = tmp_inputs
            outputs = generate_tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0][len(input_prompt) :]

            prompt_df.loc[i, "input_prompt"] = input_prompt + "\n\n"
            prompt_df.loc[i, "result"] = outputs + "\n\n"

            Ground_truth_answer = answers[i]
            is_correct = check_answer(Ground_truth_answer, outputs)
            count += is_correct
            prompt_df.loc[i, "is_correct"] = is_correct
            prompt_df.loc[i, "length"] = int(generate_ids.shape[-1] - inputs.input_ids.shape[-1])
            prompt_df.loc[i, "GT_answer"] = Ground_truth_answer
            wandb.log({"count": i, "correct_count": count, "tmp_acc": count / (i + 1)})
        prompt_df.to_csv(output_path, index=False)
        acc = count / len(input_prompts)
        wandb.log({"accuracy": acc})
        wandb.finish()
    return acc
