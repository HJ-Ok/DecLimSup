for N in 1 5; do
    python inference.py --generate_model meta-llama/Llama-2-7b-chat-hf \
                        --reference_model meta-llama/Llama-2-13b-chat-hf \
                        --wandb_name GSM8k_Llama2-7b_2_13b_alphas_N${N} \
                        --alpha_start 3 --alpha_end -3 --alpha_step -0.25 \
                        --optimize --multi_exec --model_path /workspace/save_models \
                        --benchmark gsm8k --N $N
done