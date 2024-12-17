import os
import pickle

import numpy as np
import pandas as pd
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


def tensors_to_df(tensor_list, prefix):
    flat_data = [t.numpy().flatten() for t_list in tensor_list for t in t_list]
    df = pd.DataFrame(flat_data)
    df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]
    return df


with open("/workspace/TUNA_LLM/outputs/qwen_1dot8b_train.pkl", "rb") as f:
    train_7b_logit = pickle.load(f)
with open("/workspace/TUNA_LLM/outputs/qwen_1dot8b.pkl", "rb") as f:
    test_7b_logit = pickle.load(f)

with open("/workspace/TUNA_LLM/outputs/qwen_4b_train.pkl", "rb") as f:
    train_13b_logit = pickle.load(f)
with open("/workspace/TUNA_LLM/outputs/qwen_4b.pkl", "rb") as f:
    test_13b_logit = pickle.load(f)


df = pd.read_csv("/workspace/TUNA_bert/Qwen_train.csv")
test_df = pd.read_csv("/workspace/TUNA_bert/Qwen_test.csv")

train_df = df[df.kfold != 0]
valid_df = df[df.kfold == 0]
test_df = pd.read_csv("/workspace/TUNA_bert/Qwen_test.csv")

valid_7b_logit = [train_7b_logit[idx] for idx in valid_df.index.tolist()]
valid_13b_logit = [train_13b_logit[idx] for idx in valid_df.index.tolist()]
train_7b_logit = [train_7b_logit[idx] for idx in train_df.index.tolist()]
train_13b_logit = [train_13b_logit[idx] for idx in train_df.index.tolist()]

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

df_train_7b_logit = tensors_to_df(train_7b_logit, "7b")
df_train_13b_logit = tensors_to_df(train_13b_logit, "13b")

df_valid_7b_logit = tensors_to_df(valid_7b_logit, "7b")
df_valid_13b_logit = tensors_to_df(valid_13b_logit, "13b")

df_test_7b_logit = tensors_to_df(test_7b_logit, "7b")
df_test_13b_logit = tensors_to_df(test_13b_logit, "13b")

train_df2 = pd.concat([train_df, df_train_7b_logit, df_train_13b_logit], axis=1)
valid_df2 = pd.concat([valid_df, df_valid_7b_logit, df_valid_13b_logit], axis=1)


target_columns = [
    "alpha3.0_is_correct",
    "alpha2.75_is_correct",
    "alpha2.5_is_correct",
    "alpha2.25_is_correct",
    "alpha2.0_is_correct",
    "alpha1.75_is_correct",
    "alpha1.5_is_correct",
    "alpha1.25_is_correct",
    "alpha1.0_is_correct",
    "alpha0.75_is_correct",
    "alpha0.5_is_correct",
    "alpha0.25_is_correct",
    "alpha0.0_is_correct",
    "alpha-0.25_is_correct",
    "alpha-0.5_is_correct",
    "alpha-0.75_is_correct",
    "alpha-1.0_is_correct",
]

required_columns = (
    [
        "original_id",
        "ref_id",
        "new_id",
        "first_token_ori_ppl",
        "first_token_ori_entropy",
        "first_token_ref_ppl",
        "first_token_ref_entropy",
        "first_token_kl_loss",
    ]
    + list(df_train_7b_logit.columns)
    + list(df_train_13b_logit.columns)
)


train_features = train_df2[required_columns]
valid_features = valid_df2[required_columns]

train_labels = train_df2[target_columns]
valid_labels = valid_df2[target_columns]

test_df2 = pd.concat([test_df, df_test_7b_logit, df_test_13b_logit], axis=1)
test_features = test_df2[required_columns]
test_labels = test_df2[target_columns]

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

model = XGBClassifier(objective="binary:logistic", n_estimators=2000, tree_method="gpu_hist", eval_metric="error")

model.fit(
    train_features, train_labels, eval_set=[(valid_features, valid_labels)], early_stopping_rounds=50, verbose=True
)


def calc_f1_acc(pred, df):
    values = np.arange(3, -1 - 0.25, -0.25)
    correct_count = 0
    for idx, tmp in enumerate(pred):
        tmp_alpha = values[tmp]
        column_name = f"alpha{tmp_alpha}_is_correct"
        if df.at[idx, column_name] == 1:
            correct_count += 1
    return correct_count / len(df)


import numpy as np


def get_highest_probability_indices(pred_probs):
    return np.argmax(pred_probs, axis=1)


pred_probs = model.predict_proba(test_features)
highest_indices = get_highest_probability_indices(pred_probs)


pred_probs = model.predict_proba(test_features)
highest_indices = get_highest_probability_indices(pred_probs)
calc_f1_acc(highest_indices, test_df)
