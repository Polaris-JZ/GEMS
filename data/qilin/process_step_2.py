import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent
ORI_DATA_DIR = DATA_DIR / "ori_data"
SEQ_DATA_DIR = DATA_DIR / "seq_data"
NUM_NEGATIVES = 99


def load_frame(name):
    with open(ORI_DATA_DIR / name, "rb") as file:
        return pickle.load(file)


def save_frame(frame, name):
    with open(SEQ_DATA_DIR / name, "wb") as file:
        pickle.dump(frame.to_dict(orient="index"), file)


def collect_all_items(datasets):
    item_set = set()
    for data in datasets:
        for _, row in data.iterrows():
            targets = row["gt_note_idx"] if isinstance(row["gt_note_idx"], list) else [row["gt_note_idx"]]
            history = row["history"] if isinstance(row["history"], list) else []
            negatives = row["neg_note_idx"] if isinstance(row["neg_note_idx"], list) else []
            item_set.update(targets)
            item_set.update(history)
            item_set.update(negatives)
    return item_set


def sample_negative_items(user_history, target_items, item_corpus):
    excluded_items = set(user_history + target_items)
    candidates = [item for item in item_corpus if item not in excluded_items]
    if len(candidates) <= NUM_NEGATIVES:
        return candidates
    return random.sample(candidates, NUM_NEGATIVES)


def normalize_columns(data):
    return data.rename(
        columns={
            "gt_note_idx": "target_item",
            "history": "item_list",
            "user_idx": "user_id",
        }
    ).reset_index(drop=True)


def expand_training_data(data, item_corpus, description):
    expanded_data = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc=description):
        for target_item in row["gt_note_idx"]:
            new_row = row.copy()
            new_row["gt_note_idx"] = target_item
            history = row["history"] if isinstance(row["history"], list) else []
            new_row["neg_note_idx"] = sample_negative_items(
                history, [target_item], item_corpus
            )
            expanded_data.append(new_row)
    return normalize_columns(pd.DataFrame(expanded_data))


def resample_test_data(data, item_corpus, description):
    data = data.copy()
    for index, row in tqdm(data.iterrows(), total=len(data), desc=description):
        history = row["history"] if isinstance(row["history"], list) else []
        targets = row["gt_note_idx"] if isinstance(row["gt_note_idx"], list) else [row["gt_note_idx"]]
        data.at[index, "neg_note_idx"] = sample_negative_items(
            history, targets, item_corpus
        )
    return normalize_columns(data)


def save_split(rec_data, src_data, split):
    combined = pd.concat([rec_data, src_data], ignore_index=True)
    combined.sort_values(by=["user_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    save_frame(combined, f"{split}.pkl")
    save_frame(rec_data, f"{split}_rec.pkl")
    save_frame(src_data, f"{split}_src.pkl")


def validate_inputs():
    required_files = [
        "rec_train.pkl",
        "src_train.pkl",
        "rec_valid.pkl",
        "src_valid.pkl",
        "rec_test.pkl",
        "src_test.pkl",
    ]
    missing_files = [name for name in required_files if not (ORI_DATA_DIR / name).is_file()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing intermediate data in {ORI_DATA_DIR}: {missing}. "
            "Run process_step_1.py first."
        )


def main():
    random.seed(42)
    np.random.seed(42)
    validate_inputs()
    SEQ_DATA_DIR.mkdir(parents=True, exist_ok=True)

    rec_train = load_frame("rec_train.pkl")
    src_train = load_frame("src_train.pkl")
    item_corpus = collect_all_items([rec_train, src_train])
    print(f"训练集 item corpus 大小: {len(item_corpus)}")

    rec_train = expand_training_data(rec_train, item_corpus, "处理 rec_train")
    src_train = expand_training_data(src_train, item_corpus, "处理 src_train")
    save_split(rec_train, src_train, "train")

    rec_valid = load_frame("rec_valid.pkl")
    src_valid = load_frame("src_valid.pkl")
    item_corpus.update(collect_all_items([rec_valid, src_valid]))
    print(f"加入验证集后的 item corpus 大小: {len(item_corpus)}")

    rec_valid = expand_training_data(rec_valid, item_corpus, "处理 rec_valid")
    src_valid = expand_training_data(src_valid, item_corpus, "处理 src_valid")
    save_split(rec_valid, src_valid, "valid")

    rec_test = load_frame("rec_test.pkl")
    src_test = load_frame("src_test.pkl")
    item_corpus.update(collect_all_items([rec_test, src_test]))
    print(f"最终 item corpus 大小: {len(item_corpus)}")

    rec_test = resample_test_data(rec_test, item_corpus, "处理 rec_test")
    src_test = resample_test_data(src_test, item_corpus, "处理 src_test")
    save_split(rec_test, src_test, "test")

    required_outputs = [
        "train_rec.pkl",
        "train_src.pkl",
        "valid_rec.pkl",
        "valid_src.pkl",
        "test_rec.pkl",
        "test_src.pkl",
    ]
    missing_outputs = [name for name in required_outputs if not (SEQ_DATA_DIR / name).is_file()]
    if missing_outputs:
        raise RuntimeError(f"数据生成不完整: {', '.join(missing_outputs)}")

    print(f"训练数据已生成到: {SEQ_DATA_DIR}")


if __name__ == "__main__":
    main()
