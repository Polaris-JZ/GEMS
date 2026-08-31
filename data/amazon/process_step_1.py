import gzip
import pickle
import random
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent
ORI_DATA_DIR = DATA_DIR / "ori_data"
SEQ_DATA_DIR = DATA_DIR / "seq_data"
ORIG_DATA_DIR = DATA_DIR / "orig_data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
MAX_HISTORY_LENGTH = 20
NUM_NEGATIVES = 99


def load_frame(name):
    with open(ORI_DATA_DIR / name, "rb") as file:
        data = pickle.load(file)
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame.from_dict(data, orient="index")


def save_records(data, name):
    with open(SEQ_DATA_DIR / name, "wb") as file:
        pickle.dump(data.to_dict(orient="index"), file)


def prepare_events(rec_data, src_data):
    rec_data = rec_data.copy().rename(columns={"keyword": "query"})
    src_data = src_data.copy().rename(columns={"keyword": "query"})
    rec_data["event_type"] = "rec"
    src_data["event_type"] = "src"
    return pd.concat([rec_data, src_data], ignore_index=True)


def group_events(events):
    events = events.sort_values(by=["user_id", "ts", "all_his"]).reset_index(drop=True)
    return events.groupby("user_id").agg(
        item_id=("item_id", list),
        ts=("ts", list),
        query=("query", list),
        event_type=("event_type", list),
    ).reset_index()


def make_example(user_id, items, queries, event_types, target_index):
    history_end = target_index
    if target_index > 0 and items[target_index] == items[target_index - 1]:
        history_end -= 1
    if history_end <= 4:
        return None

    item_list = items[:history_end]
    target_item = items[target_index]
    example_types = event_types[:history_end] + [event_types[target_index]]
    example_queries = queries[:history_end] + [queries[target_index]]
    return {
        "user_id": user_id,
        "item_list": item_list,
        "target_item": target_item,
        "query": example_queries[-1],
        "his_task": example_types[:-1],
        "his_query": example_queries[:-1],
        "length": len(item_list),
        "task": example_types[-1],
    }


def build_train_examples(grouped_train):
    examples = []
    for _, row in tqdm(grouped_train.iterrows(), total=len(grouped_train), desc="Build train"):
        for target_index in range(1, len(row["item_id"])):
            example = make_example(
                row["user_id"],
                row["item_id"],
                row["query"],
                row["event_type"],
                target_index,
            )
            if example is not None:
                examples.append(example)

    train = pd.DataFrame(examples)
    if train.empty:
        return train

    train = train.sort_values(by=["user_id", "length"], ascending=False)
    last_examples = train.groupby("user_id").head(1)
    other_examples = train[~train.index.isin(last_examples.index)]
    if not other_examples.empty:
        other_examples = other_examples.groupby("user_id", group_keys=False).sample(
            frac=0.3, random_state=42
        )
    train = pd.concat([other_examples, last_examples], ignore_index=True)
    return truncate_histories(train).sort_values(by="user_id").reset_index(drop=True)


def build_eval_examples(grouped_split, prior_events, description):
    examples = []
    for _, row in tqdm(grouped_split.iterrows(), total=len(grouped_split), desc=description):
        prior = prior_events[row["user_id"]]
        items = prior["item_id"] + row["item_id"]
        queries = prior["query"] + row["query"]
        event_types = prior["event_type"] + row["event_type"]
        prior_length = len(prior["item_id"])
        for offset in range(len(row["item_id"])):
            example = make_example(
                row["user_id"], items, queries, event_types, prior_length + offset
            )
            if example is not None:
                examples.append(example)
    return truncate_histories(pd.DataFrame(examples))


def truncate_histories(data):
    if data.empty:
        return data
    data = data.copy()
    for column in ("item_list", "his_task", "his_query"):
        data[column] = data[column].apply(lambda values: values[-MAX_HISTORY_LENGTH:])
    data["length"] = data["item_list"].apply(len)
    return data


def split_and_save(data, split):
    save_records(data, f"{split}.pkl")
    for task in ("rec", "src"):
        task_data = data[data["task"] == task].reset_index(drop=True)
        save_records(task_data, f"{split}_{task}.pkl")


def add_test_negatives(test_data, item_set):
    random.seed(42)
    test_data = test_data.copy()
    test_data["neg_note_idx"] = [None] * len(test_data)
    for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Sample negatives"):
        existing = set(row["item_list"] + [row["target_item"]])
        candidates = list(item_set - existing)
        test_data.at[index, "neg_note_idx"] = random.sample(
            candidates, min(NUM_NEGATIVES, len(candidates))
        )
    return test_data


def write_item_text(item_set):
    query_path = ORIG_DATA_DIR / "query_text.txt.gz"
    qrels_path = ORIG_DATA_DIR / "train.qrels.gz"
    metadata_path = RAW_DATA_DIR / "item_meta.pkl"
    if not all(path.is_file() for path in (query_path, qrels_path, metadata_path)):
        print("Skip item_plain_text.txt: optional query/qrels/item metadata files are missing.")
        return

    with gzip.open(query_path) as file:
        query_text = [line.strip().decode() for line in file]
    with open(metadata_path, "rb") as file:
        item_meta = pickle.load(file)

    filtered_meta = {key: value for key, value in item_meta.items() if key in item_set}
    asin_to_id = {value["asin"]: key for key, value in filtered_meta.items()}
    with gzip.open(qrels_path) as file:
        for line in file:
            user_query_id, _, asin, _ = line.strip().decode().split(" ")
            _, query_id = user_query_id.split("_")
            if asin in asin_to_id:
                filtered_meta[asin_to_id[asin]].setdefault("query", []).append(
                    query_text[int(query_id)]
                )

    with open(SEQ_DATA_DIR / "item_plain_text.txt", "w", encoding="utf-8") as file:
        for item_id, value in filtered_meta.items():
            categories = ", ".join(
                category for group in value.get("categories", []) for category in group
            )
            parts = [
                f"title: {value['title']}" if isinstance(value.get("title"), str) else "",
                f"description: {value['description']}" if isinstance(value.get("description"), str) else "",
                f"categories: {categories}" if categories else "",
                f"query: {', '.join(value.get('query', []))}" if value.get("query") else "",
                f"brand: {value['brand']}" if isinstance(value.get("brand"), str) else "",
            ]
            text = re.sub(r"\n", "", " ".join(part for part in parts if part).strip())
            file.write(f"{item_id + 1} {text}\n")


def validate_inputs():
    required = [
        "rec_train.pkl",
        "src_train.pkl",
        "rec_val.pkl",
        "src_val.pkl",
        "rec_test.pkl",
        "src_test.pkl",
    ]
    missing = [name for name in required if not (ORI_DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Amazon intermediate data in {ORI_DATA_DIR}: {', '.join(missing)}"
        )


def main():
    validate_inputs()
    SEQ_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_events = group_events(
        prepare_events(load_frame("rec_train.pkl"), load_frame("src_train.pkl"))
    )
    valid_events = group_events(
        prepare_events(load_frame("rec_val.pkl"), load_frame("src_val.pkl"))
    )
    test_events = group_events(
        prepare_events(load_frame("rec_test.pkl"), load_frame("src_test.pkl"))
    )

    train_users = set(train_events["user_id"])
    valid_events = valid_events[valid_events["user_id"].isin(train_users)]
    test_events = test_events[test_events["user_id"].isin(train_users)]

    train_data = build_train_examples(train_events)
    train_history = train_events.set_index("user_id").to_dict("index")
    valid_data = build_eval_examples(valid_events, train_history, "Build valid")

    valid_history = valid_events.set_index("user_id").to_dict("index")
    test_prior = {}
    for user_id, history in train_history.items():
        test_prior[user_id] = {key: list(value) for key, value in history.items()}
        if user_id in valid_history:
            for key in ("item_id", "ts", "query", "event_type"):
                test_prior[user_id][key] += valid_history[user_id][key]
    test_data = build_eval_examples(test_events, test_prior, "Build test")

    item_set = set()
    for data in (train_data, valid_data, test_data):
        for _, row in data.iterrows():
            item_set.update(row["item_list"])
            item_set.add(row["target_item"])
    test_data = add_test_negatives(test_data, item_set)

    split_and_save(train_data, "train")
    split_and_save(valid_data, "valid")
    split_and_save(test_data, "test")
    write_item_text(item_set)
    print(f"Amazon sequential data has been written to {SEQ_DATA_DIR}")


if __name__ == "__main__":
    main()
