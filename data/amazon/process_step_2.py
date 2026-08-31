import pickle
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
SEQ_DATA_DIR = DATA_DIR / "seq_data"
BSR_DATA_DIR = DATA_DIR / "bsr_data"
SPLITS = ("train", "valid", "test")
TASKS = ("rec", "src")


def load_records(name):
    with open(SEQ_DATA_DIR / name, "rb") as file:
        data = pickle.load(file)
    return pd.DataFrame.from_dict(data, orient="index")


def save_records(data, name):
    with open(BSR_DATA_DIR / name, "wb") as file:
        pickle.dump(data.reset_index(drop=True).to_dict(orient="index"), file)


def filter_recommendation_history(data):
    data = data.copy()
    if "his_task" in data.columns:
        data["item_list"] = data.apply(
            lambda row: [
                item
                for item, task in zip(row["item_list"], row["his_task"])
                if task == "rec"
            ],
            axis=1,
        )
    return data


def select_columns(data, task, split):
    columns = ["user_id", "item_list", "target_item"]
    if task == "src":
        columns.append("query")
    if split == "test":
        columns.append("neg_note_idx")

    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"{split}_{task}.pkl is missing required columns: {', '.join(missing)}"
        )
    return data[columns]


def validate_inputs():
    required = [f"{split}_{task}.pkl" for split in SPLITS for task in TASKS]
    missing = [name for name in required if not (SEQ_DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing Amazon sequential data in {SEQ_DATA_DIR}: {', '.join(missing)}. "
            "Run process_step_1.py first."
        )


def main():
    validate_inputs()
    BSR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for task in TASKS:
            name = f"{split}_{task}.pkl"
            data = load_records(name)
            if task == "rec":
                data = filter_recommendation_history(data)
            save_records(select_columns(data, task, split), name)

    print(f"Amazon training data has been written to {BSR_DATA_DIR}")


if __name__ == "__main__":
    main()
