# GEMS
This is the pytorch implementation of our paper
> Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning

## Data preparation

Run all commands from the repository root. Install the preprocessing dependencies first:

```bash
pip install pandas numpy tqdm
```

Place the following Qilin source files in `data/qilin/raw_data/`:

```text
recommendation_train.csv
recommendation_test.csv
search_train.csv
search_test.csv
```

Then generate the training data in two steps:

```bash
python data/qilin/process_step_1.py
python data/qilin/process_step_2.py
```

The first command performs 3-core filtering and writes intermediate files to `data/qilin/ori_data/`. The second command expands positive interactions, resamples negatives, and writes the model-ready files to `data/qilin/seq_data/`:

```text
train_rec.pkl   train_src.pkl
valid_rec.pkl   valid_src.pkl
test_rec.pkl    test_src.pkl
```

This output directory is the same path used by `train.sh`; no additional path changes are required.

## Training

Open `train.sh` and replace the placeholder with the path to your local T5-base model:

```bash
BASE_MODEL=/path/to/your/t5-base-model
```

Start training from the repository root:

```bash
bash train.sh
```

Training runs in the background. Check `output/train.txt` for console output and `output/training.log` for the training log. `train.sh` validates the six generated Qilin data files before launching training.