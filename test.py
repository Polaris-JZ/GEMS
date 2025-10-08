import argparse
import torch
import transformers
from transformers import EarlyStoppingCallback

from parser import parse_all_args
from data import load_datasets
from utils import set_seed, init_distributed_mode
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from collator import Collator, TestCollator
from tqdm import tqdm
from torch.utils.data import DataLoader
from generation_trie import Trie, prefix_allowed_tokens_fn
from evaluate import get_topk_results, get_metrics_results
import logging

def test(args):
    # load data
    train_data, valid_data, test_rec_data, test_src_data = load_datasets(args)

    # define model
    print(f"Loading model from: {args.ckpt_path}")
    
    # 先加载配置检查vocab_size
    config = T5Config.from_pretrained(args.ckpt_path)
    print(f"Model config vocab_size: {config.vocab_size}")
    
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    print(f"Tokenizer vocab_size: {len(tokenizer)}")
    
    # 确保配置一致
    if config.vocab_size != len(tokenizer):
        print(f"WARNING: Config vocab_size ({config.vocab_size}) != Tokenizer vocab_size ({len(tokenizer)})")
        print("Adjusting config to match tokenizer...")
        config.vocab_size = len(tokenizer)
    
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        config=config,  # 使用调整后的配置
        low_cpu_mem_usage=True,
        device_map=args.device,
    )

    print(f"Final model vocab_size: {model.config.vocab_size}")
    print(f"Model embedding size: {model.shared.weight.shape}")

    # print info of tokenizer and data
    if args.rank == 0:
        print("test rec data num:", len(test_rec_data))
        print("test rec data sample:", test_rec_data[100])
        print("test src data num:", len(test_src_data))
        print("test src data sample:", test_src_data[100])

    # 优化1: 预计算所有候选项的全局Trie (用于非rerank模式)
    train_items = train_data.get_new_tokens()
    valid_items = valid_data.get_new_tokens()
    test_rec_items = test_rec_data.get_new_tokens()
    test_src_items = test_src_data.get_new_tokens()
    all_items = set(train_items + valid_items + test_rec_items + test_src_items)
    print("Number of unique items:", len(all_items))
    print("Sample items:", list(all_items)[:5])

    global_prefix_allowed_tokens = None
    if not args.rerank:
        candidate_trie = Trie(
            [
                [0] + tokenizer.encode(candidate)
                for candidate in all_items
            ]
        )
        print("Trie size:", len(candidate_trie))
        print("Sample trie paths:", list(candidate_trie)[:5])
        global_prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    # 优化2: 使用统一的测试函数
    print("Testing Recommendation Task...")
    rec_metrics_results = test_single_task(
        args, model, tokenizer, test_rec_data, global_prefix_allowed_tokens, all_items, "recommendation"
    )

    print("Testing Search Task...")
    src_metrics_results = test_single_task(
        args, model, tokenizer, test_src_data, global_prefix_allowed_tokens, all_items, "search"
    )

    # 输出最终结果
    if args.rank == 0:
        logging.info("======================================================")
        logging.info(f"Recommendation Test results: {rec_metrics_results}")
        logging.info("======================================================")
        logging.info(f"Search Test results: {src_metrics_results}")
        logging.info("======================================================")


def test_single_task(args, model, tokenizer, test_data, global_prefix_allowed_tokens, all_items, task_name):
    """优化后的单任务测试函数 - 针对独特候选集优化"""
    # 在rerank模式下，每个样本候选集都独特，所以batch_size必须为1
    effective_batch_size = 1 if args.rerank else max(1, args.test_batch_size)
    
    collator = TestCollator(args, tokenizer)
    test_loader = DataLoader(
        test_data, 
        batch_size=effective_batch_size, 
        collate_fn=collator,
        shuffle=False,  # 测试时不需要shuffle，可以提高效率
        num_workers=1,  # rerank模式下减少worker避免开销
        pin_memory=True
    )

    model.eval()
    metrics = args.metrics.split(",")
    metrics_results = {}
    total_samples = 0
    
    # 优化：预编译tokenizer以减少重复调用开销
    if hasattr(tokenizer, 'backend_tokenizer'):
        # 对于快速tokenizer，可以进行一些预热
        _ = tokenizer.encode("warm_up_token")

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc=f"Testing {task_name}")):
            inputs = batch[0].to(args.device)
            targets = batch[1]
            batch_candidates_list = batch[2]
            
            # 处理批次中的每个样本
            for sample_idx in range(len(targets)):
                sample_targets = targets[sample_idx] if isinstance(targets[sample_idx], list) else [targets[sample_idx]]
                batch_candidates = batch_candidates_list[sample_idx] if len(batch_candidates_list) > sample_idx else batch_candidates_list[0]
                total_samples += len(sample_targets)

                # 准备该样本的输入
                sample_inputs = {
                    "input_ids": inputs["input_ids"][sample_idx:sample_idx+1],
                    "attention_mask": inputs["attention_mask"][sample_idx:sample_idx+1]
                }

                # 获取prefix_allowed_tokens
                if args.rerank:
                    # 优化：由于每个候选集都独特，直接构建，但优化构建过程
                    # 使用批量编码减少tokenizer调用次数
                    encoded_candidates = tokenizer(
                        [str(candidate) for candidate in batch_candidates], 
                        add_special_tokens=False,
                        return_tensors=None,  # 返回list而不是tensor，更轻量
                        padding=False,
                        truncation=False
                    )['input_ids']
                    
                    candidate_trie = Trie([
                        [0] + encoded_candidate
                        for encoded_candidate in encoded_candidates
                    ])
                    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
                else:
                    prefix_allowed_tokens = global_prefix_allowed_tokens

                # 生成预测
                output = model.generate(
                    input_ids=sample_inputs["input_ids"],
                    attention_mask=sample_inputs["attention_mask"],
                    max_new_tokens=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                # 解码输出
                decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                # 计算指标
                topk_res = get_topk_results(
                    decoded_output, scores, sample_targets, args.num_beams,
                    all_items=all_items if args.filter_items else None
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)

                # 累积指标
                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

    # 计算平均指标
    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total_samples

    return metrics_results


def test_search_only(args):
    """只测试搜索任务"""
    train_data, valid_data, test_rec_data, test_src_data = load_datasets(args)
    
    # 加载模型和tokenizer的代码...
    config = T5Config.from_pretrained(args.ckpt_path)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    if config.vocab_size != len(tokenizer):
        config.vocab_size = len(tokenizer)
    
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path, config=config, low_cpu_mem_usage=True, device_map=args.device
    )
    
    # 准备全局候选项
    all_items = set(train_data.get_new_tokens() + valid_data.get_new_tokens() + 
                   test_rec_data.get_new_tokens() + test_src_data.get_new_tokens())
    
    global_prefix_allowed_tokens = None
    if not args.rerank:
        candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
        global_prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    
    # 只测试搜索任务
    src_metrics_results = test_single_task(
        args, model, tokenizer, test_src_data, global_prefix_allowed_tokens, all_items, "search"
    )
    
    if args.rank == 0:
        logging.info("======================================================")
        logging.info(f"Search Test results: {src_metrics_results}")
        logging.info("======================================================")


def test_recommendation_only(args):
    """只测试推荐任务"""
    train_data, valid_data, test_rec_data, test_src_data = load_datasets(args)
    
    # 加载模型和tokenizer的代码...
    config = T5Config.from_pretrained(args.ckpt_path)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    if config.vocab_size != len(tokenizer):
        config.vocab_size = len(tokenizer)
    
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path, config=config, low_cpu_mem_usage=True, device_map=args.device
    )
    
    # 准备全局候选项
    all_items = set(train_data.get_new_tokens() + valid_data.get_new_tokens() + 
                   test_rec_data.get_new_tokens() + test_src_data.get_new_tokens())
    
    global_prefix_allowed_tokens = None
    if not args.rerank:
        candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
        global_prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    
    # 只测试推荐任务
    rec_metrics_results = test_single_task(
        args, model, tokenizer, test_rec_data, global_prefix_allowed_tokens, all_items, "recommendation"
    )
    
    if args.rank == 0:
        logging.info("======================================================")
        logging.info(f"Recommendation Test results: {rec_metrics_results}")
        logging.info("======================================================")

        




if __name__ == "__main__":
    # get parser
    parser = argparse.ArgumentParser(description='BSR')
    parser = parse_all_args(parser)
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # init distributed mode
    init_distributed_mode(args)

    # 根据参数决定运行哪种测试
    if args.search_only:
        test_search_only(args)
    elif args.rec_only:
        test_recommendation_only(args)
    else:
        test(args)


