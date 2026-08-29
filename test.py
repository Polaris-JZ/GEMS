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

def test_recommendation_only(args):
    """只测试推荐任务"""
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
        config=config,
        low_cpu_mem_usage=True,
        device_map=args.device,
    )

    print(f"Final model vocab_size: {model.config.vocab_size}")
    print(f"Model embedding size: {model.shared.weight.shape}")

    # print info of tokenizer and data
    if args.rank == 0:
        print("test rec data num:", len(test_rec_data))
        print("test rec data sample:", test_rec_data[100])

    # define collator
    collator = TestCollator(args, tokenizer)
    test_rec_loader = DataLoader(test_rec_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    # define constrain generation
    train_items = train_data.get_new_tokens()
    valid_items = valid_data.get_new_tokens()
    test_rec_items = test_rec_data.get_new_tokens()
    test_src_items = test_src_data.get_new_tokens()
    all_items = set(train_items + valid_items + test_rec_items + test_src_items)
    print("Number of unique items:", len(all_items))
    print("Sample items:", list(all_items)[:5])

    if not args.rerank:
        candidate_trie = Trie(
            [
                [0] + tokenizer.encode(candidate)
                for candidate in all_items
            ]
        )
        print("Trie size:", len(candidate_trie))
        print("Sample trie paths:", list(candidate_trie)[:5])
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    # start eval
    model.eval()
    metrics = args.metrics.split(",")
    rec_metrics_results = {}
    rec_total = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_rec_loader)):
            inputs = batch[0].to(args.device)
            targets = batch[1]
            batch_candidates = batch[2][0]
            rec_total += len(targets)
            
            if args.rerank:
                candidate_trie = Trie(
                    [
                        [0] + tokenizer.encode(str(candidate))
                        for candidate in batch_candidates
                    ]
                )
                prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
                
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
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

            output = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                        all_items=all_items if args.filter_items else None)

            batch_metrics_res = get_metrics_results(topk_res, metrics)

            for m, res in batch_metrics_res.items():
                if m not in rec_metrics_results:
                    rec_metrics_results[m] = res
                else:
                    rec_metrics_results[m] += res

        for m in rec_metrics_results:
            rec_metrics_results[m] = rec_metrics_results[m] / rec_total

        if args.rank == 0:
            logging.info("======================================================")
            logging.info(f"Recommendation Test results: {rec_metrics_results}")
            logging.info("======================================================")


def test_search_only(args):
    """只测试搜索任务"""
    # load data
    train_data, valid_data, test_rec_data, test_src_data = load_datasets(args)

    # define model
    print(f"Loading model from: {args.ckpt_path}")
    
    # 先加载配置检查vocab_size
    config = T5Config.from_pretrained(args.ckpt_path)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    
    # 确保配置一致
    if config.vocab_size != len(tokenizer):
        config.vocab_size = len(tokenizer)
    
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        config=config,
        low_cpu_mem_usage=True,
        device_map=args.device,
    )

    # print info of tokenizer and data
    if args.rank == 0:
        print("test src data num:", len(test_src_data))
        print("test src data sample:", test_src_data[100])

    # define collator
    collator = TestCollator(args, tokenizer)
    test_src_loader = DataLoader(test_src_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    # define constrain generation
    train_items = train_data.get_new_tokens()
    valid_items = valid_data.get_new_tokens()
    test_rec_items = test_rec_data.get_new_tokens()
    test_src_items = test_src_data.get_new_tokens()
    all_items = set(train_items + valid_items + test_rec_items + test_src_items)

    if not args.rerank:
        candidate_trie = Trie(
            [
                [0] + tokenizer.encode(candidate)
                for candidate in all_items
            ]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    # start eval
    model.eval()
    metrics = args.metrics.split(",")
    src_metrics_results = {}
    src_total = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_src_loader)):
            inputs = batch[0].to(args.device)
            targets = batch[1]
            batch_candidates = batch[2][0]
            src_total += len(targets)

            if args.rerank:
                candidate_trie = Trie(
                    [
                        [0] + tokenizer.encode(str(candidate))
                        for candidate in batch_candidates
                    ]
                )
                prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
                
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
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

            output = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                        all_items=all_items if args.filter_items else None)

            batch_metrics_res = get_metrics_results(topk_res, metrics)

            for m, res in batch_metrics_res.items():
                if m not in src_metrics_results:
                    src_metrics_results[m] = res
                else:
                    src_metrics_results[m] += res

        for m in src_metrics_results:
            src_metrics_results[m] = src_metrics_results[m] / src_total

        if args.rank == 0:
            logging.info("======================================================")
            logging.info(f"Search Test results: {src_metrics_results}")
            logging.info("======================================================")

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
    
    # try:
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

    # define collator
    collator = TestCollator(args, tokenizer)
    test_rec_loader = DataLoader(test_rec_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    test_src_loader = DataLoader(test_src_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    # define constrain generation
    train_items = train_data.get_new_tokens()
    valid_items = valid_data.get_new_tokens()
    test_rec_items = test_rec_data.get_new_tokens()
    test_src_items = test_src_data.get_new_tokens()
    all_items = set(train_items + valid_items + test_rec_items + test_src_items)
    print("Number of unique items:", len(all_items))
    print("Sample items:", list(all_items)[:5])

    if not args.rerank:
        candidate_trie = Trie(
            [
                [0] + tokenizer.encode(candidate)
                for candidate in all_items
            ]
        )

        print("Trie size:", len(candidate_trie))
        print("Sample trie paths:", list(candidate_trie)[:5])
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)


    # start eval
    model.eval()
    metrics = args.metrics.split(",")
    rec_metrics_results = {}
    src_metrics_results = {}
    rec_total = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_rec_loader)):
            inputs = batch[0].to(args.device)
            targets = batch[1]
            batch_candidates = batch[2][0]
            rec_total += len(targets)
            
            if args.rerank:
                candidate_trie = Trie(
                    [
                        [0] + tokenizer.encode(str(candidate))
                        for candidate in batch_candidates
                    ]
                )
                prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
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
            

            output = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                        all_items=all_items if args.filter_items else None)

            batch_metrics_res = get_metrics_results(topk_res, metrics)

            for m, res in batch_metrics_res.items():
                if m not in rec_metrics_results:
                    rec_metrics_results[m] = res
                else:
                    rec_metrics_results[m] += res

        for m in rec_metrics_results:
            rec_metrics_results[m] = rec_metrics_results[m] / rec_total

        if args.rank == 0:
            logging.info("======================================================")
            logging.info(f"Recommendation Test results: {rec_metrics_results}")
            logging.info("======================================================")


    with torch.no_grad():
        src_total = 0  # 为search循环单独初始化total
        for step, batch in enumerate(tqdm(test_src_loader)):
            inputs = batch[0].to(args.device)
            targets = batch[1]
            batch_candidates = batch[2][0]
            src_total += len(targets)

            if args.rerank:
                candidate_trie = Trie(
                    [
                        [0] + tokenizer.encode(str(candidate))
                        for candidate in batch_candidates
                    ]
                )
                prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)  # 添加缺失的定义
                
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
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

            output = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            # print(output)
            topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                        all_items=all_items if args.filter_items else None)

            batch_metrics_res = get_metrics_results(topk_res, metrics)


            for m, res in batch_metrics_res.items():
                if m not in src_metrics_results:
                    src_metrics_results[m] = res
                else:
                    src_metrics_results[m] += res

        for m in src_metrics_results:
            src_metrics_results[m] = src_metrics_results[m] / src_total  # 使用src_total而不是total

        if args.rank == 0:
            logging.info("======================================================")
            logging.info(f"Recommendation Test results: {rec_metrics_results}")
            logging.info("======================================================")
            logging.info(f"Search Test results: {src_metrics_results}")
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


