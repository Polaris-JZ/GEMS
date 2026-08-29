import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
import logging
import os
from test import test, test_recommendation_only, test_search_only
from parser import parse_all_args
from data import load_datasets
from utils import set_seed, init_distributed_mode, setup_logging
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from collator import Collator
from trainer import (
    create_data_loaders, 
    create_optimizer_and_scheduler, 
    train_model
)
from null_space_utils import NullSpaceProjector, get_target_layers


def train(args):
    if args.test:
        args.ckpt_path = os.path.join(args.output_dir, 'checkpoint-best')
        if args.search_only:
            test_search_only(args)
        elif args.rec_only:
            test_recommendation_only(args)
        else:
            test(args)
        return

    # 加载数据
    train_data, valid_data, test_rec_data, test_src_data = load_datasets(args)

    # 定义模型
    config = T5Config.from_pretrained(args.base_model, local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
        local_files_only=True
    )

    ori_vocab_size = config.vocab_size
    tokenizer.add_tokens(train_data.get_new_tokens())
    tokenizer.add_tokens(valid_data.get_new_tokens())
    tokenizer.add_tokens(test_rec_data.get_new_tokens())
    tokenizer.add_tokens(test_src_data.get_new_tokens())
    config.vocab_size = len(tokenizer)
    add_num = config.vocab_size - ori_vocab_size
    
    # 打印tokenizer和数据信息
    if args.rank == 0:
        print("add {} new token.".format(add_num))
        print("train data num:", len(train_data))
        print("valid data num:", len(valid_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print("train data sample:", train_data[100])
        print("valid data sample:", valid_data[100])

    # 定义collator
    collator = Collator(args, tokenizer)

    # 定义模型
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)
    
    if args.rank == 0:
        print(model)

    # 创建数据加载器
    train_loader, valid_loader, train_sampler, valid_sampler = create_data_loaders(
        train_data, valid_data, args
    )


    
    # 检查是否使用null space projection
    use_null_space = getattr(args, 'use_null_space', False)
    null_space_projections = {}
    
    if use_null_space:
        # 设置null space相关参数
        cache_dir = getattr(args, 'null_space_cache_dir', './null_space_cache')
        nullspace_threshold = getattr(args, 'nullspace_threshold', 2e-2)
        force_recompute = getattr(args, 'force_recompute_null_space', False)
        target_modules_str = getattr(args, 'null_space_target_modules', "attn,mlp")
        target_modules = [m.strip() for m in target_modules_str.split(',')]
        n_samples = getattr(args, 'null_space_n_samples', 10000)
        dataset_name = getattr(args, 'null_space_dataset', 'wikipedia')
        
        print(f"Null space cache directory: {cache_dir}")
        print(f"Null space threshold: {nullspace_threshold}")
        print(f"Force recompute: {force_recompute}")
        print(f"Target modules: {target_modules}")
        print(f"Number of samples: {n_samples}")
        print(f"Dataset: {dataset_name}")
        
        # 更新null_space_utils中的参数
        projector = NullSpaceProjector(nullspace_threshold=nullspace_threshold)
        
        # 获取目标层
        target_layers = get_target_layers(model, target_modules)
        print(f"Found {len(target_layers)} target layers: {target_layers[:5]}...")  # 只显示前5个
        
        # 所有目标层共享一次数据遍历来计算null space projection
        null_space_projections = projector.get_or_compute_projection_matrices(
            model, tokenizer, target_layers,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            dataset_name=dataset_name,
            n_samples=n_samples
        )

        print(f"\n✓ Successfully computed {len(null_space_projections)} null space projection matrices")

    else:
        print("Null space projection disabled.")
    

    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_loader, args, null_space_projections)

    # 训练模型
    model = train_model(model, train_loader, valid_loader, optimizer, scheduler, collator, args, tokenizer)

    # 训练完成后进行测试
    if args.rank == 0:
        logging.info("Starting evaluation...")
        # 设置checkpoint路径为输出目录
        args.ckpt_path = os.path.join(args.output_dir, 'checkpoint-best')
        # test(args)

        # 根据参数决定运行哪种测试
        if args.search_only:
            test_search_only(args)
        elif args.rec_only:
            test_recommendation_only(args)
        else:
            test(args)

if __name__ == "__main__":
    # 获取解析器
    parser = argparse.ArgumentParser(description='BSR')
    parser = parse_all_args(parser)
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)

    # 初始化分布式模式
    init_distributed_mode(args)

    # 设置日志
    if args.rank == 0:
        setup_logging(args)
        logging.info("="*80)
        logging.info(f"Starting new training session at {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
        logging.info(f"Starting training with arguments: {args}")
        logging.info("="*80)

    # 训练
    train(args)


