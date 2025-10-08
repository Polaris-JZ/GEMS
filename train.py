import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
import logging
import os
from test import test
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
    # Load data
    train_data, valid_data, test_rec_data, test_src_data = load_datasets(args)

    # Define model
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
    
    # Print tokenizer and data information
    if args.rank == 0:
        print("add {} new token.".format(add_num))
        print("train data num:", len(train_data))
        print("valid data num:", len(valid_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print("train data sample:", train_data[100])
        print("valid data sample:", valid_data[100])

    # Define collator
    collator = Collator(args, tokenizer)

    # Define model
    model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)
    
    if args.rank == 0:
        print(model)

    # Create data loaders
    train_loader, valid_loader, train_sampler, valid_sampler = create_data_loaders(
        train_data, valid_data, args
    )



    
    # Check whether to use null space projection
    use_null_space = getattr(args, 'use_null_space', True)  # Enabled by default
    null_space_projections = {}
    
    if use_null_space:
        # Set null space related parameters
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
        
        # Update parameters in null_space_utils
        projector = NullSpaceProjector(nullspace_threshold=nullspace_threshold)

        # Get target layers
        target_layers = get_target_layers(model, target_modules)
        print(f"Found {len(target_layers)} target layers: {target_layers[:5]}...")  # Only show the first 5

        # Compute null space projection for each target layer
        for layer_name in target_layers:
            print(f"\nProcessing layer: {layer_name}")
            P = projector.get_or_compute_projection_matrix(
                model, tokenizer, layer_name, 
                cache_dir=cache_dir, 
                force_recompute=force_recompute,
                dataset_name=dataset_name,
                n_samples=n_samples
            )
            null_space_projections[layer_name] = P
            print(f"✓ Computed projection matrix for {layer_name}, shape: {P.shape}")

        print(f"\n✓ Successfully computed {len(null_space_projections)} null space projection matrices")

    else:
        print("Null space projection disabled.")
    

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, train_loader, args, null_space_projections)

    # Train model
    if not args.test:
        model = train_model(model, train_loader, valid_loader, optimizer, scheduler, collator, args, tokenizer)

    # Test after training
    if args.rank == 0:
        logging.info("Starting evaluation...")
        # Set checkpoint path to output directory
        args.ckpt_path = os.path.join(args.output_dir, 'checkpoint-best')
        test(args)


if __name__ == "__main__":
    # Get parser
    parser = argparse.ArgumentParser(description='BSR')
    parser = parse_all_args(parser)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)

    # Initialize distributed mode
    init_distributed_mode(args)

    # Set up logging
    if args.rank == 0:
        setup_logging(args)
        logging.info(f"Starting training with arguments: {args}")

    # Train
    train(args)


