import json
import logging
import os
import random
import datetime
import transformers
import numpy as np
import torch

class CustomCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            if 'loss' in logs:
                logging.info(f'Step: {state.global_step}, Loss: {logs["loss"]:.4f}')
            if 'epoch' in logs:
                logging.info(f'Epoch: {logs["epoch"]:.2f}')
            if 'eval_loss' in logs:
                logging.info(f'Eval Loss: {logs["eval_loss"]:.4f}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def init_distributed_mode(args):
    if args.distributed:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.gpu = int(os.environ["LOCAL_RANK"])
        else:
            print("Not using distributed mode")
            args.distributed = False
            return

        # Parse GPU IDs
        gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
        if args.rank >= len(gpu_ids):
            raise ValueError(f"Rank {args.rank} is out of range for specified GPUs {gpu_ids}")
        
        args.distributed = True
        torch.cuda.set_device(0)  # 使用cuda:0，因为CUDA_VISIBLE_DEVICES已经限制了可见设备
        args.dist_backend = "nccl"
        args.device = "cuda:0"  # 使用cuda:0
        
        # Set default master address and port if not provided
        if not hasattr(args, 'master_addr'):
            args.master_addr = 'localhost'
        if not hasattr(args, 'master_port'):
            args.master_port = '12355'
            
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        
        print(
            "| distributed init (rank {}, world {}, gpu {}): {}".format(
                args.rank, args.world_size, gpu_ids[args.rank], f"tcp://{args.master_addr}:{args.master_port}"
            ),
            flush=True,
        )
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(days=365)
        )
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.device = "cuda:0"  # 使用cuda:0

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def setup_logging(args):    
    log_file = os.path.join(args.output_dir, 'training.log')
    # Clear existing handlers to avoid duplicate logging
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 'w' mode for overwriting
            logging.StreamHandler()
        ]
    )