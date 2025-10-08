
def parse_all_args(parser):
    # DDP
    parser.add_argument("--distributed", action="store_true", default=False, help="Whether to use distributed training")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--dist_url", type=str, default="env://", help="Distributed URL")
    parser.add_argument("--master_addr", type=str, default='localhost', help='Setup MASTER_ADDR for os.environ')
    parser.add_argument("--master_port", type=str, default='12313', help='Setup MASTER_PORT for os.environ')

    # global
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str,default="./output/",help="The output directory")
    parser.add_argument("--ckpt_path", type=str, default="./output/", help="ckpt path")
    
    # data
    parser.add_argument("--data_path", type=str, default="/projects/0/prjs1158/jujia_ws/GenSR_Surface/data/kindle/bsr_data", help="data path")
    
    # model
    parser.add_argument("--base_model", type=str, default="google/flan-t5-base", help="base model")
    
    # train 
    parser.add_argument("--deepspeed", type=str, default=None, help="deepspeed config")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="gradient checkpointing")
    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--print_steps", type=int, default=10, help="Print loss every N steps")
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_steps_per_epoch", type=int, default=2, help="Number of checkpoints to save per epoch")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of validation steps without improvement before early stopping")
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)

    # GaLore 参数
    parser.add_argument("--galore_rank", type=int, default=1024, help="GaLore rank parameter")
    parser.add_argument("--galore_update_proj_gap", type=int, default=200, help="GaLore update projection gap")
    parser.add_argument("--galore_scale", type=float, default=0.25, help="GaLore scale parameter")

    # 双平面优化参数（只保留梯度平衡相关）
    parser.add_argument("--use_dual_space", action="store_true", default=False, help="Whether to use dual-space optimization with gradient balancing")
    parser.add_argument("--gradient_merge_strategy", type=str, default="weighted",
                        choices=["weighted", "rec_priority", "conflict_aware"],
                        help="Strategy for merging gradients in dual-space optimization")

    # eval
    parser.add_argument("--metrics", type=str, default="hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20,mrr@5,mrr@10,mrr@20,map@5,map@10,map@20",
                        help="test metrics, separate by comma")
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--rerank", action="store_true", default=False, help="whether use rerank")
    
    # only test mode
    parser.add_argument("--test", action="store_true", default=False, help="whether to run in test mode")   

    # gate
    parser.add_argument("--use_learnable_gating", action="store_true", default=False, help="Whether to use adaptive gating for loss weighting")
    
    # ========== 门控核心超参数 ==========
    parser.add_argument("--gating_learning_rate", type=float, default=1e-4, help="Learning rate for gating network [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]")
    parser.add_argument("--initial_temperature", type=float, default=1.0, help="Temperature for gating softmax [0.5, 1.0, 1.5, 2.0]")
    parser.add_argument("--gating_hidden_dim", type=int, default=128, help="Hidden dimension for gating network [64, 128, 256]")
    
    # 启发式权重参数 (当learnable gating关闭时使用)
    parser.add_argument("--loss_weight_alpha", type=float, default=0.4, help="Weight for loss-based gating [0.2, 0.3, 0.4, 0.5, 0.6]")
    parser.add_argument("--grad_weight_beta", type=float, default=0.4, help="Weight for gradient-norm-based gating [0.2, 0.3, 0.4, 0.5, 0.6]")
    parser.add_argument("--sample_weight_gamma", type=float, default=0.2, help="Weight for sample-count-based gating")
    
    # 其他门控参数使用固定默认值
    parser.add_argument("--gating_dropout", type=float, default=0.1, help="Dropout for gating network (fixed)")
    parser.add_argument("--min_temperature", type=float, default=0.1, help="Minimum temperature (fixed)")
    parser.add_argument("--temp_decay_factor", type=float, default=0.99, help="Temperature decay factor (fixed)")

    # Null Space Projection 相关参数
    parser.add_argument("--use_null_space", action="store_true", default=True, 
                        help="Whether to use null space projection")
    parser.add_argument("--nullspace_threshold", type=float, default=2e-2, 
                        help="Threshold for null space detection (default: 2e-2)")
    parser.add_argument("--null_space_cache_dir", type=str, default="./null_space_cache", 
                        help="Directory to cache null space projection matrices")
    parser.add_argument("--force_recompute_null_space", action="store_true", default=False, 
                        help="Force recompute null space projection matrices")
    parser.add_argument("--null_space_target_modules", type=str, default="attn,mlp", 
                        help="Target modules for null space projection (comma-separated)")
    parser.add_argument("--null_space_n_samples", type=int, default=10000, 
                        help="Number of samples to use for computing covariance matrix")
    parser.add_argument("--null_space_dataset", type=str, default="wikipedia", 
                        help="Dataset to use for computing null space (wikipedia, wikitext)")
                        
    return parser
