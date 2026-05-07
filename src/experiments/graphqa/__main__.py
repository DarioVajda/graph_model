from .train import run_training, parse_args

if __name__ == "__main__":
    args = parse_args()
    run_training(
        without=args.without,
        graph_type=args.graph_type,
        task=args.task,
        lora_r=args.lora_r,
    )