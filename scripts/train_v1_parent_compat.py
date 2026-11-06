"""AD compatibility entry point; the former duplicate was identical to train_v1."""
from train_v1 import main, parse_args

if __name__ == "__main__":
    main(parse_args())
