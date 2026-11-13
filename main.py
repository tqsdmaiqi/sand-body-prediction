import argparse
from petrel import run as petrel
from reproduce import run as reproduce
from config import load_config
from features import run as features
from sampling import run as sampling
from sample_statistics import run as sample_statistics
from train import run as train
from channel import run as channel
from structure import run as structure
from evaluate import run as evaluate
from uncertainty import run as uncertainty
from ablation import run as ablation
from plot_validation import run as plot_validation
from plot_map import run as plot_map

TASKS = {
    "petrel": petrel,
    "reproduce": reproduce,
    "features": features,
    "sampling": sampling,
    "sample_statistics": sample_statistics,
    "train": train,
    "channel": channel,
    "structure": structure,
    "evaluate": evaluate,
    "uncertainty": uncertainty,
    "ablation": ablation,
    "plot_validation": plot_validation,
    "plot_map": plot_map,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    args = parser.parse_args()
    TASKS[args.task](params=load_config(args.task))


if __name__ == "__main__":
    main()
