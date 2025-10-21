from argparse import ArgumentParser
from os import system

from codebase.experiment import run_experiment as run_xp

parser = ArgumentParser(description="A simple command-line tool.")
parser.add_argument(
    "--mode",
    choices=["explore", "runxp"],
    required=True,
    help="Mode of operation: 'explore' to explore options, 'runxp' to run an experiment.",
)


def explore_options():
    print("Exploring options...")

    system("streamlit run exploration.py")


def run_experiment():
    print("Running experiment...")
    run_xp()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "explore":
        explore_options()
    elif args.mode == "runxp":
        run_experiment()
