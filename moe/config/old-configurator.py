"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import argparse
import json

def load_config():

    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process some command line arguments.')

    # Add arguments
    parser.add_argument('--config_file', type=str, help='Path to a Python config file')
    parser.add_argument('--overrides', nargs='*', help='List of key=value pairs to override config values')
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--eval_interval', type=int, help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, help='Number of evaluation iterations')
    parser.add_argument('--log_interval', type=int, help='Log interval')
    parser.add_argument('--always_save_checkpoint', type=bool, help='Always save checkpoint')
    parser.add_argument('--wandb_log', type=bool, help='Wandb log')
    parser.add_argument('--wandb_project', type=str, help='Wandb project')
    parser.add_argument('--wandb_run_name', type=str, help='Wandb run name')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--block_size', type=int, help='Block size')
    parser.add_argument('--n_layer', type=int, help='Number of layers')
    parser.add_argument('--n_head', type=int, help='Number of heads')
    parser.add_argument('--n_embd', type=int, help='Embedding size')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_iters', type=int, help='Maximum number of iterations')
    parser.add_argument('--lr_decay_iters', type=int, help='Learning rate decay iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--decay_lr', type=bool, default=False, help='Decay learning rate')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for optimizer')
    parser.add_argument('--beta2', type=float, default=0.95, help='Beta2 for optimizer')
    parser.add_argument('--warmup_iters', type=int, help='Warmup iterations')
    parser.add_argument('--device', type=str, help='Device to use')
    parser.add_argument('--compile', type=bool, help='Whether to compile')
    parser.add_argument('--bias', type=bool, default=False, help='Whether to use bias')
    parser.add_argument('--init_from', type=str, default='scratch', help='Init from')

    # Parse the arguments
    args = parser.parse_args()

    # Load the default configuration
    with open(args.config_file) as f:
        config = json.load(f)

    # Override the defaults with any command-line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    
    return config