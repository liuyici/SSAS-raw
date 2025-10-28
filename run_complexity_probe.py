# Quick runner to construct models and run complexity.report_models without loading datasets or training.
import os
import torch
from types import SimpleNamespace
import network

# import project models
import network
import new_network
from new_network import MLPBase, feat_bottleneck, feat_classifier
from complexity import report_models


def main():
    args = SimpleNamespace()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output_dir = 'complexity_run'
    args.batch_size = 50
    args.bottleneck_dim = 128
    args.num_class = 3
    args.num_class2 = 14

    # prepare snapshot dir and log file
    out_dir = os.path.join('snapshot', args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'log_probe.txt')
    log_file = open(log_path, 'w')
    args.log_file = log_file

    # model sizes same as selection_domain_new / solvers
    input_size = 2790  # SEED
    hidden_size = 320

    # instantiate models
    model = network.DFN(input_size=input_size, hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class2, radius=10)
    netA = MLPBase(input_size=input_size, hidden_size=hidden_size)
    netB = feat_bottleneck(hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim)
    netC = feat_classifier(bottleneck_dim=args.bottleneck_dim, class_num=args.num_class)
    netD = feat_classifier(bottleneck_dim=args.bottleneck_dim, class_num=args.num_class2)
    netF = feat_classifier(bottleneck_dim=args.bottleneck_dim, class_num=args.num_class2)

    # prepare models list
    models_list = [
        ("DFN", model),
        ("netA", netA),
        ("netB", netB),
        ("netC", netC),
        ("netD", netD),
        ("netF_sim", netF),
    ]

    prof_batch = min(8, args.batch_size)
    input_shapes = {
        "DFN": (prof_batch, input_size),
        "netA": (prof_batch, input_size),
        "netB": (prof_batch, hidden_size),
        "netC": (prof_batch, args.bottleneck_dim),
        "netD": (prof_batch, args.bottleneck_dim),
        "netF_sim": (prof_batch, args.bottleneck_dim),
    }

    print(f"Running complexity probe on device {args.device}, writing logs to {log_path}")
    report_models(models_list, args=args, input_shapes=input_shapes, runs=20, warmup=5)

    log_file.close()

    # list created files
    files = os.listdir(out_dir)
    print("Created files in:", out_dir)
    for f in files:
        print(" -", f)

if __name__ == '__main__':
    main()
