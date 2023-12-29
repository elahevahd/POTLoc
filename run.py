import os
import shutil
import argparse
import base_core
import main_core


def parse_args():
    descript = 'Pytorch Implementation of \'POTLoc:  Pseudo-Label Oriented Transformer for Point-Supervised Temporal Action Localization\''
    parser = argparse.ArgumentParser(description=descript)
    parser.add_argument('--feat_folder', type=str, default='./data/thumos/i3d_features')
    parser.add_argument('--data_path', type=str, default='./data/thumos/')
    parser.add_argument('--gt_path', type=str, default='./data/thumos/annotations/thumos14.json')
    parser.add_argument('--point_dict_path', type=str, default='./data/thumos/annotations/point_dict.json')
    parser.add_argument('--base_model_path', type=str, default='./base_models')
    parser.add_argument('--log_path', type=str, default='./base_models/logs')
    parser.add_argument('--base_output_path', type=str, default='./base_outputs')
    parser.add_argument('--PL_path', type=str, default='./base_outputs/PL_dict.json')
    parser.add_argument('--main_model_path', type=str, default='./main_models')
    parser.add_argument('--main_output_path', type=str, default='./main_outputs')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument('--ckpt_freq', default=5, type=int, help='checkpoint frequency (default: every 5 epochs)')
    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.base_model_path):
        os.makedirs(args.base_model_path)
    if not os.path.exists(args.main_model_path):
        os.makedirs(args.main_model_path)
    ######################################################
    if not os.path.exists(args.base_output_path):
        os.makedirs(args.base_output_path)
    if not os.path.exists(args.main_output_path):
        os.makedirs(args.main_output_path)
    ######################################################
    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    return args


if __name__ == "__main__":
    args = parse_args()

    print("Generate pseudo-labels on the train set......")
    base_core.base_function(args)
    print("Pseudo-labels are generated!")

    print("Train the main model......")
    main_core.train(args)

    print("Test the main model......")
    main_core.test(args)
