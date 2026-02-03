import argparse
import os
import glob
import time
import json
import warnings
import open_clip
import torch
import torch.nn as nn
import numpy as np

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed
from metric import calculate_micro_auc, evaluate_ed, evaluate_seg
from libs.datasets.clip_encode import action_tokenize_egoper

warnings.filterwarnings("ignore")

def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    # assert len(cfg['val_split']) > 0, "Test set must be specified!"
    assert len(cfg['test_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            if args.is_best:
                ckpt_file = os.path.join(args.ckpt, 'best.pth.tar')
            else:
                ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
                ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    # load CLIP model
    clip, _, preprocess = open_clip.create_model_and_transforms(
        model_name="EVA02-L-14-336", 
        pretrained="merged2b_s6b_b61k",
        device="cuda"
    )
    tokenizer = open_clip.get_tokenizer("EVA02-L-14-336")
    clip_model = (clip.eval(), tokenizer, preprocess)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    # pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    test_dataset = make_dataset(
            cfg['dataset_name'], False, clip_model, cfg['test_split'], args.use_gcn, **cfg['dataset']
        )
    # set bs = 1, and disable shuffle
    test_loader = make_data_loader(
        test_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    # load annotation
    with open("data/annotation.json", 'r') as fp:
        all_annot = json.load(fp)
    task_annot = all_annot[cfg['dataset']['task']]

    ## move clip model to device
    clip_model = (clip_model[0].to(cfg['devices'][0]), clip_model[1], clip_model[2])
    action_tokens = action_tokenize_egoper(
        clip_model,
        cfg['dataset']['task'],
        annot=task_annot,
        device=cfg['devices'][0]
    )

    """3. create model and evaluator"""
    # model
    cfg['model']['backbone_type'] = 'convGCNTransformer' if args.use_gcn else 'convTransformer'
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    # model.load_state_dict(checkpoint['state_dict_ema'])
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()

    model.eval()
    # dict for results (for our evaluation code)
    results = {}
    for iter_idx, video_list in enumerate(test_loader, 0):
        with torch.no_grad():
            batched_result = model.module.inference(
                video_list=video_list,
                clip_model=clip_model,
                action_tokens=action_tokens
            )
            results.update(batched_result)
            
    """4. Segmentation Accuracy"""
    list_of_gt_frame_action = []
    list_of_pred_frame_action = []
    for k,v in results.items():
        list_of_gt_frame_action.append(v['frame_action_label'].clone().cpu().numpy())
        list_of_pred_frame_action.append(v['frame_action_pred'].clone().cpu().numpy())    
    seg_outputs = evaluate_seg(list_of_gt_frame_action, list_of_pred_frame_action)
    print("Segmentation Accuracy: {:.3f} mIoU: {:.3f} Edit: {:.3f} F1@0.5:{:.3f}".format(\
        seg_outputs['Accuracy'], seg_outputs['mIoU'], seg_outputs['Edit'], seg_outputs['F1@0.5'])
    )

    """5. Error Detection Accuracy"""
    threshold = 0.0
    eda_list = []
    micro_fprs, micro_tprs = [], []

    while threshold <= 1.0:
        outputs = evaluate_ed(results, threshold=threshold)
        eda_list.append(outputs['error_acc'])
        micro_fprs.append(outputs['micro']["fpr"])
        micro_tprs.append(outputs['micro']["tpr"])
        threshold += 0.025  ## follow EGOPER paper
        
    eda = np.array(eda_list).mean() * 100
    micro_auc_value = calculate_micro_auc(micro_fprs, micro_tprs)
    micro_auc_value = micro_auc_value * 100
    end = time.time()

    print("Detection Error Accuracy: {:0.2f}%, Micro AUC: {:.2f}% ".format(eda, micro_auc_value))
    print("All done! Total time: {:0.2f} sec".format(end - start))

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--is_best', action='store_true')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--use_gcn', default=False, action='store_true',
                help='no effect modeling for the model')
    args = parser.parse_args()
    main(args)
