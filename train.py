import argparse
import os
import time
import datetime
import json
import warnings
import torch
import torch.nn as nn
import open_clip

from tensorboardX import SummaryWriter
from pprint import pprint
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, save_checkpoint,
                        make_optimizer, make_scheduler, fix_random_seed, ModelEma)
from libs.datasets.clip_encode import action_tokenize_egoper

warnings.filterwarnings("ignore")

def train_model(
    cfg, args, 
    model, model_ema, 
    optimizer, scheduler, 
    train_loader, val_loader, 
    action_tokens, clip_model, 
    tb_writer, ckpt_folder, 
    logger_folder
):
    """
    Handles the main training loop, including validation and checkpointing.
    """
    print(f"\n--- Starting Training ---")
    max_epochs = cfg['opt'].get('early_stop_epochs', cfg['opt']['epochs'] + cfg['opt']['warmup_epochs'])
    
    for epoch in range(max_epochs):
        model_ema = train_one_epoch(
            train_loader, action_tokens, model, clip_model, optimizer, scheduler, epoch, max_epochs,
            model_ema=model_ema, clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'], tb_writer=tb_writer
        )

        valid_one_epoch(
            val_loader, action_tokens, model, clip_model, epoch, max_epochs, tb_writer=tb_writer
        )

        # Save checkpoint periodically
        if ((epoch + 1) == max_epochs) or ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'state_dict_ema': model_ema.module.state_dict()
            }
            save_checkpoint(
                save_states, False,
                file_folder=ckpt_folder,
                file_name=f'epoch_{epoch + 1:03d}.pth.tar'
            )

    # Save the last model checkpoint
    last_states = {
        'epoch': max_epochs,
        'state_dict': model.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'state_dict_ema': model_ema.module.state_dict()
    }
    save_checkpoint(last_states, False, file_folder=logger_folder, file_name=f'epoch_{max_epochs:03d}.pth.tar')


def main(args):
    """
    Main function to orchestrate the setup, training, and evaluation.
    """

    """1. Setup parameters / folders"""
    args.start_epoch = 0
    cfg = load_config(args.config)

    # Prepare output folder
    if not os.path.exists(cfg['output_folder']):
        os.makedirs(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
        
    logger_folder = os.path.join(ckpt_folder, 'logs', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    tb_writer = SummaryWriter(logger_folder)

    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # Scale learning rate and workers based on GPU count
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk

    # Load CLIP model
    clip, _, preprocess = open_clip.create_model_and_transforms(
        model_name="EVA02-L-14-336",
        pretrained="merged2b_s6b_b61k",
        device="cuda"
    )
    tokenizer = open_clip.get_tokenizer("EVA02-L-14-336")
    clip_model = (clip.eval(), tokenizer, preprocess)

    """2. Create datasets / dataloaders"""
    # Training set
    train_dataset = make_dataset(cfg['dataset_name'], True, clip_model, cfg['train_split'], args.use_gcn, **cfg['dataset'])
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(cfg['dataset_name'], False, clip_model, cfg['val_split'], args.use_gcn, **cfg['dataset'])
    val_loader = make_data_loader(val_dataset, False, None, 1, cfg['loader']['num_workers'])

    # Load annotations and tokenize actions
    with open("data/annotation.json", 'r') as fp:
        all_annot = json.load(fp)
    task_annot = all_annot[cfg['dataset']['task']]
    action_tokens = action_tokenize_egoper(
        clip_model, 
        cfg['dataset']['task'], 
        annot=task_annot, 
        device="cuda"
    )

    """3. Create model, optimizer, and scheduler"""
    cfg['model']['backbone_type'] = 'convGCNTransformer' if args.use_gcn else 'convTransformer'
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    print("Using model EMA ...")
    model_ema = ModelEma(model)

    # Resume from a checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(cfg['devices']))
            device_id = cfg['devices'][0] if isinstance(cfg['devices'], list) else cfg['devices']
            # Convert device_id to int if it's a string, default to 0 if conversion fails
            if isinstance(device_id, str):
                device_id = 0
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(device_id))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'], strict=False)
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            del checkpoint
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            return

    # Save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()
    
    """4. Run Training and Evaluation"""
    # Call the dedicated training function
    train_model(
        cfg, args, model, model_ema, optimizer, scheduler, train_loader, val_loader,
        action_tokens, clip_model, tb_writer, ckpt_folder, logger_folder
    )
    
    tb_writer.close()
    print("\nAll done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Evaluate a transformer for action localization')
    parser.add_argument('config', metavar='DIR', help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int, help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str, help='name of experiment folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to a checkpoint to resume training (default: none)')
    parser.add_argument('--use_gcn', default=False, action='store_true', help='use GCN in the model backbone')
    parser.add_argument('-t', '--topk', default=-1, type=int, help='max number of output actions for evaluation (default: -1)')
    
    args = parser.parse_args()
    main(args)