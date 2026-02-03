import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from copy import deepcopy
from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR


def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config['weight_decay']
    )

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def train_one_epoch(
    train_loader,
    action_tokens,
    model,
    clip_model,
    optimizer,
    scheduler,
    curr_epoch,
    tot_epoch,
    model_ema = None,
    clip_grad_l2norm = 1.0,
    tb_writer = None
):
    """Training the model for one epoch"""
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    tot_loss_meter = AverageMeter()
    action_contrast_loss_meter = AverageMeter()
    action_contrast_top1_acc_meter = AverageMeter()
    action_contrast_top3_acc_meter = AverageMeter()
    action_contrast_top5_acc_meter = AverageMeter()
    img_graph_rel_loss_meter = AverageMeter()
    img_graph_stt_loss_meter = AverageMeter()
    effect_rel_loss_meter = AverageMeter()
    effect_stt_loss_meter = AverageMeter()

    model.train()
    for iter_idx, video_list in enumerate(train_loader, 0):
        optimizer.zero_grad(set_to_none=True)
        losses, topk_acc = model(
            clip_model=clip_model,
            video_list=video_list, 
            action_tokens=action_tokens,
            cur_epoch=curr_epoch,
            tot_epoch=tot_epoch
        )
        losses['final_loss'].backward()
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        ## update meters
        cls_loss_meter.update(losses['cls_loss'].item())
        reg_loss_meter.update(losses['reg_loss'].item())
        action_contrast_loss_meter.update(losses['action_contrast_loss'].item())
        tot_loss_meter.update(losses['final_loss'].item())
        action_contrast_top1_acc_meter.update(topk_acc['action_top1_acc'].item())
        action_contrast_top3_acc_meter.update(topk_acc['action_top3_acc'].item())
        action_contrast_top5_acc_meter.update(topk_acc['action_top5_acc'].item())

        img_graph_rel_loss_meter.update(losses['rel_contrast_loss'].item())
        img_graph_stt_loss_meter.update(losses['stt_contrast_loss'].item())
        effect_rel_loss_meter.update(losses['effect_rel_contrast_loss'].item())
        effect_stt_loss_meter.update(losses['effect_stt_contrast_loss'].item())
    
    if tb_writer is not None:
        tb_writer.add_scalar('train/cls_loss', cls_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/reg_loss', reg_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/tot_loss', tot_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/action_contrast_loss', action_contrast_loss_meter.avg, curr_epoch)

        tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], curr_epoch)
        tb_writer.add_scalar('train/action_top1_acc', action_contrast_top1_acc_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/action_top3_acc', action_contrast_top3_acc_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/action_top5_acc', action_contrast_top5_acc_meter.avg, curr_epoch)

        tb_writer.add_scalar('train/img_graph_relation_loss', img_graph_rel_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/img_graph_state_loss', img_graph_stt_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/effect_relation_loss', effect_rel_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('train/effect_state_loss', effect_stt_loss_meter.avg, curr_epoch)

    prefix = f"[Train]: Epoch {curr_epoch}"
    loss_print = f"tot_loss={tot_loss_meter.avg:.3f}, cls_loss={cls_loss_meter.avg:.3f}, reg_loss={reg_loss_meter.avg:.3f}"    
    acc_print = f"action_top1_acc={action_contrast_top1_acc_meter.avg:.3f}, action_top3_acc={action_contrast_top3_acc_meter.avg:.3f}, action_top5_acc={action_contrast_top5_acc_meter.avg:.3f}"
    print(f"{prefix}\n{loss_print}\n{acc_print}")

    return model_ema

def valid_one_epoch(
    val_loader,
    action_tokens,
    model,
    clip_model,
    curr_epoch,
    tot_epoch,
    tb_writer = None
):
    """Validate the model for one epoch"""
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    action_contrast_loss_meter = AverageMeter()
    tot_loss_meter = AverageMeter()
    action_contrast_top1_acc_meter = AverageMeter()
    action_contrast_top3_acc_meter = AverageMeter()
    action_contrast_top5_acc_meter = AverageMeter()
    img_graph_rel_loss_meter = AverageMeter()
    img_graph_stt_loss_meter = AverageMeter()
    effect_rel_loss_meter = AverageMeter()
    effect_stt_loss_meter = AverageMeter()

    model.eval()
    for iter_idx, video_list in enumerate(val_loader, 0):
        losses, topk_acc = model(
            clip_model=clip_model,
            video_list=video_list, 
            action_tokens=action_tokens,
            cur_epoch=curr_epoch,
            tot_epoch=tot_epoch
        )

        torch.cuda.synchronize()
        ## update meters
        action_contrast_loss_meter.update(losses['action_contrast_loss'].item())
        cls_loss_meter.update(losses['cls_loss'].item())
        reg_loss_meter.update(losses['reg_loss'].item())
        tot_loss_meter.update(losses['final_loss'].item())
        action_contrast_top1_acc_meter.update(topk_acc['action_top1_acc'].item())
        action_contrast_top3_acc_meter.update(topk_acc['action_top3_acc'].item())
        action_contrast_top5_acc_meter.update(topk_acc['action_top5_acc'].item())
        img_graph_rel_loss_meter.update(losses['rel_contrast_loss'].item())
        img_graph_stt_loss_meter.update(losses['stt_contrast_loss'].item())
        effect_rel_loss_meter.update(losses['effect_rel_contrast_loss'].item())
        effect_stt_loss_meter.update(losses['effect_stt_contrast_loss'].item())

    if tb_writer is not None:
        tb_writer.add_scalar('val/cls_loss', cls_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/reg_loss', reg_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/tot_loss', tot_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/action_contrast_loss', action_contrast_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/action_top1_acc', action_contrast_top1_acc_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/action_top3_acc', action_contrast_top3_acc_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/action_top5_acc', action_contrast_top5_acc_meter.avg, curr_epoch)

        tb_writer.add_scalar('val/img_graph_rel_loss', img_graph_rel_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/img_graph_stt_loss', img_graph_stt_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/effect_rel_loss', effect_rel_loss_meter.avg, curr_epoch)
        tb_writer.add_scalar('val/effect_stt_loss', effect_stt_loss_meter.avg, curr_epoch)

    prefix = f"[Valid]: Epoch {curr_epoch}"
    loss_print = f"tot_loss={tot_loss_meter.avg:.3f}, cls_loss={cls_loss_meter.avg:.3f}, reg_loss={reg_loss_meter.avg:.3f}"
    acc_print = f"action_top1_acc={action_contrast_top1_acc_meter.avg:.3f}, action_top3_acc={action_contrast_top3_acc_meter.avg:.3f}, action_top5_acc={action_contrast_top5_acc_meter.avg:.3f}"
    print(f"{prefix}\n{loss_print}\n{acc_print}")
    
    return