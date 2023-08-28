	# -*- coding: utf-8 -*-
import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import librosa
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed." #判断当前设备是否有GPU，仅当有GPU时才可运行

  n_gpus = torch.cuda.device_count()#统计当前设备拥有的GPU数量
  os.environ['MASTER_ADDR'] = 'localhost'#主机地址
  os.environ['MASTER_PORT'] = '8000'#主机端口

  hps = utils.get_hparams()#获取超参数
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))#启动运行进程


def run(rank, n_gpus, hps):
  global global_step
  symbols = hps['symbols']
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  # Use gloo backend on Windows for Pytorch
  dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)#初始化进程组
  torch.manual_seed(hps.train.seed)#设置随机数种子
  torch.cuda.set_device(rank)#设置进程运行在哪个GPU上

  # 音频文件：一个音频文件读取后是一个一维序列，每个元素值表征幅度信息
  # 频谱图/线性谱图：对音频文件进行STFT(短时傅里叶变换)得到，将信号从时域转换到频域
  # mel谱图：是一个音高单位，以使相等的音高距离听起来与听众相等，称为梅尔音阶。mel谱图是频率转换为mel标度的频谱图
  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, symbols)#定义语音数据集的加载器
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)#定义桶排序的样本选取器
  collate_fn = TextAudioSpeakerCollate()#定义构建dataloader时调用的回调函数
  train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)#构建读取训练数据的dataloader，train_dataset为自定义的数据加载器，num_workers是加载数据的进程数，shuffle为是否在训练时打乱数据，
                              # pin_memory表示在将数据传输到GPU之前，是否将数据放置在主机内存中的固定位置，主要用于加速。collate_fn为自定义的将单个数据样本组合成一个批次的方式，
                              # batch_sampler为自定义的如何从数据集中抽取批次样本
  # train_loader = DataLoader(train_dataset, batch_size=hps.train.batch_size, num_workers=2, shuffle=False, pin_memory=True,
  #                           collate_fn=collate_fn)
  if rank == 0:#如果为主进程
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, symbols)#定义验证集的数据加载器
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)#构建读取验证数据的dataloader

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).cuda(rank)#定义语音合成模型的具体结构，并定义放入当前进程所在的显卡上
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)#定义多周期判别器

  # load existing model
  if hps.cont:#是否在最新的模型基础上继续训练
      try:
          _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_latest.pth"), net_g, None)#加载最新的生成器模型
          _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_latest.pth"), net_d, None)#加载最新的判别器模型
          global_step = (epoch_str - 1) * len(train_loader)#根据当前的epoch获取当前的step
      except:
          print("Failed to find latest checkpoint, loading G_0.pth...")
          if hps.train_with_pretrained_model:#当最新的模型不存在或者加载失败时，加载预训练模型
              print("Train with pretrained model...")
              _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/G_0.pth", net_g, None)
              _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/D_0.pth", net_d, None)
          else:
              print("Train without pretrained model...")#当预训练模型也不存在时，从头开始训练
          epoch_str = 1
          global_step = 0
  else:
      if hps.train_with_pretrained_model:#如果不使用最新模型，使用预训练模型训练，加载预训练模型
          print("Train with pretrained model...")
          _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/G_0.pth", net_g, None)
          _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/D_0.pth", net_d, None)
      else:#最新模型和预训练模型都不存在时，从头开始训练
          print("Train without pretrained model...")
      epoch_str = 1
      global_step = 0
  # freeze all other layers except speaker embedding
  for p in net_g.parameters():#获取生成器模型中所有的参数，并将其置为可求梯度的状态
      p.requires_grad = True
  for p in net_d.parameters():#获取判别器模型中所有的参数，并将其置为可求梯度的状态
      p.requires_grad = True
  # for p in net_d.parameters():
  #     p.requires_grad = False
  # net_g.emb_g.weight.requires_grad = True
  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)#对生成器模型使用AdamW优化函数（在Adam优化器的基础上加上L2正则的改进版本），并设置优化器所需的学习率等超参数
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)#对判别器模型使用AdamW优化函数（在Adam优化器的基础上加上L2正则的改进版本），并设置优化器所需的学习率等超参数
  # optim_d = None
  net_g = DDP(net_g, device_ids=[rank])#使用DistributedDataParallel将模型参数分散到多个GPU上进行训练，并实现梯度汇总和参数更新，每个进程具有自己的optimizer，并独立完成所有的优化步骤，进程内与一般的训练无异。
  net_d = DDP(net_d, device_ids=[rank])#在各进程梯度计算完成之后，各进程需要将梯度进行汇总平均，然后再由rank=0的进程将其broadcast到所有进程。之后，各进程用该梯度来独立的更新参数。相较于DataParallel，DistributedDataParallel传输的数据量更少，因此速度更快，效率更高

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)#ExponentialLR是一种学习率调整方法，按指数衰减调整学习率，调整公式为lr=lr*gamma{{e}}，其中e代表指数，gamma代表调整倍数
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)#这种方法适合后期调试使用，观察loss曲线，通过参数milestones给定衰减的epoch列表，可以在指定的epoch时期进行衰减

  scaler = GradScaler(enabled=hps.train.fp16_run)#GradScaler是PyTorch中使用混合精度训练时用于缩放梯度的模块，可以避免数值下溢或溢出的问题，同时保持足够的精度以避免模型的性能下降。在每次计算损失时，调用GradScaler的scale方法对损失进行缩放，再使用缩放后的损失进行反向传播、梯度更新等操作，最后调用GradScaler的update方法更新GradScaler对象的内部状态

  for epoch in range(epoch_str, hps.train.epochs + 1):#在每个epoch中对模型进行训练
    if rank==0:#如果为主进程，则进行对验证集的验证
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()#调整生成器模型学习率
    scheduler_d.step()#调整判别器模型学习率


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets#获取生成器及判别器模型实例
  optim_g, optim_d = optims#获取生成器及判别器对应的优化器实例
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders#获取训练集及验证集的dataloader
  if writers is not None:
    writer, writer_eval = writers

  # train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()#设置生成器模型为训练模式，调用net_g.train()后，模型会关闭dropout和batch normalization等层的训练，并开启参数更新。
  net_d.train()#设置判别器模型为训练模式
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(train_loader)):#根据dataloader中设置的collate_fn函数拿取数据
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)#将pad后的文本及其长度，放置到gpu上，并设置为非阻塞操作，即将张量移动到GPU上的过程中，不会阻塞程序的其他执行。
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)#将pad后的语音频谱数据及其长度，放置到gpu上，并设置为非阻塞操作
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)#将pad后的语音数据及其长度，放置到gpu上，并设置为非阻塞操作
    speakers = speakers.cuda(rank, non_blocking=True)#将说话人的编号放置到gpu上，并设置为非阻塞操作

    with autocast(enabled=hps.train.fp16_run):#with autocast是PyTorch中使用自动混合精度训练的上下文管理器。它允许在训练过程中自动地将模型参数从浮点数转换为较低精度的整数，同时将梯度值保持为浮点数。这样可以利用GPU的Tensor Core硬件加速，提高训练速度并降低显存消耗。
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)#执行生成器模型的推理操作

      mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin,
          hps.data.mel_fmax)#将音频频谱转换为梅尔频谱，便于后续计算L_recon
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)#以ids_slice作为指导，采样对应窗口的mel谱图作为target
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax
      )#从生成的音频波形y_hat中提取对应的mel谱图

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice，从完成的音频数据中以ids_slice获取对应窗口部分的音频数据，判别器判别时需要真实波形数据

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())#y_d_hat_r, y_d_hat_g记录所有子判别器对batch中真实波形y和生成波形y_hat的判别结果
      with autocast(enabled=False):#损失的计算不进行banjingdu计算
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)#判别器的损失
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)#梯度剪裁前先进行unscale
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)#梯度剪裁
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)#将生成的波形和真实波形分别送入到判别器中，希望两者在判别器的中间特征尽可能保持一致，即论文中的L_{fm}，需要fmap_r，fmap_g进行计算
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())#时间预测器loss，直接求和
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel#重构loss，论文中系数c_mel为45
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl#计算模型基于文本学习到的先验分布和从音频线型谱图中学习到的后验分布之间的KL散度，系数c_mel为1

        loss_fm = feature_loss(fmap_r, fmap_g)#feature map的loss
        loss_gen, losses_gen = generator_loss(y_d_hat_g)#生成器的对抗loss
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl#总的损失，优化该损失即可进行训练
    #生成器更新
    optim_g.zero_grad()#清空生成器模块的优化器梯度缓冲区
    scaler.scale(loss_gen_all).backward()#scaler是一个缩放器对象，用于对损失进行缩放。确保梯度不会因为数值过大或过小而出现不稳定的情况。并执行反向传播计算梯度
    scaler.unscale_(optim_g)#unscale_方法的作用是去除之前对梯度的缩放，将其恢复到原始的梯度值。对优化器对象的梯度进行去缩放，然后使用原始的梯度值进行权重更新。
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)#将生成器的参数和梯度传递给clip_grad_value_函数进行裁剪操作。裁剪梯度的目的是为了控制梯度的幅度，防止梯度爆炸或梯度消失的问题。
    scaler.step(optim_g)#对优化器进行步长操作，使用缩放后的梯度进行权重更新
    scaler.update()#更新模型权重

    if rank==0:#主进程上的loss打印，记录和模型验证，保存
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])#输出的loss依次为：判别器loss，生成器loss，中间特征loss，重构loss，时长loss，kl散度loss。后接当前step和学习率lr

        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}#记录损失和梯度
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        # 以图像的形式记录mel谱图和对齐信息
        image_dict = {
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        #调用定义的tensorboard的writer记录上述信息
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0 and global_step!=0:
        evaluate(hps, net_g, eval_loader, writer_eval)#验证
        #保存生成器和判别器的参数
        utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "G_latest.pth"))
        
        utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "D_latest.pth"))
        # save to google drive
        if os.path.exists("/content/drive/MyDrive/"):
            utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                  os.path.join("/content/drive/MyDrive/", "G_latest.pth"))

            utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                  os.path.join("/content/drive/MyDrive/", "D_latest.pth"))
        if hps.preserved > 0:
          utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
          utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
          old_g = utils.oldest_checkpoint_path(hps.model_dir, "G_[0-9]*.pth",
                                               preserved=hps.preserved)  # Preserve 4 (default) historical checkpoints.
          old_d = utils.oldest_checkpoint_path(hps.model_dir, "D_[0-9]*.pth", preserved=hps.preserved)
          if os.path.exists(old_g):
            print(f"remove {old_g}")
            os.remove(old_g)
          if os.path.exists(old_d):
            print(f"remove {old_d}")
            os.remove(old_d)
          if os.path.exists("/content/drive/MyDrive/"):
              utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                    os.path.join("/content/drive/MyDrive/", "G_{}.pth".format(global_step)))
              utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                    os.path.join("/content/drive/MyDrive/", "D_{}.pth".format(global_step)))
              old_g = utils.oldest_checkpoint_path("/content/drive/MyDrive/", "G_[0-9]*.pth",
                                                   preserved=hps.preserved)  # Preserve 4 (default) historical checkpoints.
              old_d = utils.oldest_checkpoint_path("/content/drive/MyDrive/", "D_[0-9]*.pth", preserved=hps.preserved)
              if os.path.exists(old_g):
                  print(f"remove {old_g}")
                  os.remove(old_g)
              if os.path.exists(old_d):
                  print(f"remove {old_d}")
                  os.remove(old_d)
    global_step += 1
    if epoch > hps.max_epochs:
        print("Maximum epoch reached, closing training...")
        exit()

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval() #验证模式，验证模式下模型中的某些层将不起作用（比如Dropout层和BatchNorm的参数）
    with torch.no_grad():#这个上下文管理器会临时关闭梯度计算。在Tensor上调用.detach()或.requires_grad为False会确保该Tensor不会计算梯度。
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)#基于文本生成音频
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length
      #提取真实的mel谱图
      mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax)
      #从预测的音频中提取mel谱图
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})
    #记录信息
    utils.summarize(
      writer=writer_eval,
      global_step=global_step,
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
  main()
