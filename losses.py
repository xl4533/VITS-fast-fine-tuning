import torch 
from torch.nn import functional as F

import commons

#计算对抗训练中生成波形和真是波形在判别器中间特征之间的距离损失
def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):#遍历真是波形和预测波形在判别器每层的特征图
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))#计算L1损失

  return loss * 2 

#判别器的损失
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):#遍历多个子判别器的判别结果
    dr = dr.float()#一个子判别器对真实波形的判别结果
    dg = dg.float()#一个子判别器对生成波形的判别结果
    r_loss = torch.mean((1-dr)**2)#真实波形的判别结果越接近1越好
    g_loss = torch.mean(dg**2)#生成波形的判别结果越接近于0越好
    loss += (r_loss + g_loss)#累加当前子判别器的损失
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses

#生成器的对抗损失，就是将生成器生成的波形经过判别器后的输出与i计算距离损失，L2损失
def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses

#先验分布和后验分布之间KL散度
def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
