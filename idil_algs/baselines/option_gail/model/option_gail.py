import torch
import torch.nn.functional as F
import numpy as np
from .option_policy import OptionPolicy, Policy, MoEPolicy
from .option_discriminator import (OptionDiscriminator, Discriminator,
                                   MoEDiscriminator)
from ..utils.model_util import conv_nn_input
from omegaconf import DictConfig


class GAIL(torch.nn.Module):

  def __init__(self, config: DictConfig, dim_s=2, dim_a=2):
    super(GAIL, self).__init__()
    self.dim_a = dim_a
    self.dim_s = dim_s
    self.device = torch.device(config.device)
    self.mini_bs = config.mini_batch_size
    lr = config.optimizer_lr_discriminator

    self.discriminator = Discriminator(config, dim_s=dim_s, dim_a=dim_a)
    self.policy = Policy(config, dim_s=self.dim_s, dim_a=self.dim_a)

    self.optim = torch.optim.Adam(self.discriminator.parameters(),
                                  lr=lr,
                                  weight_decay=1.e-3)

    self.to(self.device)

  def gail_reward(self, s, a):
    d = self.discriminator.get_unnormed_d(s, a)
    reward = -F.logsigmoid(d)
    return reward

  def step_original_gan(self, sample_sar, demo_sar, n_step=10):
    sp = torch.cat([s for s, a, r in sample_sar], dim=0)
    se = torch.cat([s for s, a, r in demo_sar], dim=0)
    ap = torch.cat([a for s, a, r in sample_sar], dim=0)
    ae = torch.cat([a for s, a, r in demo_sar], dim=0)
    tp = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device)
    te = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device)

    for _ in range(n_step):
      inds = torch.randperm(sp.size(0), device=self.device)
      for ind_p in inds.split(self.mini_bs):
        sp_b, ap_b, tp_b = sp[ind_p], ap[ind_p], tp[:ind_p.size(0)]
        ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
        se_b, ae_b, te_b = se[ind_e], ae[ind_e], te[:ind_p.size(0)]

        s_array = torch.cat((sp_b, se_b), dim=0)
        a_array = torch.cat((ap_b, ae_b), dim=0)
        t_array = torch.cat((tp_b, te_b), dim=0)
        for _ in range(3):
          src = self.discriminator.get_unnormed_d(s_array, a_array)
          loss = F.binary_cross_entropy_with_logits(src, t_array)
          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

  def step_wgail_gp(self, sample_sar, demo_sar, n_step=10):
    sp = torch.cat([s for s, a, r in sample_sar], dim=0)
    se = torch.cat([s for s, a, r in demo_sar], dim=0)
    ap = torch.cat([a for s, a, r in sample_sar], dim=0)
    ae = torch.cat([a for s, a, r in demo_sar], dim=0)

    for _ in range(n_step):
      inds = torch.randperm(sp.size(0), device=self.device)
      for ind_p in inds.split(self.mini_bs):
        sp_b, ap_b = sp[ind_p], ap[ind_p]
        ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
        se_b, ae_b = se[ind_e], ae[ind_e]

        for _ in range(3):
          Wasserstein_D = self.discriminator.get_unnormed_d(sp_b, ap_b).mean() \
                          - self.discriminator.get_unnormed_d(se_b, ae_b).mean()

          alpha = torch.rand(sp_b.size(0),
                             1,
                             dtype=torch.float32,
                             device=self.device)

          inter_s = alpha * se_b + (1 - alpha) * sp_b
          inter_a = alpha * ae_b + (1 - alpha) * ap_b

          gradient_penalty = self.discriminator.gradient_penalty(
              inter_s, inter_a)
          loss = Wasserstein_D + gradient_penalty
          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

  def step(self, sample_sar, demo_sar, n_step=10):
    return self.step_original_gan(sample_sar, demo_sar, n_step)

  def convert_demo(self, demo_sa):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for s_array, a_array in demo_sa:
        r_fake_array = self.gail_reward(s_array, a_array)
        out_sample.append((s_array, a_array, r_fake_array))
        r_sum_avg += r_fake_array.sum().item()
      r_sum_avg /= len(demo_sa)
    return out_sample, r_sum_avg

  def convert_sample(self, sample_sar):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for s_array, a_array, r_real_array in sample_sar:
        r_fake_array = self.gail_reward(s_array, a_array)
        out_sample.append((s_array, a_array, r_fake_array))
        r_sum_avg += r_real_array.sum().item()
      r_sum_avg /= len(sample_sar)
    return out_sample, r_sum_avg


class OptionGAIL(torch.nn.Module):

  def __init__(self,
               config: DictConfig,
               dim_s=2,
               dim_a=2,
               discrete_s=False,
               discrete_a=False):
    super(OptionGAIL, self).__init__()
    self.dim_a = dim_a
    self.dim_s = dim_s
    self.dim_c = config.dim_c
    self.with_c = config.use_c_in_discriminator
    self.mini_bs = config.mini_batch_size
    self.use_d_info_gail = config.use_d_info_gail
    self.device = torch.device(config.device)

    self.discriminator = OptionDiscriminator(config,
                                             dim_s=dim_s,
                                             dim_a=dim_a,
                                             discrete_s=discrete_s,
                                             discrete_a=discrete_a)
    self.policy = OptionPolicy(config,
                               dim_s=self.dim_s,
                               dim_a=self.dim_a,
                               discrete_s=discrete_s,
                               discrete_a=discrete_a)

    self.optim = torch.optim.Adam(self.discriminator.parameters(),
                                  weight_decay=1.e-3)

    self.to(self.device)

    # NOTE: for compatibility
    self.PREV_LATENT = self.dim_c
    self.PREV_ACTION = float("nan")

  # NOTE: for compatibility
  def choose_policy_action(self, state, option, sample=False):
    dim_s = 1 if self.policy.discrete_s else self.dim_s
    state = torch.tensor(state).to(self.device).reshape(-1, dim_s)
    option = torch.tensor(option).to(self.device).reshape(-1, 1)
    with torch.no_grad():
      return self.policy.sample_action(state, option,
                                       not sample)[0].cpu().numpy()

  def choose_mental_state(self, state, prev_option, sample=False):
    dim_s = 1 if self.policy.discrete_s else self.dim_s
    state = torch.tensor(state).to(self.device).reshape(-1, dim_s)
    prev_option = torch.tensor(prev_option).to(self.device).reshape(-1, 1)
    with torch.no_grad():
      return self.policy.sample_option(state, prev_option,
                                       not sample)[0].cpu().numpy()

  def original_gail_reward(self, s, c_1, a, c):
    d = self.discriminator.get_unnormed_d(s, c_1, a, c)
    reward = -F.logsigmoid(d)
    return reward

  def d_info_gail_reward(self, s, c_1, a, c):
    d = self.discriminator.get_unnormed_d(s, c_1, a, c)
    # la, lb, _, _, _ = self.policy.log_alpha_beta(s, a)
    # logpc = (la + lb).log_softmax(dim=-1).gather(dim=-1, index=c)
    reward = -F.logsigmoid(d)
    reward += 0.001 * self.policy.log_prob_option(s, c_1, c)
    return reward

  def gail_reward(self, s, c_1, a, c):
    if not self.use_d_info_gail:
      return self.original_gail_reward(s, c_1, a, c)
    else:
      return self.d_info_gail_reward(s, c_1, a, c)

  def step_original_gan(self, sample_scar, demo_scar, n_step=10):
    sp = torch.cat([s for s, c, a, r in sample_scar], dim=0)
    se = torch.cat([s for s, c, a, r in demo_scar], dim=0)
    c_1p = torch.cat([c[:-1] for s, c, a, r in sample_scar], dim=0)
    c_1e = torch.cat([c[:-1] for s, c, a, r in demo_scar], dim=0)
    cp = torch.cat([c[1:] for s, c, a, r in sample_scar], dim=0)
    ce = torch.cat([c[1:] for s, c, a, r in demo_scar], dim=0)
    ap = torch.cat([a for s, c, a, r in sample_scar], dim=0)
    ae = torch.cat([a for s, c, a, r in demo_scar], dim=0)
    tp = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device)
    te = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device)

    for _ in range(n_step):
      inds = torch.randperm(sp.size(0), device=self.device)
      for ind_p in inds.split(self.mini_bs):
        sp_b, cp_1b, ap_b, cp_b, tp_b = sp[ind_p], c_1p[ind_p], ap[ind_p], cp[
            ind_p], tp[:ind_p.size(0)]
        ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
        se_b, ce_1b, ae_b, ce_b, te_b = se[ind_e], c_1e[ind_e], ae[ind_e], ce[
            ind_e], te[:ind_p.size(0)]

        s_array = torch.cat((sp_b, se_b), dim=0)
        a_array = torch.cat((ap_b, ae_b), dim=0)
        c_1array = torch.cat((cp_1b, ce_1b), dim=0)
        c_array = torch.cat((cp_b, ce_b), dim=0)
        t_array = torch.cat((tp_b, te_b), dim=0)
        for _ in range(3):
          src = self.discriminator.get_unnormed_d(s_array, c_1array, a_array,
                                                  c_array)
          loss = F.binary_cross_entropy_with_logits(src, t_array)
          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

  def step(self, sample_sar, demo_sar, n_step=10):
    return self.step_original_gan(sample_sar, demo_sar, n_step)

  def convert_demo(self, demo_sa, demo_labels):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for idx, item in enumerate(demo_sa):
        s_array, a_array = item
        if self.with_c:
          if demo_labels[idx] is None:
            c_array, _ = self.policy.viterbi_path(s_array, a_array)
          else:
            c_array = demo_labels[idx]
        else:
          c_array = torch.zeros(s_array.size(0) + 1,
                                1,
                                dtype=torch.long,
                                device=self.device)
        r_array = self.gail_reward(s_array, c_array[:-1], a_array, c_array[1:])
        out_sample.append((s_array, c_array, a_array, r_array))
        r_sum_avg += r_array.sum().item()
      r_sum_avg /= len(demo_sa)
    return out_sample, r_sum_avg

  def convert_sample(self, sample_scadr):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for s_array, c_array, a_array, success, r_real_array in sample_scadr:
        r_fake_array = self.gail_reward(s_array, c_array[:-1], a_array,
                                        c_array[1:])
        out_sample.append((s_array, c_array, a_array, r_fake_array))
        r_sum_avg += r_real_array.sum().item()
      r_sum_avg /= len(sample_scadr)
    return out_sample, r_sum_avg

  def infer_mental_states(self, s_array, a_array):
    # s_array = conv_nn_input(np.array(s_array), self.policy.discrete_s,
    #                         self.dim_s, self.device)
    # a_array = conv_nn_input(np.array(a_array), self.policy.discrete_a,
    #                         self.dim_a, self.device)

    c_array, log_prob_traj = self.policy.viterbi_path(s_array, a_array)
    return c_array[1:].cpu().numpy(), log_prob_traj.cpu().numpy()


class MoEGAIL(torch.nn.Module):

  def __init__(self, config: DictConfig, dim_s=2, dim_a=2):
    super(MoEGAIL, self).__init__()
    self.dim_a = dim_a
    self.dim_s = dim_s
    self.dim_c = config.dim_c
    self.mini_bs = config.mini_batch_size
    self.device = torch.device(config.device)

    self.discriminator = MoEDiscriminator(config, dim_s=dim_s)
    self.policy = MoEPolicy(config, dim_s=self.dim_s, dim_a=self.dim_a)

    self.optim = torch.optim.Adam(list(self.discriminator.parameters()) +
                                  self.policy.get_param(low_policy=False),
                                  weight_decay=1.e-3)
    self.to(self.device)

  def original_gail_reward(self, s):
    # N x 1
    d = self.discriminator.get_unnormed_d(s)  # N x C x 1
    weight = self.policy.mix(s)  # N x C x 1
    reward = -(F.logsigmoid(d) * weight).sum(dim=-2)
    return reward

  def gail_reward(self, s):
    # N x 1
    return self.original_gail_reward(s)

  def step_original_gan(self, sample_sar, demo_sar, n_step=10):
    sp = torch.cat([s for s, a, r in sample_sar], dim=0)
    se = torch.cat([s for s, a, r in demo_sar], dim=0)
    tp = torch.ones(self.mini_bs,
                    self.dim_c,
                    dtype=torch.float32,
                    device=self.device)
    te = torch.zeros(self.mini_bs,
                     self.dim_c,
                     dtype=torch.float32,
                     device=self.device)

    for _ in range(n_step):
      inds = torch.randperm(sp.size(0), device=self.device)
      for ind_p in inds.split(self.mini_bs):
        sp_b, tp_b = sp[ind_p], tp[:ind_p.size(0)]
        ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
        se_b, te_b = se[ind_e], te[:ind_p.size(0)]

        s_array = torch.cat((sp_b, se_b), dim=0)
        t_array = torch.cat((tp_b, te_b), dim=0)
        for _ in range(3):
          w_array = self.policy.mix(s_array).squeeze(dim=-1)  # N x C
          src = self.discriminator.get_unnormed_d(s_array).squeeze(
              dim=-1)  # N x C
          loss = (F.binary_cross_entropy_with_logits(
              src, t_array, reduction='none') * w_array).mean(dim=-2).sum()
          lb = (w_array.mean(dim=0) - 0.5).square().sum()
          le = (w_array.mean(dim=1) - 0.5).square().mean()
          lv = -(
              (w_array - w_array.mean(dim=0, keepdim=True)).mean(dim=0)).sum()
          loss = loss + 0.01 * lb + 10.0 * le + 1.0 * lv
          self.optim.zero_grad()
          loss.backward()
          self.optim.step()

  def step(self, sample_sar, demo_sar, n_step=10):
    return self.step_original_gan(sample_sar, demo_sar, n_step)

  def convert_demo(self, demo_sa):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for s_array, a_array in demo_sa:
        r_array = self.gail_reward(s_array)
        out_sample.append((s_array, a_array, r_array))
        r_sum_avg += r_array.sum().item()
      r_sum_avg /= len(demo_sa)
    return out_sample, r_sum_avg

  def convert_sample(self, sample_sar):
    with torch.no_grad():
      out_sample = []
      r_sum_avg = 0.
      for s_array, a_array, r_real_array in sample_sar:
        r_fake_array = self.gail_reward(s_array)
        out_sample.append((s_array, a_array, r_fake_array))
        r_sum_avg += r_real_array.sum().item()
      r_sum_avg /= len(sample_sar)
    return out_sample, r_sum_avg
