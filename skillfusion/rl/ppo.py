"""Compact PPO for one skill on an Isaac Lab vector env, on top of ``FusedPolicy``.

Only the *trunk* is federated; ``prox_mu > 0`` adds a FedProx term that keeps the local trunk close
to the global one, which matters because the clients solve very different tasks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..fl.model import FusedPolicy


@dataclass
class PPOCfg:
    horizon: int = 24
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 1.0
    ent_coef: float = 0.005
    max_grad_norm: float = 1.0
    prox_mu: float = 0.0


def _obs(o):
    return o["policy"] if isinstance(o, dict) or hasattr(o, "keys") else o


class SkillTrainer:
    def __init__(self, env, policy: FusedPolicy, skill: str, cfg: PPOCfg | None = None, device: str = "cuda:0"):
        self.env, self.policy, self.skill = env, policy.to(device), skill
        self.cfg, self.device = cfg or PPOCfg(), device
        params = list(policy.trunk.parameters()) + list(policy.adapter[skill].parameters()) \
            + list(policy.head[skill].parameters()) + list(policy.value[skill].parameters())
        self.opt = torch.optim.Adam(params, lr=self.cfg.lr)
        self.obs, _ = env.reset()
        self.obs = _obs(self.obs).to(device)
        self._anchor = None

    def set_global_trunk(self) -> None:
        """Remember the received global trunk (FedProx anchor)."""
        self._anchor = [p.detach().clone() for p in self.policy.trunk.parameters()]

    @torch.no_grad()
    def _rollout(self):
        c, n = self.cfg, self.obs.shape[0]
        O = torch.zeros(c.horizon, n, self.obs.shape[-1], device=self.device)
        A = torch.zeros(c.horizon, n, self.policy.skills[self.skill]["act_dim"], device=self.device)
        LP, V, R, D = (torch.zeros(c.horizon, n, device=self.device) for _ in range(4))
        for t in range(c.horizon):
            a, v, lp = self.policy.act(self.obs, self.skill)
            nxt, rew, term, trunc, _ = self.env.step(a)
            O[t], A[t], LP[t], V[t] = self.obs, a, lp, v
            R[t] = rew.to(self.device)
            D[t] = (term | trunc).float().to(self.device)
            self.obs = _obs(nxt).to(self.device)
        _, last_v, _ = self.policy(self.obs, self.skill)
        adv = torch.zeros_like(R)
        gae = torch.zeros(n, device=self.device)
        for t in reversed(range(c.horizon)):
            nv = last_v if t == c.horizon - 1 else V[t + 1]
            delta = R[t] + c.gamma * nv * (1 - D[t]) - V[t]
            gae = delta + c.gamma * c.lam * (1 - D[t]) * gae
            adv[t] = gae
        return O, A, LP, adv, adv + V, R

    def iterate(self) -> dict:
        c = self.cfg
        O, A, LP, adv, ret, R = self._rollout()
        O, A, LP, adv, ret = (x.flatten(0, 1) for x in (O, A, LP, adv, ret))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = O.shape[0]
        stats = {"loss": 0.0}
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.device)
            for idx in perm.chunk(c.minibatches):
                mu, v, log_std = self.policy(O[idx], self.skill)
                dist = torch.distributions.Normal(mu, log_std.exp())
                lp = dist.log_prob(A[idx]).sum(-1)
                ratio = (lp - LP[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - c.clip, 1 + c.clip) * adv[idx]).mean()
                vf = (v - ret[idx]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + c.vf_coef * vf - c.ent_coef * ent
                if c.prox_mu > 0 and self._anchor is not None:
                    loss = loss + 0.5 * c.prox_mu * sum(
                        (p - a).pow(2).sum() for p, a in zip(self.policy.trunk.parameters(), self._anchor))
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.opt.step()
                stats["loss"] = float(loss)
        stats["mean_reward"] = float(R.mean())
        stats["samples"] = c.horizon * self.obs.shape[0]
        return stats

    def train(self, iterations: int, log_every: int = 10) -> dict:
        total, last = 0, {}
        for i in range(iterations):
            last = self.iterate()
            total += last["samples"]
            if log_every and i % log_every == 0:
                print(f"[{self.skill}] it={i} reward={last['mean_reward']:.3f}", flush=True)
        last["samples"] = total
        return last
