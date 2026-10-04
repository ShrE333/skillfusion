"""FusedPolicy: one network that can run every skill.

    obs_s --adapter[s]--> z --concat(fixed skill code)--> trunk (SHARED, federated) --> head[s] / value[s]

Different skills have different observation/action sizes, so each skill owns a small private
adapter + head; the trunk is what federated learning fuses. Parameter names follow the convention
in ``fl.aggregate`` (``trunk.*`` shared, ``adapter.<skill>.*`` / ``head.<skill>.*`` / ``value.<skill>.*`` private).
"""
from __future__ import annotations

import torch
from torch import nn

from .aggregate import skill_code

HIDDEN = 256
CODE_DIM = 32


class Head(nn.Module):
    def __init__(self, hidden: int, out: int):
        super().__init__()
        self.mu = nn.Linear(hidden, out)
        self.log_std = nn.Parameter(torch.full((out,), -0.5))
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.zeros_(self.mu.bias)


class Trunk(nn.Module):
    """The federated part. Its state_dict keys ('net.0.weight', ...) are what NVFlare exchanges."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(HIDDEN + CODE_DIM, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU())

    def forward(self, x):
        return self.net(x)


class FusedPolicy(nn.Module):
    def __init__(self, skills: dict[str, dict], max_skills: int = 16):
        """skills: {name: {"index": int, "obs_dim": int, "act_dim": int}}"""
        super().__init__()
        self.skills = dict(skills)
        self.trunk = Trunk()
        self.adapter = nn.ModuleDict()
        self.head = nn.ModuleDict()
        self.value = nn.ModuleDict()
        for name, s in skills.items():
            self.add_skill(name, s["index"], s["obs_dim"], s["act_dim"], max_skills)

    def add_skill(self, name: str, index: int, obs_dim: int, act_dim: int, max_skills: int = 16) -> None:
        self.skills[name] = {"index": index, "obs_dim": obs_dim, "act_dim": act_dim}
        self.adapter[name] = nn.Sequential(nn.Linear(obs_dim, HIDDEN), nn.ELU())
        self.head[name] = Head(HIDDEN, act_dim)
        self.value[name] = nn.Linear(HIDDEN, 1)
        self.register_buffer(f"code_{name}", torch.from_numpy(skill_code(index, CODE_DIM, max_skills)),
                             persistent=False)

    def features(self, obs: torch.Tensor, skill: str) -> torch.Tensor:
        z = self.adapter[skill](obs)
        code = getattr(self, f"code_{skill}").expand(obs.shape[0], -1)
        return self.trunk(torch.cat([z, code], dim=-1))

    def forward(self, obs: torch.Tensor, skill: str):
        h = self.features(obs, skill)
        return self.head[skill].mu(h), self.value[skill](h).squeeze(-1), self.head[skill].log_std

    def act(self, obs: torch.Tensor, skill: str, deterministic: bool = False):
        mu, v, log_std = self(obs, skill)
        if deterministic:
            return mu, v
        dist = torch.distributions.Normal(mu, log_std.exp())
        a = dist.sample()
        return a, v, dist.log_prob(a).sum(-1)

    # ---- federated plumbing -------------------------------------------------------------------
    def params_numpy(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

    def load_numpy(self, params: dict, strict: bool = False) -> None:
        sd = {k: torch.as_tensor(v) for k, v in params.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if strict and (missing or unexpected):
            raise KeyError(f"missing={missing} unexpected={unexpected}")

    def trunk_state(self) -> dict:
        return {k: v for k, v in self.params_numpy().items() if k.startswith("trunk.")}
