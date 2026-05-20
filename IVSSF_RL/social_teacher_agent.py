import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .social_fuzzy_teacher import SocialFuzzyTeacher
except ImportError:
    from social_fuzzy_teacher import SocialFuzzyTeacher


class DoubleQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)

        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        sa = torch.cat([states, actions], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2

    def q1(self, states, actions):
        sa = torch.cat([states, actions], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


class SocialTeacherAgent:
    def __init__(
        self,
        *,
        teacher=None,
        state_dim=5,
        action_dim=2,
        hidden_dim=256,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        replay_capacity=int(1e5),
        policy_delay=2,
        target_noise=(0.015, 0.08),
        target_noise_clip=(0.03, 0.20),
        explore_noise=(0.01, 0.12),
        max_rules=15,
        device=None,
        v_max=0.22,
        w_max=1.5,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.policy_delay = int(policy_delay)
        self.update_count = 0
        self.v_max = float(v_max)
        self.w_max = float(w_max)

        self.teacher = teacher or SocialFuzzyTeacher(
            device=self.device,
            max_rules=max_rules,
            max_speed=v_max,
            max_turn=w_max,
        )
        self.teacher_target = copy.deepcopy(self.teacher)
        self.critic = DoubleQCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_capacity, self.device)

        self.target_noise = torch.tensor(target_noise, dtype=torch.float32, device=self.device)
        self.target_noise_clip = torch.tensor(target_noise_clip, dtype=torch.float32, device=self.device)
        self.explore_noise = torch.tensor(explore_noise, dtype=torch.float32, device=self.device)
        self.action_low = torch.tensor([0.0, -self.w_max], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor([self.v_max, self.w_max], dtype=torch.float32, device=self.device)

    def rebuild_actor_optimizer(self, *, sync_target=True):
        lr = self.actor_optimizer.param_groups[0]["lr"]
        self.actor_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=lr)
        if sync_target:
            self.teacher_target = copy.deepcopy(self.teacher)

    def select_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.teacher(state_tensor).squeeze(0)
            if not deterministic:
                action = action + torch.randn_like(action) * self.explore_noise
            action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.detach().cpu().numpy().astype(np.float32)

    def random_action(self):
        return np.array(
            [
                np.random.uniform(0.0, self.v_max),
                np.random.uniform(-self.w_max, self.w_max),
            ],
            dtype=np.float32,
        )

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None

        self.update_count += 1
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.teacher_target(next_states)
            noise = torch.randn_like(next_actions) * self.target_noise
            noise = torch.max(torch.min(noise, self.target_noise_clip), -self.target_noise_clip)
            next_actions = torch.max(torch.min(next_actions + noise, self.action_high), self.action_low)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": None,
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
        }

        if self.update_count % self.policy_delay == 0:
            actor_actions = self.teacher(states)
            actor_loss = -self.critic.q1(states, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.teacher.enforce_bounds()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.teacher, self.teacher_target)
            metrics["actor_loss"] = float(actor_loss.item())
        return metrics

    def _soft_update(self, source, target):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data)

    def save(self, path):
        payload = {
            "teacher_rulebook": self.teacher.export_rulebook(),
            "teacher_target_rulebook": self.teacher_target.export_rulebook(),
            "critic_state": self.critic.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.teacher = SocialFuzzyTeacher.load_rulebook_from_payload(payload["teacher_rulebook"], device=self.device)
        self.teacher_target = SocialFuzzyTeacher.load_rulebook_from_payload(
            payload["teacher_target_rulebook"], device=self.device
        )
        self.critic.load_state_dict(payload["critic_state"])
        self.critic_target.load_state_dict(payload["critic_target_state"])
        self.rebuild_actor_optimizer(sync_target=False)
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
        self.update_count = int(payload.get("update_count", 0))
