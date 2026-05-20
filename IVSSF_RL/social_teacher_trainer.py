import torch
import torch.nn.functional as F

try:
    from .social_fuzzy_teacher import SocialFuzzyTeacher
except ImportError:
    from social_fuzzy_teacher import SocialFuzzyTeacher


class SocialTeacherTrainer:
    def __init__(
        self,
        teacher=None,
        *,
        device=None,
        lr=1e-3,
        uncovered_threshold=0.12,
        expand_cooldown_steps=250,
        max_rules=100,
        merge_interval=0,
        merge_threshold=0.12,
        action_merge_threshold=0.12,
        sigma_regularizer=1e-4,
    ):
        self.teacher = teacher or SocialFuzzyTeacher(
            device=device,
            max_rules=max_rules,
            uncovered_threshold=uncovered_threshold,
        )
        self.device = self.teacher.device
        self.expand_cooldown_steps = max(1, int(expand_cooldown_steps))
        self.merge_interval = int(merge_interval)
        self.merge_threshold = float(merge_threshold)
        self.action_merge_threshold = float(action_merge_threshold)
        self.sigma_regularizer = float(sigma_regularizer)

        self.optimizer = torch.optim.Adam(self.teacher.parameters(), lr=lr)
        self.train_step_count = 0
        self.expand_cooldown_remaining = 0

    def _reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.teacher.parameters(),
            lr=self.optimizer.param_groups[0]["lr"],
        )

    @staticmethod
    def _merge_changed(merge_info):
        if not merge_info:
            return False
        return bool(merge_info.get("fuzzy_changed", False)) or int(merge_info.get("merged_rule_count", 0)) > 0

    def _ensure_tensor(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _sigma_loss(self):
        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for var_name in self.teacher.variable_names:
            sigmas = getattr(self.teacher, f"{var_name}_sigmas")
            loss = loss + (sigmas ** 2).mean()
        return loss

    def train_step(self, states, target_actions):
        states = self._ensure_tensor(states)
        target_actions = self._ensure_tensor(target_actions)

        pred_actions = self.teacher(states)
        mse_loss = F.mse_loss(pred_actions, target_actions)
        reg_loss = self.sigma_regularizer * self._sigma_loss()
        loss = mse_loss + reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.teacher.enforce_bounds()
        self.train_step_count += 1

        merge_info = None
        if self.merge_interval > 0 and self.train_step_count % self.merge_interval == 0:
            merge_info = self.teacher.merge_rules(
                merge_threshold=self.merge_threshold,
                action_merge_threshold=self.action_merge_threshold,
            )
            if self._merge_changed(merge_info):
                self._reset_optimizer()

        return {
            "loss": float(loss.item()),
            "mse_loss": float(mse_loss.item()),
            "reg_loss": float(reg_loss.item()),
            "merge_info": merge_info,
            "rule_count": int(self.teacher.rule_indices.shape[0]),
        }

    def observe_and_adapt(
        self,
        state,
        *,
        bypass_cooldown=False,
        uncovered_threshold=None,
        expand_frozen=False,
    ):
        uncovered_info = self.teacher.check_uncovered_state(state, threshold=uncovered_threshold)

        event = None
        expansion_blocked = False
        if expand_frozen:
            expansion_blocked = bool(uncovered_info["uncovered"])
        elif not bypass_cooldown and self.expand_cooldown_remaining > 0:
            self.expand_cooldown_remaining -= 1

        can_expand = (not expand_frozen) and (bypass_cooldown or self.expand_cooldown_remaining == 0)
        if can_expand and uncovered_info["uncovered"]:
            event = self.teacher.add_rule(state, uncovered_info=uncovered_info)
            if event is not None:
                self.expand_cooldown_remaining = 0 if bypass_cooldown else self.expand_cooldown_steps
                if bool(event.get("structure_changed", False)):
                    self._reset_optimizer()
        elif uncovered_info["uncovered"]:
            expansion_blocked = True

        return {
            "uncovered_info": uncovered_info,
            "expand_event": event,
            "expand_cooldown_remaining": int(self.expand_cooldown_remaining),
            "bypass_cooldown": bool(bypass_cooldown),
            "expand_frozen": bool(expand_frozen),
            "expansion_blocked": bool(expansion_blocked),
            "rule_count": int(self.teacher.rule_indices.shape[0]),
        }

    def maybe_merge(self, current_step=None):
        if self.merge_interval <= 0:
            return None
        step = self.train_step_count if current_step is None else int(current_step)
        if step <= 0 or step % self.merge_interval != 0:
            return None
        merge_info = self.teacher.merge_rules(
            merge_threshold=self.merge_threshold,
            action_merge_threshold=self.action_merge_threshold,
        )
        if self._merge_changed(merge_info):
            self._reset_optimizer()
        return merge_info

    def force_merge(self, rounds=3):
        rounds = max(1, int(rounds))
        initial_rule_count = int(self.teacher.rule_indices.shape[0])
        merge_details = []
        total_merged_rule_count = 0
        duplicate_merged_rule_count = 0
        generalized_rule_count = 0
        fuzzy_changed = False
        changed = False

        for _ in range(rounds):
            merge_info = self.teacher.merge_rules(
                merge_threshold=self.merge_threshold,
                action_merge_threshold=self.action_merge_threshold,
            )
            merge_details.append(merge_info)
            total_merged_rule_count += int(merge_info.get("merged_rule_count", 0))
            duplicate_merged_rule_count += int(merge_info.get("duplicate_merged_rule_count", 0))
            generalized_rule_count += int(merge_info.get("generalized_rule_count", 0))
            fuzzy_changed = fuzzy_changed or bool(merge_info.get("fuzzy_changed", False))
            changed = changed or self._merge_changed(merge_info)

        if changed:
            self._reset_optimizer()

        final_rule_count = int(self.teacher.rule_indices.shape[0])
        return {
            "rounds": rounds,
            "details": merge_details,
            "fuzzy_changed": bool(fuzzy_changed),
            "merged_rule_count": int(total_merged_rule_count),
            "duplicate_merged_rule_count": int(duplicate_merged_rule_count),
            "generalized_rule_count": int(generalized_rule_count),
            "initial_rule_count": initial_rule_count,
            "rule_count": final_rule_count,
            "rule_reduction": int(initial_rule_count - final_rule_count),
            "optimizer_reset_needed": bool(changed),
        }

    def export_rule_texts(self):
        return self.teacher.export_rule_texts(export_semantics=True)

    def save_checkpoint(self, path):
        payload = {
            "teacher_rulebook": self.teacher.export_rulebook_with_semantics(export_semantics=False),
            "optimizer_state": self.optimizer.state_dict(),
            "train_step_count": self.train_step_count,
            "expand_cooldown_steps": self.expand_cooldown_steps,
            "expand_cooldown_remaining": self.expand_cooldown_remaining,
            "merge_interval": self.merge_interval,
            "merge_threshold": self.merge_threshold,
            "action_merge_threshold": self.action_merge_threshold,
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, *, device=None):
        payload = torch.load(path, map_location=device or "cpu")
        teacher = SocialFuzzyTeacher.load_rulebook_from_payload(payload["teacher_rulebook"], device=device)
        trainer = cls(
            teacher=teacher,
            device=device,
            expand_cooldown_steps=payload.get("expand_cooldown_steps", 250),
            merge_interval=payload.get("merge_interval", 0),
            merge_threshold=payload.get("merge_threshold", 0.12),
            action_merge_threshold=payload.get("action_merge_threshold", 0.12),
        )
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.train_step_count = int(payload.get("train_step_count", 0))
        trainer.expand_cooldown_remaining = int(payload.get("expand_cooldown_remaining", 0))
        return trainer
