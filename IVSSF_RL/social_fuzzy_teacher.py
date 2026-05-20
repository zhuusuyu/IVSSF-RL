from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

try:
    from .social_rulebook import (
        COVERAGE_SPECIFICITY_ALPHA,
        DONT_CARE_INDEX,
        RULE_FORMAT,
        VARIABLE_NAMES,
        build_export_fuzzy_sets,
        build_initial_rulebook,
        generate_rule_texts,
        load_rulebook as load_rulebook_file,
        save_rulebook as save_rulebook_file,
    )
except ImportError:
    from social_rulebook import (
        COVERAGE_SPECIFICITY_ALPHA,
        DONT_CARE_INDEX,
        RULE_FORMAT,
        VARIABLE_NAMES,
        build_export_fuzzy_sets,
        build_initial_rulebook,
        generate_rule_texts,
        load_rulebook as load_rulebook_file,
        save_rulebook as save_rulebook_file,
    )


DIMENSION_PRIORITY = (
    "front_clearance",
    "left_social_pressure",
    "right_social_pressure",
    "goal_heading",
    "goal_dist",
)

EXPLANATION_PRIORITY = (
    "front_clearance",
    "goal_heading",
    "left_social_pressure",
    "right_social_pressure",
    "goal_dist",
)
EXPLANATION_PRIORITY_INDEX = {
    var_name: idx for idx, var_name in enumerate(EXPLANATION_PRIORITY)
}

VALUE_BOUNDS = {
    "goal_dist": (0.0, 1.0),
    "goal_heading": (-1.0, 1.0),
    "front_clearance": (0.0, 1.0),
    "left_social_pressure": (0.0, 1.0),
    "right_social_pressure": (0.0, 1.0),
}


def gaussian_membership(x_value, centers, sigmas):
    return torch.exp(-0.5 * ((x_value - centers) / (sigmas + 1e-6)) ** 2)


class SocialFuzzyTeacher(nn.Module):
    def __init__(
        self,
        *,
        device=None,
        max_rules=15,
        uncovered_threshold=0.12,
        min_sigma=0.06,
        max_sigma=0.45,
        max_speed=0.22,
        max_turn=1.5,
        coverage_specificity_alpha=COVERAGE_SPECIFICITY_ALPHA,
        added_rule_mismatch_threshold=0.35,
        added_rule_min_care_dims=2,
        selected_variable_mismatch_threshold=0.35,
        new_rule_min_dims=2,
        new_rule_max_dims=4,
        confidence_margin_threshold=0.15,
        new_fuzzy_set_min_gap_ratio=0.75,
        new_fuzzy_set_min_abs_gap=0.08,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_rules = int(max_rules)
        self.uncovered_threshold = float(uncovered_threshold)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.max_speed = float(max_speed)
        self.max_turn = float(max_turn)
        self.coverage_specificity_alpha = float(coverage_specificity_alpha)
        self.added_rule_mismatch_threshold = float(added_rule_mismatch_threshold)
        self.added_rule_min_care_dims = max(1, int(added_rule_min_care_dims))
        self.selected_variable_mismatch_threshold = float(selected_variable_mismatch_threshold)
        self.new_rule_min_dims = max(2, int(new_rule_min_dims))
        self.new_rule_max_dims = min(
            max(2, len(VARIABLE_NAMES) - 1),
            max(self.new_rule_min_dims, int(new_rule_max_dims)),
        )
        self.confidence_margin_threshold = float(confidence_margin_threshold)
        self.new_fuzzy_set_min_gap_ratio = float(new_fuzzy_set_min_gap_ratio)
        self.new_fuzzy_set_min_abs_gap = float(new_fuzzy_set_min_abs_gap)

        self.rulebook = build_initial_rulebook()
        self.variable_names = tuple(self.rulebook["variable_names"])
        self.rule_origins = [rule.get("origin", "seed") for rule in self.rulebook["rules"]]

        for var_name in self.variable_names:
            fuzzy_cfg = self.rulebook["fuzzy_sets"][var_name]
            centers = torch.tensor(fuzzy_cfg["centers"], dtype=torch.float32, device=self.device)
            sigmas = torch.tensor(fuzzy_cfg["sigmas"], dtype=torch.float32, device=self.device)
            setattr(self, f"{var_name}_centers", nn.Parameter(centers))
            setattr(self, f"{var_name}_sigmas", nn.Parameter(sigmas))

        antecedents = [rule["antecedent"] for rule in self.rulebook["rules"]]
        consequents = [rule["action"] for rule in self.rulebook["rules"]]
        self.register_buffer(
            "rule_indices",
            torch.tensor(antecedents, dtype=torch.long, device=self.device),
        )
        self.consequents = nn.Parameter(
            torch.tensor(consequents, dtype=torch.float32, device=self.device)
        )
        self.rule_texts = generate_rule_texts(self.rulebook["fuzzy_sets"], self.rulebook["rules"])
        self.to(self.device)

    def _ensure_batch(self, x_state):
        if isinstance(x_state, np.ndarray):
            tensor = torch.as_tensor(x_state, dtype=torch.float32, device=self.device)
        else:
            tensor = x_state.to(self.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _get_variable_params(self, var_name):
        centers = getattr(self, f"{var_name}_centers")
        sigmas = getattr(self, f"{var_name}_sigmas")
        return centers, sigmas

    def _best_matching_label_index(self, var_name, value):
        centers, sigmas = self._get_variable_params(var_name)
        value_tensor = torch.tensor([[float(value)]], dtype=torch.float32, device=self.device)
        memberships = gaussian_membership(value_tensor, centers.view(1, -1), sigmas.view(1, -1)).squeeze(0)
        return int(torch.argmax(memberships).item())

    def _state_signature_entry(self, var_name, value):
        centers, sigmas = self._get_variable_params(var_name)
        value_tensor = torch.tensor([[float(value)]], dtype=torch.float32, device=self.device)
        memberships = gaussian_membership(value_tensor, centers.view(1, -1), sigmas.view(1, -1)).squeeze(0)
        topk = min(2, int(memberships.shape[0]))
        top_values, top_indices = torch.topk(memberships, k=topk)
        top1 = float(top_values[0].item())
        top2 = float(top_values[1].item()) if topk > 1 else 0.0
        return {
            "best_label_index": int(top_indices[0].item()),
            "top1_membership": top1,
            "top2_membership": top2,
            "confidence_margin": max(0.0, top1 - top2),
        }

    def _build_state_signature(self, state_vec):
        signature = {}
        for dim_idx, var_name in enumerate(self.variable_names):
            signature[var_name] = self._state_signature_entry(var_name, state_vec[dim_idx])
        return signature

    def _priority_rank(self, var_name):
        return EXPLANATION_PRIORITY_INDEX.get(var_name, len(EXPLANATION_PRIORITY_INDEX))

    def _sorted_by_priority(self, values, score_fn):
        return sorted(
            values,
            key=lambda var_name: (-float(score_fn(var_name)), self._priority_rank(var_name)),
        )

    def _care_stats(self):
        care_mask = self.rule_indices != DONT_CARE_INDEX
        care_count = care_mask.sum(dim=1).to(dtype=torch.float32)
        total_dims = float(len(self.variable_names))
        specificity_weight = torch.zeros_like(care_count)
        valid_mask = care_count > 0
        if torch.any(valid_mask):
            specificity_weight[valid_mask] = (
                care_count[valid_mask] / total_dims
            ) ** self.coverage_specificity_alpha
        return care_mask, care_count, specificity_weight

    def compute_rule_activations(self, x_state):
        x_state = self._ensure_batch(x_state)
        batch_size = x_state.shape[0]
        rule_count = self.rule_indices.shape[0]
        var_memberships = {}
        activation_parts = []
        mismatch_scores = {}

        for dim_idx, var_name in enumerate(self.variable_names):
            centers, sigmas = self._get_variable_params(var_name)
            x_val = x_state[:, dim_idx : dim_idx + 1]
            memberships = gaussian_membership(x_val, centers.view(1, -1), sigmas.view(1, -1))
            chosen_idx = self.rule_indices[:, dim_idx]
            selected = torch.ones((batch_size, rule_count), dtype=torch.float32, device=self.device)
            mismatch = torch.zeros((batch_size, rule_count), dtype=torch.float32, device=self.device)
            valid_mask = chosen_idx != DONT_CARE_INDEX
            if torch.any(valid_mask):
                valid_indices = chosen_idx[valid_mask]
                valid_selected = memberships[:, valid_indices]
                selected[:, valid_mask] = valid_selected
                mismatch[:, valid_mask] = 1.0 - valid_selected
            activation_parts.append(selected)
            var_memberships[var_name] = memberships
            mismatch_scores[var_name] = mismatch

        activation = torch.ones_like(activation_parts[0])
        for part in activation_parts:
            activation = activation * part

        care_mask, care_count, specificity_weight = self._care_stats()
        coverage_score = activation * specificity_weight.unsqueeze(0)
        activation_sum = activation.sum(dim=1, keepdim=True) + 1e-6
        return {
            "x_state": x_state,
            "activation": activation,
            "activation_sum": activation_sum,
            "coverage_score": coverage_score,
            "care_mask": care_mask,
            "care_count": care_count,
            "specificity_weight": specificity_weight,
            "var_memberships": var_memberships,
            "mismatch_scores": mismatch_scores,
        }

    def forward(self, x_state):
        stats = self.compute_rule_activations(x_state)
        activation = stats["activation"]
        activation_sum = stats["activation_sum"]
        weighted = activation.unsqueeze(-1) * self.consequents.unsqueeze(0)
        action = weighted.sum(dim=1) / activation_sum
        v_action = torch.clamp(action[:, 0], 0.0, self.max_speed)
        w_action = torch.clamp(action[:, 1], -self.max_turn, self.max_turn)
        return torch.stack((v_action, w_action), dim=1)

    def explain(self, x_state, topk=3):
        stats = self.compute_rule_activations(x_state)
        activation = stats["activation"][0]
        topk = int(min(topk, activation.numel()))
        values, indices = torch.topk(activation, k=topk)
        explanation = []
        for value, idx in zip(values.tolist(), indices.tolist()):
            explanation.append(
                {
                    "rule_index": idx,
                    "activation": float(value),
                    "text": self.rule_texts[idx],
                    "action": self.consequents[idx].detach().cpu().tolist(),
                }
            )
        return explanation

    def check_uncovered_state(self, x_state, threshold=None):
        threshold = self.uncovered_threshold if threshold is None else float(threshold)
        stats = self.compute_rule_activations(x_state)
        activation = stats["activation"][0]
        coverage_score = stats["coverage_score"][0]
        max_coverage_score, best_idx = torch.max(coverage_score, dim=0)
        max_activation = torch.max(activation)
        uncovered = bool(max_coverage_score.item() < threshold)
        best_rule_care_mask = stats["care_mask"][best_idx]

        candidate_scores = {}
        wildcard_specialization_scores = {}
        cared_scores = {}
        for var_name in DIMENSION_PRIORITY:
            dim_idx = self.variable_names.index(var_name)
            mismatch_score = float(stats["mismatch_scores"][var_name][0, best_idx].item())
            candidate_scores[var_name] = mismatch_score
            if bool(best_rule_care_mask[dim_idx].item()):
                cared_scores[var_name] = mismatch_score
            else:
                wildcard_specialization_scores[var_name] = float(stats["var_memberships"][var_name][0].max().item())

        chosen_var = None
        selected_variable_source = None
        if uncovered:
            mismatch_candidates = {
                var_name: score
                for var_name, score in cared_scores.items()
                if score >= self.selected_variable_mismatch_threshold
            }
            if mismatch_candidates:
                chosen_var = max(mismatch_candidates, key=mismatch_candidates.get)
                selected_variable_source = "mismatch"
            elif wildcard_specialization_scores:
                chosen_var = max(wildcard_specialization_scores, key=wildcard_specialization_scores.get)
                selected_variable_source = "wildcard_specialization"
            elif cared_scores:
                chosen_var = max(cared_scores, key=cared_scores.get)
                selected_variable_source = "mismatch"

        return {
            "uncovered": uncovered,
            "max_activation": float(max_activation.item()),
            "max_coverage_score": float(max_coverage_score.item()),
            "best_rule_idx": int(best_idx.item()),
            "best_rule_care_count": int(stats["care_count"][best_idx].item()),
            "selected_variable": chosen_var,
            "selected_variable_source": selected_variable_source,
            "candidate_scores": candidate_scores,
            "wildcard_specialization_scores": wildcard_specialization_scores,
            "state": self._ensure_batch(x_state)[0].detach().cpu().numpy(),
        }

    def _replace_variable_partition(self, var_name, centers, sigmas, labels):
        setattr(
            self,
            f"{var_name}_centers",
            nn.Parameter(torch.as_tensor(centers, dtype=torch.float32, device=self.device)),
        )
        setattr(
            self,
            f"{var_name}_sigmas",
            nn.Parameter(torch.as_tensor(sigmas, dtype=torch.float32, device=self.device)),
        )
        self.rulebook["fuzzy_sets"][var_name]["centers"] = [float(v) for v in centers]
        self.rulebook["fuzzy_sets"][var_name]["sigmas"] = [float(v) for v in sigmas]
        self.rulebook["fuzzy_sets"][var_name]["labels"] = list(labels)

    def _refresh_rule_texts(self):
        rules = []
        antecedents = self.rule_indices.detach().cpu().tolist()
        consequents = self.consequents.detach().cpu().tolist()
        for idx, (ant, act) in enumerate(zip(antecedents, consequents)):
            rules.append(
                {
                    "antecedent": ant,
                    "action": act,
                    "origin": self.rule_origins[idx] if idx < len(self.rule_origins) else "adapted",
                    "name": f"rule_{idx}",
                }
            )
        self.rulebook["rules"] = rules
        self.rule_texts = generate_rule_texts(self.rulebook["fuzzy_sets"], rules)

    def _rank_additional_rule_variables(self, selected_variable, uncovered_info, state_signature):
        candidate_scores = uncovered_info.get("candidate_scores", {})
        remaining = [var_name for var_name in self.variable_names if var_name != selected_variable]

        high_mismatch = [
            var_name
            for var_name in remaining
            if float(candidate_scores.get(var_name, 0.0)) >= self.added_rule_mismatch_threshold
        ]
        high_mismatch = self._sorted_by_priority(
            high_mismatch,
            lambda var_name: candidate_scores.get(var_name, 0.0),
        )

        leftover = [var_name for var_name in remaining if var_name not in high_mismatch]
        confident = [
            var_name
            for var_name in leftover
            if float(state_signature[var_name]["confidence_margin"]) >= self.confidence_margin_threshold
        ]
        confident = self._sorted_by_priority(
            confident,
            lambda var_name: state_signature[var_name]["confidence_margin"],
        )

        low_confidence = [var_name for var_name in leftover if var_name not in confident]
        low_confidence = self._sorted_by_priority(
            low_confidence,
            lambda var_name: state_signature[var_name]["confidence_margin"],
        )
        return high_mismatch + confident + low_confidence

    def _candidate_rule_dimensions(self, selected_variable, ordered_variables, target_dim_count):
        cared_variables = [selected_variable]
        for var_name in ordered_variables:
            if var_name == selected_variable or var_name in cared_variables:
                continue
            cared_variables.append(var_name)
            if len(cared_variables) >= target_dim_count:
                break
        return cared_variables

    def _build_candidate_antecedent(self, state_signature, cared_variables):
        cared_set = set(cared_variables)
        antecedent = []
        for var_name in self.variable_names:
            if var_name in cared_set:
                antecedent.append(int(state_signature[var_name]["best_label_index"]))
            else:
                antecedent.append(DONT_CARE_INDEX)
        return antecedent

    def _nearest_center_details(self, var_name, value):
        centers, sigmas = self._get_variable_params(var_name)
        distances = torch.abs(centers.detach() - float(value))
        nearest_idx = int(torch.argmin(distances).item())
        nearest_gap = float(distances[nearest_idx].item())
        nearest_sigma = float(sigmas[nearest_idx].detach().cpu().item())
        return nearest_idx, nearest_gap, nearest_sigma

    def _find_identical_antecedent(self, antecedent):
        target = tuple(int(value) for value in antecedent)
        for idx, existing in enumerate(self.rule_indices.detach().cpu().tolist()):
            if tuple(int(value) for value in existing) == target:
                return idx
        return None

    def _speed_class(self, v_value):
        v_value = float(v_value)
        if v_value < 0.05:
            return "stop"
        if v_value < 0.15:
            return "slow"
        return "cruise"

    def _turn_class(self, w_value):
        w_value = float(w_value)
        if w_value <= -0.9:
            return "hard_right"
        if w_value <= -0.25:
            return "right"
        if abs(w_value) < 0.25:
            return "straight"
        if w_value < 0.9:
            return "left"
        return "hard_left"

    def _actions_compatible(self, action_a, action_b):
        action_a = torch.as_tensor(action_a, dtype=torch.float32, device=self.device)
        action_b = torch.as_tensor(action_b, dtype=torch.float32, device=self.device)
        return (
            self._speed_class(action_a[0].item()) == self._speed_class(action_b[0].item())
            and self._turn_class(action_a[1].item()) == self._turn_class(action_b[1].item())
        )

    def add_rule(self, x_state, uncovered_info=None):
        uncovered_info = uncovered_info or self.check_uncovered_state(x_state)
        if not uncovered_info["uncovered"]:
            return None

        state_vec = uncovered_info["state"]
        selected_variable = uncovered_info["selected_variable"] or DIMENSION_PRIORITY[0]
        dim_idx = self.variable_names.index(selected_variable)
        state_signature = self._build_state_signature(state_vec)
        ordered_variables = self._rank_additional_rule_variables(
            selected_variable,
            uncovered_info,
            state_signature,
        )
        with torch.no_grad():
            current_policy_action = self.forward(self._ensure_batch(state_vec))[0].detach().clone()

        chosen_indices = None
        chosen_care_variables = None
        merge_into_existing_idx = None
        target_dim_range = range(self.new_rule_min_dims, self.new_rule_max_dims + 1)
        for target_dim_count in target_dim_range:
            cared_variables = self._candidate_rule_dimensions(
                selected_variable,
                ordered_variables,
                target_dim_count,
            )
            candidate_indices = self._build_candidate_antecedent(state_signature, cared_variables)
            existing_idx = self._find_identical_antecedent(candidate_indices)
            if existing_idx is None:
                chosen_indices = candidate_indices
                chosen_care_variables = cared_variables
                break
            chosen_care_variables = cared_variables
            existing_action = self.consequents[existing_idx].detach()
            if self._actions_compatible(existing_action, current_policy_action):
                merge_into_existing_idx = existing_idx
                break

        if merge_into_existing_idx is not None:
            merged_action = (
                0.7 * self.consequents[merge_into_existing_idx].detach()
                + 0.3 * current_policy_action
            )
            with torch.no_grad():
                self.consequents[merge_into_existing_idx].copy_(merged_action)
            self.rule_origins[merge_into_existing_idx] = f"expanded_merged:{selected_variable}"
            self._refresh_rule_texts()
            self.enforce_bounds()
            return {
                "event_type": "merged_existing_rule",
                "structure_changed": False,
                "merged_into_rule_index": int(merge_into_existing_idx),
                "selected_variable": selected_variable,
                "care_variables": list(chosen_care_variables or [selected_variable]),
                "rule_dim_count": len(chosen_care_variables or [selected_variable]),
            }

        if chosen_indices is None:
            return None

        if self.rule_indices.shape[0] >= self.max_rules:
            return None

        nearest_idx, nearest_gap, nearest_sigma = self._nearest_center_details(
            selected_variable,
            state_vec[dim_idx],
        )
        should_add_fuzzy_set = nearest_gap >= max(
            self.new_fuzzy_set_min_abs_gap,
            self.new_fuzzy_set_min_gap_ratio * nearest_sigma,
        )

        final_indices = list(chosen_indices)
        if should_add_fuzzy_set:
            centers, sigmas = self._get_variable_params(selected_variable)
            centers_np = centers.detach().cpu().numpy().tolist()
            sigmas_np = sigmas.detach().cpu().numpy().tolist()
            labels = list(self.rulebook["fuzzy_sets"][selected_variable]["labels"])

            new_center = float(np.clip(state_vec[dim_idx], *VALUE_BOUNDS[selected_variable]))
            new_sigma = float(np.clip(nearest_sigma, self.min_sigma, self.max_sigma))
            centers_np.append(new_center)
            sigmas_np.append(new_sigma)
            labels.append(f"{selected_variable}_extra_{len(labels)}")
            self._replace_variable_partition(selected_variable, centers_np, sigmas_np, labels)
            final_indices[dim_idx] = len(centers_np) - 1

        new_rule_indices = torch.tensor([final_indices], dtype=torch.long, device=self.device)
        self.rule_indices = torch.cat([self.rule_indices, new_rule_indices], dim=0)
        self.consequents = nn.Parameter(
            torch.cat([self.consequents, current_policy_action.unsqueeze(0)], dim=0)
        )
        self.rule_origins.append(f"expanded:{selected_variable}")
        self._refresh_rule_texts()
        self.enforce_bounds()
        return {
            "event_type": "added_rule",
            "structure_changed": True,
            "added_rule_index": int(self.rule_indices.shape[0] - 1),
            "selected_variable": selected_variable,
            "care_variables": list(chosen_care_variables),
            "rule_dim_count": len(chosen_care_variables),
            "added_fuzzy_set": bool(should_add_fuzzy_set),
        }

    def _merge_single_variable(self, var_name, merge_threshold):
        centers, sigmas = self._get_variable_params(var_name)
        centers_list = centers.detach().cpu().tolist()
        sigmas_list = sigmas.detach().cpu().tolist()
        labels_list = list(self.rulebook["fuzzy_sets"][var_name]["labels"])

        changed = False
        while True:
            pair = None
            for idx_a in range(len(centers_list)):
                for idx_b in range(idx_a + 1, len(centers_list)):
                    center_gap = abs(centers_list[idx_a] - centers_list[idx_b])
                    sigma_gap = abs(sigmas_list[idx_a] - sigmas_list[idx_b])
                    if center_gap <= merge_threshold and sigma_gap <= merge_threshold:
                        pair = (idx_a, idx_b)
                        break
                if pair is not None:
                    break
            if pair is None:
                break

            idx_a, idx_b = pair
            merged_center = 0.5 * (centers_list[idx_a] + centers_list[idx_b])
            merged_sigma = max(self.min_sigma, 0.5 * (sigmas_list[idx_a] + sigmas_list[idx_b]))
            merged_label = labels_list[idx_a]

            new_centers = []
            new_sigmas = []
            new_labels = []
            index_map = {}
            new_idx = 0
            for old_idx in range(len(centers_list)):
                if old_idx == idx_a:
                    new_centers.append(merged_center)
                    new_sigmas.append(merged_sigma)
                    new_labels.append(merged_label)
                    index_map[old_idx] = new_idx
                    new_idx += 1
                elif old_idx == idx_b:
                    index_map[old_idx] = index_map[idx_a]
                else:
                    new_centers.append(centers_list[old_idx])
                    new_sigmas.append(sigmas_list[old_idx])
                    new_labels.append(labels_list[old_idx])
                    index_map[old_idx] = new_idx
                    new_idx += 1

            dim_idx = self.variable_names.index(var_name)
            updated_indices = self.rule_indices.detach().cpu().clone()
            for row_idx in range(updated_indices.shape[0]):
                old_val = int(updated_indices[row_idx, dim_idx].item())
                if old_val == DONT_CARE_INDEX:
                    continue
                updated_indices[row_idx, dim_idx] = index_map[old_val]

            self.rule_indices = updated_indices.to(self.device)
            centers_list = new_centers
            sigmas_list = new_sigmas
            labels_list = new_labels
            changed = True

        if changed:
            self._replace_variable_partition(var_name, centers_list, sigmas_list, labels_list)
        return changed

    def _deduplicate_identical_rules(self):
        grouped = {}
        for idx, antecedent in enumerate(self.rule_indices.detach().cpu().tolist()):
            grouped.setdefault(tuple(antecedent), []).append(idx)

        duplicate_merged_rules = 0
        if all(len(indices) == 1 for indices in grouped.values()):
            return duplicate_merged_rules

        keep_indices = []
        merged_actions = []
        merged_origins = []
        for indices in grouped.values():
            base_idx = indices[0]
            if len(indices) > 1:
                merged_actions.append(
                    torch.stack([self.consequents[idx].detach() for idx in indices], dim=0).mean(dim=0)
                )
                duplicate_merged_rules += len(indices) - 1
                merged_origins.append("merged_duplicate")
            else:
                merged_actions.append(self.consequents[base_idx].detach())
                merged_origins.append(self.rule_origins[base_idx])
            keep_indices.append(base_idx)

        new_rule_indices = self.rule_indices[keep_indices].detach().clone()
        new_consequents = torch.stack(merged_actions, dim=0).to(self.device)
        self.rule_indices = new_rule_indices.to(self.device)
        self.consequents = nn.Parameter(new_consequents)
        self.rule_origins = merged_origins
        return duplicate_merged_rules

    def _find_single_dim_generalization_candidates(self, action_merge_threshold):
        antecedents = self.rule_indices.detach().cpu().tolist()
        action_values = self.consequents.detach()
        candidates = []

        for idx_a in range(len(antecedents)):
            for idx_b in range(idx_a + 1, len(antecedents)):
                diff_dims = [dim_idx for dim_idx in range(len(self.variable_names)) if antecedents[idx_a][dim_idx] != antecedents[idx_b][dim_idx]]
                if len(diff_dims) != 1:
                    continue

                diff_dim = diff_dims[0]
                left_val = antecedents[idx_a][diff_dim]
                right_val = antecedents[idx_b][diff_dim]
                if left_val == DONT_CARE_INDEX or right_val == DONT_CARE_INDEX:
                    continue

                generalized = list(antecedents[idx_a])
                generalized[diff_dim] = DONT_CARE_INDEX
                care_count = sum(1 for value in generalized if value != DONT_CARE_INDEX)
                if care_count < 2:
                    continue

                action_distance = float(torch.norm(action_values[idx_a] - action_values[idx_b], p=2).item())
                if action_distance > action_merge_threshold:
                    continue

                candidates.append((action_distance, idx_a, idx_b, diff_dim, generalized))

        candidates.sort(key=lambda item: item[0])
        return candidates

    def _apply_single_dim_generalization(self, action_merge_threshold):
        candidates = self._find_single_dim_generalization_candidates(action_merge_threshold)
        if not candidates:
            return 0

        used_rule_indices = set()
        selected_pairs = []
        for _, idx_a, idx_b, diff_dim, generalized in candidates:
            if idx_a in used_rule_indices or idx_b in used_rule_indices:
                continue
            used_rule_indices.add(idx_a)
            used_rule_indices.add(idx_b)
            selected_pairs.append((idx_a, idx_b, diff_dim, generalized))

        if not selected_pairs:
            return 0

        kept_indices = [idx for idx in range(self.rule_indices.shape[0]) if idx not in used_rule_indices]
        new_rule_indices = [self.rule_indices[idx].detach().cpu().tolist() for idx in kept_indices]
        new_consequents = [self.consequents[idx].detach() for idx in kept_indices]
        new_origins = [self.rule_origins[idx] for idx in kept_indices]

        for idx_a, idx_b, diff_dim, generalized in selected_pairs:
            new_rule_indices.append(generalized)
            new_consequents.append(0.5 * (self.consequents[idx_a].detach() + self.consequents[idx_b].detach()))
            new_origins.append(f"generalized:{self.variable_names[diff_dim]}")

        self.rule_indices = torch.tensor(new_rule_indices, dtype=torch.long, device=self.device)
        self.consequents = nn.Parameter(torch.stack(new_consequents, dim=0).to(self.device))
        self.rule_origins = new_origins
        return len(selected_pairs)

    def merge_rules(self, merge_threshold=0.12, action_merge_threshold=0.12):
        merge_threshold = float(merge_threshold)
        action_merge_threshold = float(action_merge_threshold)

        fuzzy_changed = False
        for var_name in self.variable_names:
            fuzzy_changed |= self._merge_single_variable(var_name, merge_threshold)

        duplicate_merged_rules = self._deduplicate_identical_rules()
        generalized_rule_count = 0
        generalization_passes = 0
        max_passes = max(1, int(self.rule_indices.shape[0]))
        for _ in range(max_passes):
            generalized_this_pass = self._apply_single_dim_generalization(action_merge_threshold)
            if generalized_this_pass <= 0:
                break
            generalization_passes += 1
            generalized_rule_count += generalized_this_pass
            duplicate_merged_rules += self._deduplicate_identical_rules()

        if fuzzy_changed or duplicate_merged_rules > 0 or generalized_rule_count > 0:
            self._refresh_rule_texts()
            self.enforce_bounds()

        return {
            "fuzzy_changed": bool(fuzzy_changed),
            "merged_rule_count": int(duplicate_merged_rules + generalized_rule_count),
            "duplicate_merged_rule_count": int(duplicate_merged_rules),
            "generalized_rule_count": int(generalized_rule_count),
            "generalization_passes": int(generalization_passes),
            "rule_count": int(self.rule_indices.shape[0]),
        }

    def enforce_bounds(self):
        with torch.no_grad():
            for var_name in self.variable_names:
                centers, sigmas = self._get_variable_params(var_name)
                low, high = VALUE_BOUNDS[var_name]
                centers.data.clamp_(low, high)
                sigmas.data.clamp_(self.min_sigma, self.max_sigma)
            self.consequents.data[:, 0].clamp_(0.0, self.max_speed)
            self.consequents.data[:, 1].clamp_(-self.max_turn, self.max_turn)

    def export_rulebook(self):
        return self.export_rulebook_with_semantics(export_semantics=True)

    def export_rule_texts(self, *, export_semantics=True):
        payload = self.export_rulebook_with_semantics(export_semantics=export_semantics)
        return list(payload["rule_texts"])

    def export_rulebook_with_semantics(self, *, export_semantics=True):
        fuzzy_sets = deepcopy(self.rulebook["fuzzy_sets"])
        for var_name in self.variable_names:
            centers, sigmas = self._get_variable_params(var_name)
            fuzzy_sets[var_name]["centers"] = centers.detach().cpu().tolist()
            fuzzy_sets[var_name]["sigmas"] = sigmas.detach().cpu().tolist()

        rules = []
        for idx, antecedent in enumerate(self.rule_indices.detach().cpu().tolist()):
            rules.append(
                {
                    "antecedent": antecedent,
                    "action": self.consequents[idx].detach().cpu().tolist(),
                    "origin": self.rule_origins[idx],
                    "name": f"rule_{idx}",
                }
            )

        export_fuzzy_sets = build_export_fuzzy_sets(fuzzy_sets) if export_semantics else fuzzy_sets
        export_rule_texts = generate_rule_texts(export_fuzzy_sets, rules) if export_semantics else list(self.rule_texts)

        return {
            "version": 2,
            "rule_format": RULE_FORMAT,
            "coverage_specificity_alpha": self.coverage_specificity_alpha,
            "added_rule_mismatch_threshold": self.added_rule_mismatch_threshold,
            "added_rule_min_care_dims": self.added_rule_min_care_dims,
            "selected_variable_mismatch_threshold": self.selected_variable_mismatch_threshold,
            "new_rule_min_dims": self.new_rule_min_dims,
            "new_rule_max_dims": self.new_rule_max_dims,
            "confidence_margin_threshold": self.confidence_margin_threshold,
            "new_fuzzy_set_min_gap_ratio": self.new_fuzzy_set_min_gap_ratio,
            "new_fuzzy_set_min_abs_gap": self.new_fuzzy_set_min_abs_gap,
            "variable_names": list(self.variable_names),
            "max_rules": self.max_rules,
            "uncovered_threshold": self.uncovered_threshold,
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
            "max_speed": self.max_speed,
            "max_turn": self.max_turn,
            "fuzzy_sets": export_fuzzy_sets,
            "rules": rules,
            "rule_texts": export_rule_texts,
        }

    def save_rulebook(self, path):
        save_rulebook_file(self.export_rulebook(), path)

    @classmethod
    def load_rulebook(cls, path, *, device=None):
        payload = load_rulebook_file(path)
        return cls.load_rulebook_from_payload(payload, device=device)

    @classmethod
    def load_rulebook_from_payload(cls, payload, *, device=None):
        if payload.get("rule_format") != RULE_FORMAT:
            raise ValueError(
                f"Unsupported rule_format '{payload.get('rule_format')}'. Expected '{RULE_FORMAT}'."
            )

        teacher = cls(
            device=device,
            max_rules=payload.get("max_rules", 15),
            uncovered_threshold=payload.get("uncovered_threshold", 0.12),
            min_sigma=payload.get("min_sigma", 0.06),
            max_sigma=payload.get("max_sigma", 0.45),
            max_speed=payload.get("max_speed", 0.22),
            max_turn=payload.get("max_turn", 1.5),
            coverage_specificity_alpha=payload.get(
                "coverage_specificity_alpha",
                COVERAGE_SPECIFICITY_ALPHA,
            ),
            added_rule_mismatch_threshold=payload.get("added_rule_mismatch_threshold", 0.35),
            added_rule_min_care_dims=payload.get("added_rule_min_care_dims", 2),
            selected_variable_mismatch_threshold=payload.get("selected_variable_mismatch_threshold", 0.35),
            new_rule_min_dims=payload.get("new_rule_min_dims", 2),
            new_rule_max_dims=payload.get("new_rule_max_dims", 4),
            confidence_margin_threshold=payload.get("confidence_margin_threshold", 0.15),
            new_fuzzy_set_min_gap_ratio=payload.get("new_fuzzy_set_min_gap_ratio", 0.75),
            new_fuzzy_set_min_abs_gap=payload.get("new_fuzzy_set_min_abs_gap", 0.08),
        )
        teacher.rulebook = {
            "version": payload.get("version", 2),
            "rule_format": payload["rule_format"],
            "coverage_specificity_alpha": payload.get(
                "coverage_specificity_alpha",
                COVERAGE_SPECIFICITY_ALPHA,
            ),
            "added_rule_mismatch_threshold": payload.get("added_rule_mismatch_threshold", 0.35),
            "added_rule_min_care_dims": payload.get("added_rule_min_care_dims", 2),
            "selected_variable_mismatch_threshold": payload.get("selected_variable_mismatch_threshold", 0.35),
            "new_rule_min_dims": payload.get("new_rule_min_dims", 2),
            "new_rule_max_dims": payload.get("new_rule_max_dims", 4),
            "confidence_margin_threshold": payload.get("confidence_margin_threshold", 0.15),
            "new_fuzzy_set_min_gap_ratio": payload.get("new_fuzzy_set_min_gap_ratio", 0.75),
            "new_fuzzy_set_min_abs_gap": payload.get("new_fuzzy_set_min_abs_gap", 0.08),
            "variable_names": payload["variable_names"],
            "fuzzy_sets": deepcopy(payload["fuzzy_sets"]),
            "rules": deepcopy(payload["rules"]),
        }
        teacher.variable_names = tuple(payload["variable_names"])
        teacher.rule_origins = [rule.get("origin", "loaded") for rule in payload["rules"]]

        for var_name in teacher.variable_names:
            fuzzy_cfg = payload["fuzzy_sets"][var_name]
            teacher._replace_variable_partition(
                var_name,
                fuzzy_cfg["centers"],
                fuzzy_cfg["sigmas"],
                fuzzy_cfg["labels"],
            )

        teacher.rule_indices = torch.tensor(
            [rule["antecedent"] for rule in payload["rules"]],
            dtype=torch.long,
            device=teacher.device,
        )
        teacher.consequents = nn.Parameter(
            torch.tensor([rule["action"] for rule in payload["rules"]], dtype=torch.float32, device=teacher.device)
        )
        teacher.rule_texts = payload.get(
            "rule_texts",
            generate_rule_texts(payload["fuzzy_sets"], payload["rules"]),
        )
        teacher.enforce_bounds()
        return teacher
