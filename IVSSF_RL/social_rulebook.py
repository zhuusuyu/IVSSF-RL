import json
from copy import deepcopy


VARIABLE_NAMES = (
    "goal_dist",
    "goal_heading",
    "front_clearance",
    "left_social_pressure",
    "right_social_pressure",
)

DONT_CARE_INDEX = -1
RULE_FORMAT = "wildcard_v2"
COVERAGE_SPECIFICITY_ALPHA = 1.5

MONOTONIC_EXPORT_LABELS = {
    "goal_dist": ("near", "far"),
    "front_clearance": ("blocked", "clear"),
    "left_social_pressure": ("low", "high"),
    "right_social_pressure": ("low", "high"),
}


def initial_fuzzy_partition():
    return {
        "goal_dist": {
            "centers": [0.25, 0.75],
            "sigmas": [0.18, 0.18],
            "labels": ["near", "far"],
        },
        "goal_heading": {
            "centers": [-0.75, 0.0, 0.75],
            "sigmas": [0.24, 0.20, 0.24],
            "labels": ["right", "center", "left"],
        },
        "front_clearance": {
            "centers": [0.25, 0.75],
            "sigmas": [0.16, 0.16],
            "labels": ["blocked", "clear"],
        },
        "left_social_pressure": {
            "centers": [0.25, 0.75],
            "sigmas": [0.18, 0.18],
            "labels": ["low", "high"],
        },
        "right_social_pressure": {
            "centers": [0.25, 0.75],
            "sigmas": [0.18, 0.18],
            "labels": ["low", "high"],
        },
    }


def _rule(antecedent, action, name):
    return {
        "antecedent": list(antecedent),
        "action": [float(action[0]), float(action[1])],
        "name": name,
        "origin": "seed",
    }


def _initial_label_index_map():
    partition = initial_fuzzy_partition()
    return {
        var_name: {label: idx for idx, label in enumerate(cfg["labels"])}
        for var_name, cfg in partition.items()
    }


def seed_rule(name, action, **conditions):
    unknown_vars = sorted(set(conditions) - set(VARIABLE_NAMES))
    if unknown_vars:
        raise ValueError(f"Unknown seed rule variables: {unknown_vars}")

    label_map = _initial_label_index_map()
    antecedent = []
    care_count = 0

    for var_name in VARIABLE_NAMES:
        if var_name not in conditions:
            antecedent.append(DONT_CARE_INDEX)
            continue
        label = str(conditions[var_name])
        if label not in label_map[var_name]:
            raise ValueError(f"Unknown label '{label}' for variable '{var_name}'")
        antecedent.append(label_map[var_name][label])
        care_count += 1

    if care_count < 2:
        raise ValueError(f"Seed rule '{name}' must care about at least 2 dimensions")
    return _rule(antecedent, action, name)


def initial_seed_rules():
    return [
        seed_rule(
            "center clear -> fast straight",
            [0.22, 0.00],
            goal_heading="center",
            front_clearance="clear",
        ),
        seed_rule(
            "near center clear -> slow straight",
            [0.12, 0.00],
            goal_dist="near",
            goal_heading="center",
            front_clearance="clear",
        ),
        seed_rule(
            "left clear -> medium left turn",
            [0.16, 0.85],
            goal_heading="left",
            front_clearance="clear",
        ),
        seed_rule(
            "right clear -> medium right turn",
            [0.16, -0.85],
            goal_heading="right",
            front_clearance="clear",
        ),
        seed_rule(
            "near left clear -> cautious left turn",
            [0.08, 1.05],
            goal_dist="near",
            goal_heading="left",
            front_clearance="clear",
        ),
        seed_rule(
            "near right clear -> cautious right turn",
            [0.08, -1.05],
            goal_dist="near",
            goal_heading="right",
            front_clearance="clear",
        ),
        seed_rule(
            "blocked high-right -> bias left",
            [0.05, 0.95],
            front_clearance="blocked",
            right_social_pressure="high",
        ),
        seed_rule(
            "blocked high-left -> bias right",
            [0.05, -0.95],
            front_clearance="blocked",
            left_social_pressure="high",
        ),
        seed_rule(
            "near blocked high-high -> almost stop",
            [0.02, 0.00],
            goal_dist="near",
            front_clearance="blocked",
            left_social_pressure="high",
            right_social_pressure="high",
        ),
        seed_rule(
            "clear but high-high -> cautious straight",
            [0.07, 0.00],
            front_clearance="clear",
            left_social_pressure="high",
            right_social_pressure="high",
        ),
    ]


def build_initial_rulebook():
    return {
        "version": 2,
        "rule_format": RULE_FORMAT,
        "coverage_specificity_alpha": COVERAGE_SPECIFICITY_ALPHA,
        "variable_names": list(VARIABLE_NAMES),
        "fuzzy_sets": initial_fuzzy_partition(),
        "rules": initial_seed_rules(),
    }


def _sorted_indices(values):
    return sorted(range(len(values)), key=lambda idx: float(values[idx]))


def _relabel_monotonic(var_name, centers):
    order = _sorted_indices(centers)
    new_labels = [f"{var_name}_mid_{idx}" for idx in range(len(centers))]
    low_label, high_label = MONOTONIC_EXPORT_LABELS[var_name]
    if order:
        new_labels[order[0]] = low_label
        new_labels[order[-1]] = high_label
        for mid_rank, old_idx in enumerate(order[1:-1], start=1):
            new_labels[old_idx] = f"{var_name}_mid_{mid_rank}"
    return new_labels


def _relabel_goal_heading(centers):
    indexed = [(idx, float(center)) for idx, center in enumerate(centers)]
    if not indexed:
        return []

    center_idx, center_value = min(indexed, key=lambda item: abs(item[1]))
    new_labels = [f"goal_heading_extra_{idx}" for idx in range(len(centers))]
    new_labels[center_idx] = "center"

    right_side = sorted(
        [(idx, val) for idx, val in indexed if idx != center_idx and val < center_value],
        key=lambda item: item[1],
        reverse=True,
    )
    left_side = sorted(
        [(idx, val) for idx, val in indexed if idx != center_idx and val > center_value],
        key=lambda item: item[1],
    )

    if right_side:
        if len(right_side) == 1:
            new_labels[right_side[0][0]] = "right"
        else:
            for rank, (idx, _) in enumerate(right_side, start=1):
                new_labels[idx] = f"right_near_{rank}"
            new_labels[right_side[-1][0]] = f"right_far_{len(right_side)}"
            new_labels[right_side[0][0]] = "right"

    if left_side:
        if len(left_side) == 1:
            new_labels[left_side[0][0]] = "left"
        else:
            for rank, (idx, _) in enumerate(left_side, start=1):
                new_labels[idx] = f"left_near_{rank}"
            new_labels[left_side[-1][0]] = f"left_far_{len(left_side)}"
            new_labels[left_side[0][0]] = "left"

    return new_labels


def build_export_fuzzy_sets(fuzzy_sets):
    export_sets = deepcopy(fuzzy_sets)
    for var_name in VARIABLE_NAMES:
        centers = list(export_sets[var_name]["centers"])
        if var_name in MONOTONIC_EXPORT_LABELS:
            export_sets[var_name]["labels"] = _relabel_monotonic(var_name, centers)
        elif var_name == "goal_heading":
            export_sets[var_name]["labels"] = _relabel_goal_heading(centers)
    return export_sets


def antecedent_to_text(antecedent, fuzzy_sets):
    parts = []
    for var_name, mf_idx in zip(VARIABLE_NAMES, antecedent):
        if int(mf_idx) == DONT_CARE_INDEX:
            continue
        labels = fuzzy_sets[var_name]["labels"]
        if 0 <= int(mf_idx) < len(labels):
            label = labels[int(mf_idx)]
        else:
            label = f"set_{mf_idx}"
        parts.append(f"{var_name}={label}")
    return ", ".join(parts) if parts else "TRUE"


def action_to_text(action):
    v_val, w_val = action
    return f"v={v_val:.3f}, w={w_val:.3f}"


def generate_rule_texts(fuzzy_sets, rules):
    texts = []
    for idx, rule in enumerate(rules):
        antecedent = antecedent_to_text(rule["antecedent"], fuzzy_sets)
        action = action_to_text(rule["action"])
        name = rule.get("name", f"rule_{idx}")
        texts.append(f"{name}: IF {antecedent} THEN {action}")
    return texts


def save_rulebook(rulebook, path):
    serializable = deepcopy(rulebook)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=True, indent=2)


def load_rulebook(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
