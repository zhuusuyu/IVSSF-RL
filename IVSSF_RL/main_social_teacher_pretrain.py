import argparse
import csv
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

try:
    from .social_teacher_agent import SocialTeacherAgent
    from .social_teacher_env import SocialTeacherRosEnv
    from .social_teacher_trainer import SocialTeacherTrainer
except ImportError:
    from social_teacher_agent import SocialTeacherAgent
    from social_teacher_env import SocialTeacherRosEnv
    from social_teacher_trainer import SocialTeacherTrainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "t", "yes", "y"}


def evaluate_policy(env, agent, episodes=3):
    total_reward = 0.0
    total_steps = 0
    success = 0
    social_min = []

    for episode_idx in range(episodes):
        state, _ = env.reset(seed=episode_idx + 10_000)
        done = False
        truncated = False
        episode_reward = 0.0
        while not (done or truncated):
            action = agent.select_action(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            total_steps += 1
        total_reward += episode_reward
        success += int(info.get("reached", False))
        episode_min_distance = info.get("episode_min_human_distance", info.get("min_human_distance", 10.0))
        social_min.append(float(episode_min_distance))

    return {
        "avg_reward": total_reward / max(1, episodes),
        "avg_steps": total_steps / max(1, episodes),
        "success_rate": success / max(1, episodes),
        "avg_min_human_distance": float(np.mean(social_min)) if social_min else 0.0,
    }


def export_rule_texts(rule_texts, path):
    with open(path, "w", encoding="utf-8") as handle:
        for idx, line in enumerate(rule_texts):
            handle.write(f"{idx:02d}: {line}\n")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_train_log(path):
    expected_header = [
        "total_steps",
        "episode",
        "episode_reward",
        "episode_steps",
        "rule_count",
        "critic_loss",
        "actor_loss",
        "bootstrap_phase",
        "bootstrap_expansions",
        "consecutive_success_episodes",
        "success_triggered_merge_count",
        "last_merge_reduction",
        "post_merge_expand_freeze_remaining",
    ]
    rewrite = True
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        rewrite = first_line != ",".join(expected_header)
    if rewrite:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(expected_header)


def ensure_eval_log(path):
    expected_header = [
        "eval_step",
        "seed",
        "scene",
        "eval_episodes",
        "avg_reward",
        "avg_steps",
        "success_rate",
        "avg_min_human_distance",
        "rule_count",
    ]
    rewrite = True
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        rewrite = first_line != ",".join(expected_header)
    if rewrite:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(expected_header)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render", type=str2bool, default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="teacher_checkpoints_dontcare_merge_nobootstraping_cross_500cool_uncover")
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--merge_interval", type=int, default=0)
    parser.add_argument("--expand_cooldown_steps", type=int, default=500)
    parser.add_argument("--expand_patience", dest="expand_cooldown_steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max_rules", type=int, default=100)
    parser.add_argument("--bootstrap_expand_steps", type=int, default=0)
    parser.add_argument("--bootstrap_target_rules", type=int, default=40)
    parser.add_argument("--bootstrap_uncovered_threshold", type=float, default=0.25)
    parser.add_argument("--post_bootstrap_uncovered_threshold", type=float, default=0.25)
    parser.add_argument("--success_merge_episodes", type=int, default=10)
    parser.add_argument("--success_merge_rounds", type=int, default=3)
    parser.add_argument("--post_merge_expand_freeze_steps", type=int, default=1000)
    parser.add_argument("--scene", type=str, default="teacher_cross_dual_lane")
    return parser


def run_single_seed_training(args, seed, checkpoint_dir, device):
    seed_everything(seed)
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard is not available. Please install tensorboard in the current Python environment.")

    env = SocialTeacherRosEnv(
        render_mode=("human" if args.render else None),
        scene_mode=args.scene,
    )
    agent = SocialTeacherAgent(
        device=device,
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_rules=args.max_rules,
    )
    trainer = SocialTeacherTrainer(
        teacher=agent.teacher,
        device=device,
        expand_cooldown_steps=args.expand_cooldown_steps,
        merge_interval=args.merge_interval,
        max_rules=args.max_rules,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = checkpoint_dir / "teacher_train_log.csv"
    eval_log_path = checkpoint_dir / "teacher_eval_log.csv"
    tb_dir = checkpoint_dir / "tensorboard"
    ensure_train_log(train_log_path)
    ensure_eval_log(eval_log_path)
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    episode = 0
    total_steps = 0
    next_eval_step = max(1, args.eval_interval)
    last_eval_metrics = None
    bootstrap_steps = 0
    bootstrap_expansions = 0
    bootstrap_target_rules = min(max(0, int(args.bootstrap_target_rules)), int(args.max_rules))
    bootstrap_expand_steps = max(0, int(args.bootstrap_expand_steps))
    consecutive_success_episodes = 0
    success_triggered_merge_count = 0
    last_merge_reduction = 0
    post_merge_expand_freeze_remaining = 0

    while total_steps < args.max_steps:
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_steps = 0
        train_metrics = {}
        info = {}
        done = False
        truncated = False
        episode_had_bootstrap = False

        while not (done or truncated) and total_steps < args.max_steps:
            rule_count = int(agent.teacher.rule_indices.shape[0])
            bootstrap_phase = (
                bootstrap_expand_steps > 0
                and bootstrap_steps < bootstrap_expand_steps
                and rule_count < bootstrap_target_rules
            )
            episode_had_bootstrap = episode_had_bootstrap or bootstrap_phase

            action = agent.select_action(state, deterministic=bootstrap_phase)

            next_state, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done or truncated)
            if bootstrap_phase:
                train_metrics = {}
                bootstrap_steps += 1
            else:
                train_metrics = agent.train_step() or {}

            expand_frozen = (not bootstrap_phase) and post_merge_expand_freeze_remaining > 0
            adapt_metrics = trainer.observe_and_adapt(
                state,
                bypass_cooldown=bootstrap_phase,
                uncovered_threshold=(
                    args.bootstrap_uncovered_threshold
                    if bootstrap_phase
                    else args.post_bootstrap_uncovered_threshold
                ),
                expand_frozen=expand_frozen,
            )
            expand_event = adapt_metrics.get("expand_event")
            if expand_event is not None and bool(expand_event.get("structure_changed", False)):
                agent.rebuild_actor_optimizer()
                if bootstrap_phase:
                    bootstrap_expansions += 1
            if expand_frozen:
                post_merge_expand_freeze_remaining -= 1

            before_merge_rule_count = int(agent.teacher.rule_indices.shape[0])
            merge_info = None if bootstrap_phase else trainer.maybe_merge(total_steps + 1)
            if merge_info and (merge_info["fuzzy_changed"] or merge_info["merged_rule_count"] > 0):
                agent.rebuild_actor_optimizer()
                last_merge_reduction = max(
                    0,
                    before_merge_rule_count - int(agent.teacher.rule_indices.shape[0]),
                )

            state = next_state
            total_steps += 1
            episode_steps += 1
            episode_reward += reward

            if total_steps % args.save_interval == 0:
                ckpt_path = checkpoint_dir / f"teacher_step_{total_steps}.pt"
                rulebook_path = checkpoint_dir / f"teacher_rulebook_{total_steps}.json"
                rules_txt_path = checkpoint_dir / f"teacher_rules_{total_steps}.txt"
                agent.save(str(ckpt_path))
                agent.teacher.save_rulebook(str(rulebook_path))
                export_rule_texts(trainer.export_rule_texts(), str(rules_txt_path))

            while next_eval_step <= total_steps:
                eval_metrics = evaluate_policy(env, agent, episodes=args.eval_episodes)
                last_eval_metrics = dict(eval_metrics)
                with open(eval_log_path, "a", newline="", encoding="utf-8") as handle:
                    csv_writer = csv.writer(handle)
                    csv_writer.writerow(
                        [
                            next_eval_step,
                            seed,
                            args.scene,
                            args.eval_episodes,
                            round(eval_metrics["avg_reward"], 6),
                            round(eval_metrics["avg_steps"], 6),
                            round(eval_metrics["success_rate"], 6),
                            round(eval_metrics["avg_min_human_distance"], 6),
                            int(agent.teacher.rule_indices.shape[0]),
                        ]
                    )
                tb_writer.add_scalar("eval/success_rate", eval_metrics["success_rate"], next_eval_step)
                tb_writer.add_scalar("eval/avg_reward", eval_metrics["avg_reward"], next_eval_step)
                tb_writer.add_scalar("eval/avg_steps", eval_metrics["avg_steps"], next_eval_step)
                tb_writer.add_scalar(
                    "eval/avg_min_human_distance",
                    eval_metrics["avg_min_human_distance"],
                    next_eval_step,
                )
                print(
                    f"[eval] seed={seed} scene={args.scene} eval_step={next_eval_step} "
                    f"reward={eval_metrics['avg_reward']:.3f} succ={eval_metrics['success_rate']:.2f} "
                    f"rules={agent.teacher.rule_indices.shape[0]}"
                )
                next_eval_step += max(1, args.eval_interval)

        episode_success = (
            bool(info.get("reached", False))
            and not bool(info.get("collision", False))
            and not bool(truncated)
        )
        if episode_had_bootstrap:
            consecutive_success_episodes = 0
        elif episode_success:
            consecutive_success_episodes += 1
        else:
            consecutive_success_episodes = 0

        if (
            args.success_merge_episodes > 0
            and consecutive_success_episodes >= args.success_merge_episodes
        ):
            pre_merge_rules = checkpoint_dir / f"teacher_rules_pre_success_merge_{total_steps}.txt"
            export_rule_texts(trainer.export_rule_texts(), str(pre_merge_rules))

            merge_info = trainer.force_merge(rounds=args.success_merge_rounds)
            if merge_info.get("optimizer_reset_needed", False):
                agent.rebuild_actor_optimizer(sync_target=True)

            post_merge_rules = checkpoint_dir / f"teacher_rules_post_success_merge_{total_steps}.txt"
            export_rule_texts(trainer.export_rule_texts(), str(post_merge_rules))

            success_triggered_merge_count += 1
            last_merge_reduction = int(merge_info.get("rule_reduction", 0))
            post_merge_expand_freeze_remaining = max(
                post_merge_expand_freeze_remaining,
                max(0, int(args.post_merge_expand_freeze_steps)),
            )
            consecutive_success_episodes = 0

            tb_writer.add_scalar("merge/success_triggered_count", success_triggered_merge_count, total_steps)
            tb_writer.add_scalar("merge/last_rule_reduction", last_merge_reduction, total_steps)
            tb_writer.add_scalar("merge/rule_count_after", int(agent.teacher.rule_indices.shape[0]), total_steps)
            print(
                f"[success-merge] seed={seed} step={total_steps} "
                f"reduction={last_merge_reduction} rules={agent.teacher.rule_indices.shape[0]} "
                f"pre={pre_merge_rules.name} post={post_merge_rules.name}"
            )

        tb_writer.add_scalar("train/episode_reward", episode_reward, total_steps)
        tb_writer.add_scalar("train/episode_steps", episode_steps, total_steps)
        tb_writer.add_scalar("train/critic_loss", float(train_metrics.get("critic_loss", 0.0)), total_steps)
        tb_writer.add_scalar("train/bootstrap_phase", int(episode_had_bootstrap), total_steps)
        tb_writer.add_scalar("train/bootstrap_expansions", bootstrap_expansions, total_steps)
        tb_writer.add_scalar("train/consecutive_success_episodes", consecutive_success_episodes, total_steps)
        tb_writer.add_scalar("train/success_triggered_merge_count", success_triggered_merge_count, total_steps)
        tb_writer.add_scalar("train/last_merge_reduction", last_merge_reduction, total_steps)
        tb_writer.add_scalar(
            "train/post_merge_expand_freeze_remaining",
            post_merge_expand_freeze_remaining,
            total_steps,
        )
        if train_metrics.get("actor_loss") is not None:
            tb_writer.add_scalar("train/actor_loss", float(train_metrics["actor_loss"]), total_steps)

        with open(train_log_path, "a", newline="", encoding="utf-8") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(
                [
                    total_steps,
                    episode,
                    round(episode_reward, 4),
                    episode_steps,
                    int(agent.teacher.rule_indices.shape[0]),
                    round(float(train_metrics.get("critic_loss", 0.0)), 6) if episode_steps > 0 else 0.0,
                    "" if train_metrics.get("actor_loss") is None else round(float(train_metrics["actor_loss"]), 6),
                    int(episode_had_bootstrap),
                    int(bootstrap_expansions),
                    int(consecutive_success_episodes),
                    int(success_triggered_merge_count),
                    int(last_merge_reduction),
                    int(post_merge_expand_freeze_remaining),
                ]
            )

        episode += 1

    final_ckpt = checkpoint_dir / "teacher_final.pt"
    final_rulebook = checkpoint_dir / "teacher_rulebook_final.json"
    final_rules_txt = checkpoint_dir / "teacher_rules_final.txt"
    agent.save(str(final_ckpt))
    agent.teacher.save_rulebook(str(final_rulebook))
    export_rule_texts(trainer.export_rule_texts(), str(final_rules_txt))
    tb_writer.close()

    return {
        "seed": seed,
        "checkpoint_dir": str(checkpoint_dir),
        "final_ckpt": str(final_ckpt),
        "final_rulebook": str(final_rulebook),
        "final_rules_txt": str(final_rules_txt),
        "last_eval": last_eval_metrics,
    }


def summarize_multi_seed_results(root_dir, seed_dirs, scene):
    metric_columns = [
        "avg_reward",
        "avg_steps",
        "success_rate",
        "avg_min_human_distance",
    ]
    grouped = {}

    for seed_dir in seed_dirs:
        eval_log_path = Path(seed_dir) / "teacher_eval_log.csv"
        if not eval_log_path.exists():
            continue
        with open(eval_log_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                eval_step = int(row["eval_step"])
                grouped.setdefault(eval_step, {metric: [] for metric in metric_columns})
                for metric in metric_columns:
                    grouped[eval_step][metric].append(float(row[metric]))

    summary_path = Path(root_dir) / "multi_seed_eval_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "eval_step",
                "scene",
                "num_seeds",
                "reward_mean",
                "reward_std",
                "steps_mean",
                "steps_std",
                "success_rate_mean",
                "success_rate_std",
                "avg_min_human_distance_mean",
                "avg_min_human_distance_std",
            ]
        )
        for eval_step in sorted(grouped):
            values = grouped[eval_step]
            reward_vals = np.asarray(values["avg_reward"], dtype=np.float32)
            steps_vals = np.asarray(values["avg_steps"], dtype=np.float32)
            succ_vals = np.asarray(values["success_rate"], dtype=np.float32)
            dist_vals = np.asarray(values["avg_min_human_distance"], dtype=np.float32)
            writer.writerow(
                [
                    eval_step,
                    scene,
                    int(len(reward_vals)),
                    round(float(np.mean(reward_vals)), 6) if len(reward_vals) else "",
                    round(float(np.std(reward_vals)), 6) if len(reward_vals) else "",
                    round(float(np.mean(steps_vals)), 6) if len(steps_vals) else "",
                    round(float(np.std(steps_vals)), 6) if len(steps_vals) else "",
                    round(float(np.mean(succ_vals)), 6) if len(succ_vals) else "",
                    round(float(np.std(succ_vals)), 6) if len(succ_vals) else "",
                    round(float(np.mean(dist_vals)), 6) if len(dist_vals) else "",
                    round(float(np.std(dist_vals)), 6) if len(dist_vals) else "",
                ]
            )
    return summary_path


def main():
    parser = build_parser()
    args = parser.parse_args()

    base_seed = args.seed if args.base_seed is None else args.base_seed
    num_seeds = max(1, int(args.num_seeds))
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    root_checkpoint_dir = Path(args.checkpoint_dir)
    root_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed_values = [int(base_seed) + offset for offset in range(num_seeds)]
    successful_runs = []
    failed_runs = []

    for index, seed in enumerate(seed_values, start=1):
        seed_dir = root_checkpoint_dir / f"seed_{seed}"
        print(f"\n[multi-seed] run {index}/{len(seed_values)} seed={seed} output_dir={seed_dir}")
        try:
            run_result = run_single_seed_training(args, seed, seed_dir, device)
            successful_runs.append(run_result)
            last_eval = run_result.get("last_eval") or {}
            if last_eval:
                print(
                    f"[seed done] seed={seed} ckpt={run_result['final_ckpt']} "
                    f"reward={last_eval.get('avg_reward', 0.0):.3f} "
                    f"succ={last_eval.get('success_rate', 0.0):.2f}"
                )
            else:
                print(f"[seed done] seed={seed} ckpt={run_result['final_ckpt']}")
        except Exception as exc:
            failed_runs.append({"seed": seed, "error": str(exc), "output_dir": str(seed_dir)})
            print(f"[seed failed] seed={seed} output_dir={seed_dir} error={exc}")

    summary_path = summarize_multi_seed_results(
        root_dir=root_checkpoint_dir,
        seed_dirs=[result["checkpoint_dir"] for result in successful_runs],
        scene=args.scene,
    )

    print("\n[multi-seed summary]")
    print(f"scene: {args.scene}")
    print(f"requested_seeds: {seed_values}")
    print(f"successful: {[result['seed'] for result in successful_runs]}")
    print(f"failed: {[item['seed'] for item in failed_runs]}")
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
