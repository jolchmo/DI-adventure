"""
Q值可视化脚本 - 简化版
展示在遇到敌人时（第40个envstep附近），不同动作的Q值分布对比
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
from ding.utils import set_pkg_seed
from mario_dqn_config import mario_dqn_config, mario_dqn_create_config
from model import DQN
from policy import DQNPolicy
from ding.config import compile_config
from ding.envs import DingEnvWrapper
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 动作标签 - 使用英文避免字体问题
# 2动作: [["right"], ["right", "A"]]
ACTION_NAMES_2 = ['Right', 'Right+A']
# 7动作: SIMPLE_MOVEMENT = [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
ACTION_NAMES_7 = ['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'A(Jump)', 'Left']
# 12动作: COMPLEX_MOVEMENT
ACTION_NAMES_12 = ['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'A(Jump)', 'Left', 'Left+A', 'Left+B', 'Left+A+B', 'Down', 'Up']

# 敌人遭遇的关键帧（第40个envstep附近）
ENEMY_ENCOUNTER_STEP = 40
ANALYSIS_RANGE = 20  # 分析前后10帧


class ExtraInfoWrapperSimple:
    """简化的额外信息包装器"""

    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        return self.env.close()

    def __enter__(self):
        return self.env.__enter__()

    def __exit__(self, *args):
        return self.env.__exit__(*args)


def create_env(version=0, action_num=7, obs_frames=4):
    """创建包装后的Mario环境（简化版）
    version: 0=v0原版, 1=v1降采样
    action_num: 2, 7, 或 12
    """
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

    # 动作空间配置
    action_dict = {
        2: [["right"], ["right", "A"]],  # 只有右和右+跳
        7: SIMPLE_MOVEMENT,               # 7个简单动作
        12: COMPLEX_MOVEMENT              # 12个复杂动作
    }

    actions = action_dict.get(action_num, SIMPLE_MOVEMENT)

    env = DingEnvWrapper(
        JoypadSpace(
            gym_super_mario_bros.make(f"SuperMarioBros-1-1-v{version}"),
            actions
        ),
        cfg={
            'env_wrapper': [
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: WarpFrameWrapper(env, size=84),
                lambda env: ScaledFloatFrameWrapper(env),
                lambda env: FrameStackWrapper(env, n_frames=obs_frames),
            ]
        }
    )
    return ExtraInfoWrapperSimple(env)


def get_q_values(policy, obs):
    """获取所有动作的Q值"""
    with torch.no_grad():
        q_output = policy.forward({0: obs})
        q_values = q_output[0]['logit'].numpy()  # shape: (n_actions,)
        chosen_action = q_output[0]['action'].item()
    return q_values, chosen_action


def draw_q_value_bar(q_values, chosen_action, save_path, title="", action_names=None):
    """绘制Q值柱状图并保存
    action_names: 动作名称列表
    """
    n_actions = len(q_values)
    if action_names is None:
        action_names = [f'Action{i}' for i in range(n_actions)]

    colors = ['#2ecc71' if i == chosen_action else '#3498db' for i in range(n_actions)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(n_actions), q_values, color=colors, edgecolor='black', linewidth=1.2)

    # 高亮选中动作
    bars[chosen_action].set_edgecolor('#e74c3c')
    bars[chosen_action].set_linewidth(3)

    ax.set_xlabel('Action', fontsize=14)
    ax.set_ylabel('Q Value', fontsize=14)
    ax.set_title(f'{title}\nChosen: {action_names[chosen_action]} (Q={q_values[chosen_action]:.3f})', fontsize=14)
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(action_names, fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for i, (bar, q) in enumerate(zip(bars, q_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{q:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def run_visualization(checkpoint_path, version, action_num, obs_frames, output_dir, max_frames=300):
    """运行Q值可视化"""
    print(f"\n{'='*50}")
    print(f"Config: version={version}, actions={action_num}, frames={obs_frames}")
    print(f"Model: {checkpoint_path}")
    print(f"{'='*50}")

    # 根据动作数量选择标签
    if action_num == 2:
        ACTION_NAMES = ACTION_NAMES_2
    elif action_num == 7:
        ACTION_NAMES = ACTION_NAMES_7
    elif action_num == 12:
        ACTION_NAMES = ACTION_NAMES_12
    else:
        ACTION_NAMES = [f'Action{i}' for i in range(action_num)]

    # 加载配置
    cfg = compile_config(mario_dqn_config, create_cfg=mario_dqn_create_config, auto=True, save_cfg=False)

    # 更新配置
    cfg.policy.model.obs_shape = [obs_frames, 84, 84]
    cfg.policy.model.action_shape = action_num

    # 创建模型
    model = DQN(**cfg.policy.model)

    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    print("Model loaded successfully!")

    # 创建环境
    env = create_env(version=version, action_num=action_num, obs_frames=obs_frames)

    # 创建策略
    policy = DQNPolicy(cfg.policy, model=model).eval_mode

    # 设置随机种子
    env.seed(0)
    set_pkg_seed(0, use_cuda=cfg.policy.cuda)

    # 运行游戏
    obs = env.reset()
    frame_count = 0
    saved_count = 0

    print("\nStarting game (Press Ctrl+C to stop)...")

    try:
        while frame_count < max_frames:
            # 获取Q值
            q_values, chosen_action = get_q_values(policy, obs)

            # 每隔一定帧保存一次可视化
            if frame_count % 30 == 0:
                bar_path = f"{output_dir}/q_bar_v{version}_f{frame_count:04d}.png"
                draw_q_value_bar(q_values, chosen_action, bar_path,
                                 title=f"v{version}_{action_num}a_{obs_frames}f - Frame {frame_count}",
                                 action_names=ACTION_NAMES)
                saved_count += 1

            # 打印Q值信息
            if frame_count % 10 == 0:
                print(f"\nFrame {frame_count}: Chosen [{ACTION_NAMES[chosen_action]}] Q={q_values[chosen_action]:.3f}")
                for i, (name, q) in enumerate(zip(ACTION_NAMES, q_values)):
                    marker = " *" if i == chosen_action else ""
                    print(f"  {name}: {q:.4f}{marker}")

            # 执行动作
            obs, reward, done, info = env.step(chosen_action)
            frame_count += 1

            if done:
                print(f"\nGame Over! Frame={frame_count}")
                break

    except KeyboardInterrupt:
        print(f"\nUser interrupted!")

    print(f"Saved {saved_count} Q-value visualization images")
    env.close()


def compare_q_distributions(checkpoints, output_dir):
    """对比不同配置的Q值分布"""
    print("\n" + "="*60)
    print("Q-Value Distribution Comparison")
    print("="*60)

    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5*len(checkpoints), 6))
    if len(checkpoints) == 1:
        axes = [axes]

    cfg = compile_config(mario_dqn_config, create_cfg=mario_dqn_create_config, auto=True, save_cfg=False)

    for idx, (ckpt_path, (version, action_num, obs_frames), label) in enumerate(checkpoints):
        # 根据动作数量选择标签
        if action_num == 2:
            ACTION_NAMES = ACTION_NAMES_2
        elif action_num == 7:
            ACTION_NAMES = ACTION_NAMES_7
        elif action_num == 12:
            ACTION_NAMES = ACTION_NAMES_12
        else:
            ACTION_NAMES = [f'Action{i}' for i in range(action_num)]
        # 加载模型
        cfg.policy.model.obs_shape = [obs_frames, 84, 84]
        cfg.policy.model.action_shape = action_num
        model = DQN(**cfg.policy.model)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])

        # 运行环境获取一些Q值样本
        env = create_env(version=version, action_num=action_num, obs_frames=obs_frames)
        policy = DQNPolicy(cfg.policy, model=model).eval_mode
        env.seed(0)

        q_samples = []
        obs = env.reset()
        for _ in range(200):  # 收集200帧的Q值
            q_values, _ = get_q_values(policy, obs)
            q_samples.append(q_values)
            obs, _, done, _ = env.step(0)  # NOOP
            if done:
                obs = env.reset()

        q_samples = np.array(q_samples)  # shape: (200, n_actions)
        env.close()

        # 绘制箱线图
        ax = axes[idx]
        bp = ax.boxplot(q_samples, labels=ACTION_NAMES, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)

        ax.set_xlabel('Action', fontsize=12)
        ax.set_ylabel('Q Value', fontsize=12)
        ax.set_title(f'{label}\n(meanQ={q_samples.mean():.3f}, std={q_samples.std():.3f})', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    compare_path = f"{output_dir}/q_distribution_comparison.png"
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved: {compare_path}")


def create_step_comparison_plot(step_q_data, action_names, label, output_dir, version):
    """创建第40个envstep前后的Q值对比图"""
    steps = sorted(step_q_data.keys())
    n_actions = len(action_names)

    # 创建图表：上方是热力图，下方是折线图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 准备数据
    q_matrix = np.array([step_q_data[s][0] for s in steps])  # shape: (n_steps, n_actions)
    chosen_actions = [step_q_data[s][1] for s in steps]

    # 上图：热力图显示Q值
    ax1 = axes[0]
    im = ax1.imshow(q_matrix.T, aspect='auto', cmap='RdYlGn')
    ax1.set_yticks(range(n_actions))
    ax1.set_yticklabels(action_names)
    ax1.set_xticks(range(len(steps)))
    ax1.set_xticklabels([str(s) for s in steps])
    ax1.set_xlabel('Env Step', fontsize=12)
    ax1.set_ylabel('Action', fontsize=12)
    ax1.set_title(f'{label} - Q-Value Heatmap (Step {ENEMY_ENCOUNTER_STEP} +/- {ANALYSIS_RANGE})', fontsize=14)

    # 标记选中的动作
    for i, (step, action) in enumerate(zip(steps, chosen_actions)):
        ax1.plot(i, action, 'ko', markersize=8)
        if step == ENEMY_ENCOUNTER_STEP:
            ax1.axvline(x=i, color='red', linestyle='--', linewidth=2, alpha=0.7)

    plt.colorbar(im, ax=ax1, label='Q Value')

    # 下图：折线图显示每个动作的Q值变化
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_actions))

    for action_idx in range(n_actions):
        q_values_for_action = q_matrix[:, action_idx]
        ax2.plot(steps, q_values_for_action, 'o-', label=action_names[action_idx],
                 color=colors[action_idx], linewidth=2, markersize=6)

    # 标记第40步
    ax2.axvline(x=ENEMY_ENCOUNTER_STEP, color='red', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Enemy (Step {ENEMY_ENCOUNTER_STEP})')

    ax2.set_xlabel('Env Step', fontsize=12)
    ax2.set_ylabel('Q Value', fontsize=12)
    ax2.set_title(f'{label} - Q-Value Trend Around Enemy Encounter', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # 使用更详细的文件名，包含动作数量
    save_path = f"{output_dir}/enemy_step_analysis_v{version}_{n_actions}a.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Step comparison saved: {save_path}")


def create_enemy_encounter_analysis(checkpoints, output_dir):
    """专门分析遇到敌人时（第40个envstep前后）的Q值分布"""
    print("\n" + "="*60)
    print(f"Enemy Encounter Q-Value Analysis (Step {ENEMY_ENCOUNTER_STEP} +/- {ANALYSIS_RANGE})")
    print("="*60)

    cfg = compile_config(mario_dqn_config, create_cfg=mario_dqn_create_config, auto=True, save_cfg=False)

    for ckpt_path, (version, action_num, obs_frames), label in checkpoints:
        # 根据动作数量选择标签
        if action_num == 2:
            ACTION_NAMES = ACTION_NAMES_2
        elif action_num == 7:
            ACTION_NAMES = ACTION_NAMES_7
        elif action_num == 12:
            ACTION_NAMES = ACTION_NAMES_12
        else:
            ACTION_NAMES = [f'Action{i}' for i in range(action_num)]

        # 加载模型
        cfg.policy.model.obs_shape = [obs_frames, 84, 84]
        cfg.policy.model.action_shape = action_num
        model = DQN(**cfg.policy.model)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])

        # 创建环境
        env = create_env(version=version, action_num=action_num, obs_frames=obs_frames)
        policy = DQNPolicy(cfg.policy, model=model).eval_mode
        env.seed(0)

        # 收集第40个envstep前后的Q值
        step_q_data = {}  # {step: (q_values, chosen_action)}
        obs = env.reset()

        start_step = max(0, ENEMY_ENCOUNTER_STEP - ANALYSIS_RANGE)
        end_step = ENEMY_ENCOUNTER_STEP + ANALYSIS_RANGE + 1

        for step in range(end_step + 5):
            q_values, chosen_action = get_q_values(policy, obs)

            if start_step <= step <= end_step:
                step_q_data[step] = (q_values.copy(), chosen_action)

            obs, _, done, _ = env.step(chosen_action)
            if done:
                print(f"Warning: Game ended at step {step}")
                break

        env.close()

        # 创建详细的Q值对比图
        create_step_comparison_plot(step_q_data, ACTION_NAMES, label, output_dir, version)

        # 打印详细的Q值信息
        print(f"\n{'='*50}")
        print(f"Model: {label}")
        print(f"{'='*50}")
        for step in sorted(step_q_data.keys()):
            q_vals, action = step_q_data[step]
            marker = " <-- ENEMY ENCOUNTER" if step == ENEMY_ENCOUNTER_STEP else ""
            print(f"\nStep {step}{marker}:")
            print(f"  Chosen Action: {ACTION_NAMES[action]} (Q={q_vals[action]:.4f})")
            print(f"  All Q-values:")
            for i, (name, q) in enumerate(zip(ACTION_NAMES, q_vals)):
                star = " *" if i == action else ""
                print(f"    {name}: {q:.4f}{star}")


if __name__ == "__main__":
    import os
    os.makedirs("./q_visualization", exist_ok=True)

    # Config list: [(checkpoint_path, (version, action_num, obs_frames), label)]
    checkpoints = [
        # Baseline: v0 original map, 7 actions, 4 frames (best model)
        ("./exp/v0_7a_4f_seed0/ckpt/ckpt_best.pth.tar", (0, 7, 4), "Baseline: v0_7a_4f"),
        # Improved: v1 downsampled map, 2 actions, 4 frames
        ("./exp/v0_2a_4f_seed0/ckpt/ckpt_best.pth.tar", (0, 2, 4), "Improved: v0_2a_4f"),
    ]

    # Check if files exist
    valid_checkpoints = []
    for ckpt, cfg, lbl in checkpoints:
        if os.path.exists(ckpt):
            valid_checkpoints.append((ckpt, cfg, lbl))
            print(f"[OK] {lbl}: {ckpt}")
        else:
            print(f"[X] {lbl}: {ckpt} not found")

    if not valid_checkpoints:
        print("\nError: No valid model files found!")
        exit(1)

    # Run visualization
    print("\n" + "="*50)
    print("Running Q-value visualization...")
    print("="*50)

    for ckpt_path, config, label in valid_checkpoints:
        run_visualization(ckpt_path, config[0], config[1], config[2], "./q_visualization", max_frames=200)

    # Comparison analysis - focus on enemy encounter at step 40
    if len(valid_checkpoints) >= 1:
        print("\n" + "="*50)
        print(f"Generating enemy encounter analysis (Step {ENEMY_ENCOUNTER_STEP})...")
        print("="*50)
        create_enemy_encounter_analysis(valid_checkpoints, "./q_visualization")

    if len(valid_checkpoints) >= 2:
        print("\n" + "="*50)
        print("Generating comparison plots...")
        print("="*50)
        compare_q_distributions(valid_checkpoints, "./q_visualization")

    print("\n" + "="*50)
    print("Done! All images saved in ./q_visualization directory")
    print("="*50)
