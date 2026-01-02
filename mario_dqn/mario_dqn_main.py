"""
智能体训练入口，包含训练逻辑
"""
import torch
import gym_super_mario_bros
from functools import partial
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from mario_dqn_config import mario_dqn_config
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed
from model import DQN
from policy import DQNPolicy
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv, CoinRewardWrapper, MushroomRewardWrapper, ExtraInfoWrapper, CoinRewardWrapper, MushroomRewardWrapper
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper, BaseEnvManager
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.config import compile_config
from tensorboardX import SummaryWriter
import os
import gym_super_mario_bros


# 动作相关配置
action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]


# mario环境
def wrapped_mario_env(version=0, action=7, obs=1):
    return DingEnvWrapper(
        # 设置mario游戏版本与动作空间
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v"+str(version)), action_dict[int(action)]),
        cfg={
            # 添加各种wrapper
            'env_wrapper': [
                # 默认wrapper：跳帧以降低计算量
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # 默认wrapper：将mario游戏环境图片进行处理，返回大小为84X84的图片observation
                lambda env: WarpFrameWrapper(env, size=84),
                # 默认wrapper：将observation数值进行归一化
                lambda env: ScaledFloatFrameWrapper(env),
                # 默认wrapper：叠帧，将连续n_frames帧叠到一起，返回shape为(n_frames,84,84)的图片observation
                lambda env: FrameStackWrapper(env, n_frames=obs),

                # 奖励塑形：金币和蘑菇奖励
                lambda env: CoinRewardWrapper(env),
                lambda env: MushroomRewardWrapper(env),

                # 记录额外信息用于 TensorBoard 可视化（金币、状态等）
                lambda env: ExtraInfoWrapper(env),

                # 默认wrapper：在评估一局游戏结束时返回累计的奖励，方便统计
                lambda env: FinalEvalRewardEnv(env)
            ]
        }
    )


def main(cfg, args, seed=0, max_env_step=int(3e6)):
    # Easydict类实例，包含一些配置
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        seed=seed,
        save_cfg=True
    )
    # 收集经验的环境数量以及用于评估的环境数量
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # 收集经验的环境，使用并行环境管理器
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs)
                for _ in range(collector_env_num)],
        cfg=cfg.env.manager)
    # 评估性能的环境，使用并行环境管理器
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs)
                for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager)

    # 为mario环境设置种子
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    # 为torch、numpy、random等package设置种子
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # 设置 PyTorch 以减少内存碎片和优化 cuDNN
    if cfg.policy.cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # 禁用自动调优以减少内存使用
        torch.backends.cudnn.deterministic = True  # 使用确定性算法
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 采用DQN模型
    model = DQN(**cfg.policy.model)

    # 采用DQN策略
    policy = DQNPolicy(cfg.policy, model=model)

    # 设置学习、经验收集、评估、经验回放等强化学习常用配置
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    ckpt_path = args.resume_ckpt
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f'\n{"="*60}')
        print(f'加载检查点: {ckpt_path}')
        print(f'{"="*60}')

        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state['model'])
        print('✓ 已加载模型权重')

        # 同步 target network（防止 Q 值突变）
        # policy.learn_mode 返回的是一个包含 _target_model 的对象
        learn_mode = policy.learn_mode
        if hasattr(learn_mode, '_target_model'):
            learn_mode._target_model.load_state_dict(model.state_dict())
            print('✓ 已同步 target network')
        else:
            print('⚠ 未找到 target network，跳过同步')

        print("\n预热 Replay Buffer（使用低探索率）...")
        # 使用较低的 epsilon，主要利用已学策略
        warmup_eps = 0.1  # 降低探索率，更多利用已学策略

        # 收集 100 个 batch 的经验
        for i in range(200):
            new_data = collector.collect(n_sample=cfg.policy.collect.n_sample, policy_kwargs={'eps': warmup_eps})
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            if (i + 1) % 20 == 0:
                print(f'  进度: {i+1}/100, buffer size: {replay_buffer.count()}')

        print(f'✓ Buffer 预热完成: {replay_buffer.count()} samples')
        print(f'{"="*60}\n')

    # 设置epsilon greedy
    eps_cfg = cfg.policy.other.eps

    # 如果加载了权重，使用固定的低探索率（纯利用模式）
    if ckpt_path and os.path.isfile(ckpt_path):
        from easydict import EasyDict
        eps_cfg = EasyDict(eps_cfg.copy())
        eps_cfg.start = 0.05  # 固定为 0.05，不再衰减
        eps_cfg.end = 0.05
        eps_cfg.decay = 1
        print(f'✓ Epsilon 设置为固定值 0.05（利用模式，继续训练）\n')
    else:
        print(f'✓ Epsilon 从 {eps_cfg.start} 开始衰减到 {eps_cfg.end}（探索模式，从头训练）\n')

    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # 训练以及评估
    while True:
        # 根据当前训练迭代数决定是否进行评估
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # 更新epsilon greedy信息
        eps = epsilon_greedy(collector.envstep)
        # 经验收集器从环境中收集经验
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        # 将收集的经验放入replay buffer
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # 采样经验进行训练
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
        if collector.envstep >= max_env_step:
            break


if __name__ == "__main__":
    from copy import deepcopy
    import argparse
    parser = argparse.ArgumentParser()
    # 种子
    parser.add_argument("--seed", "-s", type=int, default=0)
    # 游戏版本，v0 v1 v2 v3 四种选择
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0, 1, 2, 3])
    # 动作集合种类，包含[["right"], ["right", "A"]]、SIMPLE_MOVEMENT、COMPLEX_MOVEMENT，分别对应2、7、12个动作
    parser.add_argument("--action", "-a", type=int, default=7, choices=[2, 7, 12])
    # 观测空间叠帧数目，不叠帧或叠四帧
    parser.add_argument("--obs", "-o", type=int, default=1, choices=[1, 4])
    parser.add_argument("--resume_ckpt", type=str, default='', help="Path to the checkpoint to resume from")
    args = parser.parse_args()
    mario_dqn_config.exp_name = 'exp/v'+str(args.version)+'_'+str(args.action)+'a_'+str(args.obs)+'f_seed'+str(args.seed)
    mario_dqn_config.policy.model.obs_shape = [args.obs, 84, 84]
    mario_dqn_config.policy.model.action_shape = args.action
    main(deepcopy(mario_dqn_config), args, seed=args.seed)
