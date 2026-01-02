"""
马里奥 DQN 实时游玩 + Q值可视化
结合游戏画面和 Q 值柱状图，生成可视化视频

功能：
1. 实时显示游戏画面
2. 在画面旁边显示 Q 值柱状图
3. 生成带 Q 值信息的视频
4. 生成全景图 + Q 值热力图
"""

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from ding.envs import DingEnvWrapper
from model import DQN
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, ExtraInfoWrapper
import torch
import numpy as np
import cv2
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# 动作名称映射
ACTION_NAMES = {
    2: ['Right', 'Right+A'],
    7: ['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'Jump', 'Left'],
    12: ['NOOP', 'Right', 'Right+A', 'Right+B', 'Right+A+B', 'Jump',
         'Left', 'Left+A', 'Left+B', 'Left+A+B', 'Down', 'Up']
}

# 动作空间映射
ACTION_SPACES = {
    2: [["right"], ["right", "A"]],
    7: SIMPLE_MOVEMENT,
    12: COMPLEX_MOVEMENT
}


class MarioQValueVisualizer:
    """马里奥 DQN 实时 Q 值可视化器"""

    def __init__(self, checkpoint_path, obs_frames=4, action_num=7, version=0):
        """
        初始化可视化器

        Args:
            checkpoint_path: 模型权重路径
            obs_frames: 观察帧数
            action_num: 动作数量 (2, 7, 或 12)
            version: 游戏版本 (0 或 1)
        """
        self.obs_frames = obs_frames
        self.action_num = action_num
        self.version = version
        self.action_names = ACTION_NAMES.get(action_num, [f'A{i}' for i in range(action_num)])

        print(f"{'='*60}")
        print(f"马里奥 DQN Q值可视化器")
        print(f"{'='*60}")
        print(f"模型: {checkpoint_path}")
        print(f"配置: version={version}, actions={action_num}, frames={obs_frames}")
        print(f"动作空间: {self.action_names}")

        # 创建环境
        self._create_env()

        # 加载模型
        self._load_model(checkpoint_path)

        # 游戏画面尺寸
        self.game_width = 256
        self.game_height = 240

        # Q值历史记录
        self.q_history = deque(maxlen=100)

        # 全景图相关
        self.panorama_width = 3500
        self.panorama = np.zeros((self.game_height, self.panorama_width, 3), dtype=np.uint8)
        self.filled_x = 0

        # 位置和Q值记录
        self.positions = []  # [(world_x, world_y, q_values, action), ...]

    def _create_env(self):
        """创建游戏环境"""
        actions = ACTION_SPACES.get(self.action_num, SIMPLE_MOVEMENT)

        self.raw_env = gym_super_mario_bros.make(f"SuperMarioBros-1-1-v{self.version}")
        base_env = JoypadSpace(self.raw_env, actions)

        self.env = DingEnvWrapper(
            base_env,
            cfg={
                'env_wrapper': [
                    lambda env: MaxAndSkipWrapper(env, skip=4),
                    lambda env: WarpFrameWrapper(env, size=84),
                    lambda env: ScaledFloatFrameWrapper(env),
                    lambda env: FrameStackWrapper(env, n_frames=self.obs_frames),
                    lambda env: ExtraInfoWrapper(env),
                ]
            }
        )
        print("环境创建成功!")

    def _load_model(self, checkpoint_path):
        """加载模型权重"""
        self.model = DQN(
            obs_shape=[self.obs_frames, 84, 84],
            action_shape=self.action_num,
            encoder_hidden_size_list=[32, 64, 128],
            dueling=False
        )
        self.model.eval()

        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict)
        print("模型加载成功!")

    def get_q_values(self, obs):
        """获取 Q 值"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self.model(obs_tensor)
            q_np = q_values.numpy().flatten()
            action = q_values.argmax(dim=1).item()
        return q_np, action

    def draw_q_bar(self, q_values, chosen_action, width=350, height=220):
        """
        绘制 Q 值柱状图

        Returns:
            numpy array: RGB 图像
        """
        # 创建白色背景
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        n_actions = len(q_values)
        bar_width = (width - 20) // n_actions
        margin = 10

        # 计算 Q 值范围
        q_min, q_max = q_values.min(), q_values.max()
        q_range = max(q_max - q_min, 0.1)

        # 绘制标题
        cv2.putText(img, "Q-Values", (width//2 - 40, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 绘制柱状图
        bar_area_height = height - 80
        bar_bottom = height - 45

        for i, (q, name) in enumerate(zip(q_values, self.action_names)):
            x = margin + i * bar_width

            # 归一化高度
            normalized = (q - q_min) / q_range
            bar_height = int(normalized * bar_area_height * 0.8)
            bar_height = max(5, bar_height)

            # 颜色：选中的动作用绿色，其他用蓝色
            if i == chosen_action:
                color = (0, 200, 0)  # 绿色
                border_color = (0, 100, 0)
            else:
                color = (200, 100, 50)  # 蓝色
                border_color = (100, 50, 25)

            # 绘制柱子
            cv2.rectangle(img,
                          (x, bar_bottom - bar_height),
                          (x + bar_width - 4, bar_bottom),
                          color, -1)
            cv2.rectangle(img,
                          (x, bar_bottom - bar_height),
                          (x + bar_width - 4, bar_bottom),
                          border_color, 1)

            # 绘制 Q 值
            q_text = f"{q:.1f}"
            cv2.putText(img, q_text, (x, bar_bottom - bar_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)

            # 绘制完整动作名称（旋转45度效果用两行显示）
            # 将名称分成两部分显示
            if '+' in name:
                parts = name.split('+')
                cv2.putText(img, parts[0], (x, bar_bottom + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)
                cv2.putText(img, '+' + '+'.join(parts[1:]), (x, bar_bottom + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)
            else:
                cv2.putText(img, name, (x, bar_bottom + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)

        return img

    def draw_q_trend(self, width=300, height=100):
        """
        绘制 Q 值趋势图

        Returns:
            numpy array: RGB 图像
        """
        img = np.ones((height, width, 3), dtype=np.uint8) * 240

        if len(self.q_history) < 2:
            cv2.putText(img, "Q-Value Trend", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            return img

        # 获取最大 Q 值历史
        max_q_history = [max(q) for q in self.q_history]

        # 归一化
        q_min, q_max = min(max_q_history), max(max_q_history)
        q_range = max(q_max - q_min, 0.1)

        # 绘制折线
        points = []
        for i, q in enumerate(max_q_history):
            x = int(i * (width - 20) / len(max_q_history)) + 10
            y = height - 20 - int((q - q_min) / q_range * (height - 40))
            points.append((x, y))

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (0, 100, 200), 2)

        # 标题
        cv2.putText(img, f"Max Q: {max_q_history[-1]:.2f}", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return img

    def create_combined_frame(self, game_frame, q_values, action, step, reward):
        """
        创建组合画面：游戏画面 + Q值柱状图 + 信息面板

        Returns:
            numpy array: 组合后的 RGB 图像
        """
        # Q值柱状图（使用更宽的尺寸以显示完整动作名称）
        q_bar = self.draw_q_bar(q_values, action, width=350, height=220)

        # Q值趋势图
        q_trend = self.draw_q_trend(width=350, height=100)

        # 信息面板（宽度与Q值柱状图一致）
        info_panel = np.ones((120, 350, 3), dtype=np.uint8) * 230
        cv2.putText(info_panel, f"Step: {step}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(info_panel, f"Reward: {reward:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(info_panel, f"Action: {self.action_names[action]}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(info_panel, f"Max Q: {max(q_values):.3f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
        cv2.putText(info_panel, f"Min Q: {min(q_values):.3f}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 1)

        # 右侧面板：Q值柱状图 + 趋势图 + 信息
        right_panel = np.vstack([q_bar, q_trend, info_panel])

        # 调整游戏画面大小以匹配右侧面板高度
        target_height = right_panel.shape[0]
        scale = target_height / game_frame.shape[0]
        new_width = int(game_frame.shape[1] * scale)
        game_resized = cv2.resize(game_frame, (new_width, target_height))

        # 组合
        combined = np.hstack([game_resized, right_panel])

        return combined

    def run_with_visualization(self, max_steps=2000, save_video=True,
                               video_path="mario_qvalue_play.mp4"):
        """
        运行游戏并实时可视化 Q 值

        Args:
            max_steps: 最大步数
            save_video: 是否保存视频
            video_path: 视频保存路径
        """
        print(f"\n{'='*50}")
        print("开始游戏 + Q值可视化...")
        print(f"{'='*50}")

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        step = 0
        total_reward = 0.0

        # 视频写入器
        video_writer = None
        frames = []

        try:
            while not done and step < max_steps:
                # 获取 Q 值和动作
                q_values, action = self.get_q_values(obs)
                self.q_history.append(q_values.copy())

                # 获取原始游戏画面
                raw_frame = self.raw_env.render(mode='rgb_array')
                if raw_frame is None:
                    raw_frame = np.zeros((240, 256, 3), dtype=np.uint8)

                # 创建组合画面
                combined = self.create_combined_frame(
                    raw_frame, q_values, action, step, total_reward
                )

                if save_video:
                    frames.append(combined.copy())

                # 获取位置信息
                try:
                    raw_info = self.raw_env.unwrapped._get_info()
                    world_x = raw_info.get('x_pos', 0)
                    world_y = raw_info.get('y_pos', 79)
                except:
                    world_x, world_y = 0, 79

                # 记录数据
                self.positions.append((world_x, world_y, q_values.copy(), action))

                # 更新全景图
                self._update_panorama(raw_frame, world_x)

                # 执行动作
                obs, reward, done, info = self.env.step(action)
                if isinstance(obs, tuple):
                    obs = obs[0]

                if isinstance(reward, np.ndarray):
                    reward = float(reward.item()) if reward.size == 1 else float(reward.sum())
                total_reward += reward

                step += 1

                if step % 50 == 0:
                    print(f"Step {step}: pos={world_x:.0f}, "
                          f"action={self.action_names[action]}, "
                          f"maxQ={max(q_values):.2f}, reward={total_reward:.0f}")

        except KeyboardInterrupt:
            print("\n用户中断!")

        print(f"\n游戏结束! 步数: {step}, 得分: {total_reward:.0f}")

        # 保存视频
        if save_video and frames:
            self._save_video(frames, video_path)

        return total_reward

    def _update_panorama(self, frame, world_x):
        """更新全景图"""
        if world_x < 128:
            screen_start_x = 0
        else:
            screen_start_x = world_x - 128

        if screen_start_x + self.game_width > self.filled_x:
            update_start = max(self.filled_x, int(screen_start_x))
            update_end = min(self.panorama_width, int(screen_start_x) + self.game_width)

            frame_start = update_start - int(screen_start_x)
            frame_end = update_end - int(screen_start_x)

            if frame_end > frame_start and update_end > update_start:
                copy_width = min(frame_end - frame_start, update_end - update_start)
                if update_start + copy_width <= self.panorama_width:
                    self.panorama[:, update_start:update_start + copy_width] = \
                        frame[:, frame_start:frame_start + copy_width]
                    self.filled_x = update_start + copy_width

    def _save_video(self, frames, video_path, fps=15):
        """保存视频"""
        print(f"\n保存视频: {video_path}")

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for frame in frames:
            # RGB -> BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"视频已保存: {video_path} ({len(frames)} 帧)")

    def generate_panorama_with_qvalues(self, output_path="mario_panorama_qvalues.png"):
        """
        生成带 Q 值热力图的全景图
        """
        if not self.positions:
            print("没有位置数据!")
            return

        max_x = int(max(p[0] for p in self.positions)) + 50
        max_x = min(max_x, self.panorama_width)

        result = self.panorama[:, :max_x].copy()

        print(f"\n生成全景图 (尺寸: {max_x} x {self.game_height})...")

        # 绘制 Q 值热力图轨迹
        total = len(self.positions)
        step = max(1, total // 100)

        for i in range(0, total, step):
            wx, wy, q_values, action = self.positions[i]
            if wx >= max_x:
                continue

            # 马里奥的图像 y 坐标
            img_y = 208 - (wy - 79)
            img_y = max(16, min(self.game_height - 5, int(img_y)))

            # 根据最大 Q 值确定颜色
            max_q = max(q_values)
            # 归一化到 0-1
            q_normalized = min(1.0, max(0.0, (max_q + 5) / 15))

            # 颜色：蓝(低Q) -> 绿(中Q) -> 红(高Q)
            if q_normalized < 0.5:
                r = int(q_normalized * 2 * 255)
                g = int(q_normalized * 2 * 255)
                b = int((1 - q_normalized * 2) * 255)
            else:
                r = 255
                g = int((1 - (q_normalized - 0.5) * 2) * 255)
                b = 0

            color = (r, g, b)
            radius = 3 + int(q_normalized * 3)

            cv2.circle(result, (int(wx), img_y), radius, color, -1)
            cv2.circle(result, (int(wx), img_y), radius, (0, 0, 0), 1)

        # 添加图例
        self._add_legend(result)

        # 保存
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"全景图已保存: {output_path}")

        return result

    def _add_legend(self, img):
        """添加 Q 值图例"""
        legend_x = 20
        legend_y = 20
        legend_width = 150
        legend_height = 60

        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (legend_x, legend_y),
                      (legend_x + legend_width, legend_y + legend_height),
                      (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # 标题
        cv2.putText(img, "Q-Value", (legend_x + 5, legend_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 渐变条
        for i in range(100):
            x = legend_x + 5 + i
            q_norm = i / 100.0
            if q_norm < 0.5:
                r = int(q_norm * 2 * 255)
                g = int(q_norm * 2 * 255)
                b = int((1 - q_norm * 2) * 255)
            else:
                r = 255
                g = int((1 - (q_norm - 0.5) * 2) * 255)
                b = 0
            cv2.line(img, (x, legend_y + 25), (x, legend_y + 40), (r, g, b), 1)

        # 标签
        cv2.putText(img, "Low", (legend_x + 5, legend_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(img, "High", (legend_x + 85, legend_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    def close(self):
        """关闭环境"""
        self.env.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='马里奥 DQN Q值可视化')
    parser.add_argument('--checkpoint', '-c', type=str,
                        default='./exp/v0_2a_4f_seed0/ckpt/ckpt_best.pth.tar',
                        help='模型权重路径')
    parser.add_argument('--frames', '-f', type=int, default=4,
                        help='观察帧数 (默认: 4)')
    parser.add_argument('--actions', '-a', type=int, default=2,
                        help='动作数量 (2, 7, 或 12, 默认: 7)')
    parser.add_argument('--version', '-v', type=int, default=0,
                        help='游戏版本 (0 或 1, 默认: 0)')
    parser.add_argument('--max-steps', '-s', type=int, default=2000,
                        help='最大步数 (默认: 2000)')
    parser.add_argument('--no-video', action='store_true',
                        help='不保存视频')
    parser.add_argument('--output', '-o', type=str, default='./q_visualization',
                        help='输出目录 (默认: ./q_visualization)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 检查模型文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件不存在 {args.checkpoint}")
        return

    # 创建可视化器
    visualizer = MarioQValueVisualizer(
        checkpoint_path=args.checkpoint,
        obs_frames=args.frames,
        action_num=args.actions,
        version=args.version
    )

    try:
        # 运行游戏
        video_path = os.path.join(args.output,
                                  f"mario_v{args.version}_{args.actions}a_{args.frames}f.mp4")

        visualizer.run_with_visualization(
            max_steps=args.max_steps,
            save_video=not args.no_video,
            video_path=video_path
        )

        # 生成全景图
        panorama_path = os.path.join(args.output,
                                     f"panorama_v{args.version}_{args.actions}a_{args.frames}f.png")
        visualizer.generate_panorama_with_qvalues(panorama_path)

    finally:
        visualizer.close()

    print(f"\n{'='*50}")
    print("完成! 输出文件:")
    print(f"  视频: {video_path}")
    print(f"  全景图: {panorama_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
