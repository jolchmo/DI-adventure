"""
马里奥 DQN 可视化 - 生成超长横向全景图 + 马里奥轨迹残影

功能：
1. 运行 DQN 模型玩马里奥
2. 拼接整个关卡成一张超长横图
3. 在长图上绘制马里奥的移动轨迹（使用马里奥精灵图）
"""

from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, ExtraInfoWrapper
from model import DQN
from ding.envs import DingEnvWrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym_super_mario_bros
import torch
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MarioPanoramaGenerator:
    def __init__(self, checkpoint_path=None, obs_frames=4):
        if checkpoint_path is None:
            checkpoint_path = "exp/ckpt_best.pth.tar"

        print(f"加载权重: {checkpoint_path}")

        self.actions = SIMPLE_MOVEMENT
        print(f"动作空间: {self.actions}")

        # 加载环境
        self.raw_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        base_env = JoypadSpace(self.raw_env, self.actions)
        self.env = DingEnvWrapper(
            base_env,
            cfg={
                'env_wrapper': [
                    lambda env: MaxAndSkipWrapper(env, skip=4),
                    lambda env: WarpFrameWrapper(env, size=84),
                    lambda env: ScaledFloatFrameWrapper(env),
                    lambda env: FrameStackWrapper(env, n_frames=obs_frames),
                    lambda env: ExtraInfoWrapper(env),
                ]
            }
        )

        # 加载模型
        self.model = DQN(
            obs_shape=[obs_frames, 84, 84],
            action_shape=len(self.actions),
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

        # 游戏画面尺寸
        self.game_width = 256
        self.game_height = 240

        # 全景图 (1-1关卡约3400像素宽)
        self.panorama_width = 3500
        self.panorama = np.zeros((self.game_height, self.panorama_width, 3), dtype=np.uint8)
        self.filled_x = 0  # 已填充到的x位置

        # 马里奥位置记录
        self.mario_positions = []  # [(world_x, world_y), ...]

        # 加载马里奥精灵图
        self.mario_sprite = self._load_mario_sprite()

    def _load_mario_sprite(self):
        """加载马里奥精灵图"""
        sprite_path = os.path.join(os.path.dirname(__file__), "small_mario.png")
        if os.path.exists(sprite_path):
            # 读取带透明通道的图片
            sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
            if sprite is not None:
                # 转换 BGR(A) 到 RGB(A)
                if len(sprite.shape) == 3 and sprite.shape[2] == 4:
                    # BGRA -> RGBA
                    sprite = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)
                elif len(sprite.shape) == 3 and sprite.shape[2] == 3:
                    sprite = cv2.cvtColor(sprite, cv2.COLOR_BGR2RGB)
                print(f"马里奥精灵图加载成功: {sprite.shape}")
                return sprite
        print("警告: 未找到马里奥精灵图，将使用圆点代替")
        return None

    def _overlay_sprite(self, background, sprite, x, y, alpha=1.0):
        """将精灵图叠加到背景上（支持透明度）"""
        h, w = sprite.shape[:2]

        # 计算实际绘制区域 (x, y 是马里奥脚底中心位置)
        x1, y1 = int(x - w // 2), int(y - h)
        x2, y2 = x1 + w, y1 + h

        # 边界检查
        if x1 >= background.shape[1] or x2 <= 0 or y1 >= background.shape[0] or y2 <= 0:
            return

        # 裁剪到有效范围
        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        src_x2 = w - max(0, x2 - background.shape[1])
        src_y2 = h - max(0, y2 - background.shape[0])

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(background.shape[1], x2)
        dst_y2 = min(background.shape[0], y2)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return

        # 获取精灵区域
        sprite_region = sprite[src_y1:src_y2, src_x1:src_x2]
        bg_region = background[dst_y1:dst_y2, dst_x1:dst_x2]

        if sprite_region.shape[:2] != bg_region.shape[:2]:
            return

        # 处理透明度
        if len(sprite.shape) == 3 and sprite.shape[2] == 4:  # 有 alpha 通道
            sprite_alpha = (sprite_region[:, :, 3] / 255.0) * alpha
            sprite_alpha = sprite_alpha[:, :, np.newaxis]

            # 混合
            blended = (sprite_region[:, :, :3] * sprite_alpha +
                       bg_region * (1 - sprite_alpha)).astype(np.uint8)
            background[dst_y1:dst_y2, dst_x1:dst_x2] = blended
        else:
            # 无 alpha 通道，直接混合
            blended = cv2.addWeighted(bg_region, 1 - alpha, sprite_region[:, :, :3], alpha, 0)
            background[dst_y1:dst_y2, dst_x1:dst_x2] = blended

    def run_episode(self, max_steps=10000):
        """运行一个完整的episode，收集数据"""
        print("\n开始运行游戏...")

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:
            # DQN 推理
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                q_values = self.model(obs_tensor)
                action_idx = q_values.argmax(dim=1).item()

            # 执行动作
            obs, reward, done, info = self.env.step(action_idx)
            if isinstance(obs, tuple):
                obs = obs[0]

            if isinstance(reward, np.ndarray):
                reward = float(reward.item()) if reward.size == 1 else float(reward.sum())
            total_reward += reward

            # 获取原始画面
            raw_frame = self.raw_env.render(mode='rgb_array')
            if raw_frame is None:
                continue

            # 获取马里奥世界坐标
            try:
                raw_info = self.raw_env.unwrapped._get_info()
                world_x = raw_info.get('x_pos', 0)
                world_y = raw_info.get('y_pos', 79)
            except:
                if isinstance(info, dict):
                    world_x = info.get('x_pos', info.get('x', 0))
                    world_y = info.get('y_pos', info.get('y', 79))
                else:
                    world_x = 0
                    world_y = 79

            # 记录马里奥位置
            self.mario_positions.append((world_x, world_y))

            # 更新全景图
            if world_x < 128:
                screen_start_x = 0
            else:
                screen_start_x = world_x - 128

            # 只更新新出现的部分
            if screen_start_x + self.game_width > self.filled_x:
                update_start = max(self.filled_x, int(screen_start_x))
                update_end = min(self.panorama_width, int(screen_start_x) + self.game_width)

                frame_start = update_start - int(screen_start_x)
                frame_end = update_end - int(screen_start_x)

                if frame_end > frame_start and update_end > update_start:
                    copy_width = min(frame_end - frame_start, update_end - update_start)
                    if update_start + copy_width <= self.panorama_width:
                        self.panorama[:, update_start:update_start + copy_width] = \
                            raw_frame[:, frame_start:frame_start + copy_width]
                        self.filled_x = update_start + copy_width

            step += 1

            if step % 100 == 0:
                print(f"  步数: {step}, 位置: {world_x:.0f}, 得分: {total_reward:.0f}")

        print(f"\n游戏结束! 总步数: {step}, 最终得分: {total_reward:.0f}")
        print(f"马里奥最远到达: {max(p[0] for p in self.mario_positions):.0f}")

        return total_reward

    def generate_panorama_with_trajectory(self, output_path="mario_panorama.png"):
        """生成带轨迹的全景图"""
        if not self.mario_positions:
            print("没有位置数据!")
            return

        # 裁剪到实际宽度
        max_x = int(max(p[0] for p in self.mario_positions)) + 50
        max_x = min(max_x, self.panorama_width)

        result = self.panorama[:, :max_x].copy()

        print(f"\n生成全景图 (尺寸: {max_x} x {self.game_height})...")

        total = len(self.mario_positions)
        print(f"绘制马里奥轨迹 ({total} 个位置点)...")

        # 每隔几帧绘制一个马里奥残影
        step = max(1, total // 80)  # 最多绘制约80个残影

        for i in range(0, total, step):
            wx, wy = self.mario_positions[i]
            if wx >= max_x:
                continue

            # 进度 0->1
            progress = (i + 1) / total

            # 透明度：越早的越透明
            alpha = 0.2 + 0.8 * progress

            # 马里奥的图像y坐标
            img_y = 208 - (wy - 79)
            img_y = max(16, min(self.game_height - 5, int(img_y)))

            # 绘制马里奥精灵或圆点
            if self.mario_sprite is not None:
                self._overlay_sprite(result, self.mario_sprite, int(wx), img_y, alpha)
            else:
                # 备用：绘制圆点
                color = (
                    int(50 + 200 * progress),
                    int(50),
                    int(250 - 200 * progress)
                )
                radius = 2 + int(4 * progress)
                cv2.circle(result, (int(wx), img_y), radius, color, -1)

        # 保存 - OpenCV 使用 BGR，需要转换
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"\n✓ 全景图已保存: {output_path}")
        print(f"  尺寸: {result.shape[1]} x {result.shape[0]} 像素")

        return result

    def generate_trajectory_only(self, output_path="mario_trajectory.png"):
        """生成纯轨迹图（黑色背景）"""
        if not self.mario_positions:
            print("没有位置数据!")
            return

        max_x = int(max(p[0] for p in self.mario_positions)) + 50

        # 黑色背景
        result = np.zeros((self.game_height, max_x, 3), dtype=np.uint8)
        result[:] = (15, 15, 25)  # 深蓝黑色背景

        total = len(self.mario_positions)
        step = max(1, total // 80)

        # 绘制马里奥残影
        for i in range(0, total, step):
            wx, wy = self.mario_positions[i]
            if wx >= max_x:
                continue

            progress = (i + 1) / total
            alpha = 0.2 + 0.8 * progress

            img_y = 208 - (wy - 79)
            img_y = max(16, min(self.game_height - 5, int(img_y)))

            if self.mario_sprite is not None:
                self._overlay_sprite(result, self.mario_sprite, int(wx), img_y, alpha)
            else:
                color = (
                    int(80 + 175 * progress),
                    int(150),
                    int(250 - 175 * progress)
                )
                cv2.circle(result, (int(wx), img_y), 5, color, -1)

        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"✓ 轨迹图已保存: {output_path}")
        print(f"  尺寸: {result.shape[1]} x {result.shape[0]} 像素")

        return result

    def close(self):
        self.env.close()


def main():
    print("=" * 60)
    print("马里奥 DQN - 超长全景图生成器")
    print("=" * 60)

    checkpoint = "./exp/ckpt_best.pth.tar"

    if not os.path.exists(checkpoint):
        print(f"错误: 权重文件不存在 {checkpoint}")
        return

    generator = MarioPanoramaGenerator(checkpoint_path=checkpoint)

    try:
        # 运行游戏
        generator.run_episode(max_steps=10000)

        # 生成图片
        generator.generate_panorama_with_trajectory("mario_panorama.png")
        generator.generate_trajectory_only("mario_trajectory.png")

    finally:
        generator.close()

    print("\n完成!")


if __name__ == "__main__":
    main()
