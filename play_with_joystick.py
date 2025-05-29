import random
import pygame
import time
from play_with_human import HumanGameplay
import numpy as np
import keyboard

class JoystickAIGameplay(HumanGameplay):
    def __init__(self):
        super().__init__(threshold=0)
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise ValueError("No joystick found!")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Using joystick: {self.joystick.get_name()}")

        self.current_ai_action = 4

    def sample_ai_action(self, obs):
        pygame.event.pump()

        x_axis = self.joystick.get_axis(0)
        y_axis = self.joystick.get_axis(1)
        button = self.joystick.get_button(0)
        
        if abs(x_axis) > 0.5 or abs(y_axis) > 0.5:
            if abs(x_axis) > abs(y_axis):
                if x_axis > 0:
                    self.current_ai_action = 0  # 右
                else:
                    self.current_ai_action = 2  # 左
            else:
                if y_axis > 0:
                    self.current_ai_action = 1  # 下
                else:
                    self.current_ai_action = 3  # 上
        elif button:
            self.current_ai_action = 4  # 交互
            
        return self.current_ai_action

    def cleanup(self):
        pygame.quit()

def wait_for_space():
    """等待空格键按下"""
    print("\n按空格键继续...")
    while True:
        if keyboard.is_pressed('space'):
            # 等待空格键释放
            while keyboard.is_pressed('space'):
                time.sleep(0.1)
            break
        time.sleep(0.1)

def run_game_session(threshold):
    """运行一次游戏会话"""
    try:
        if threshold == 0:
            print("\nAI agent由摇杆控制，human agent由键盘控制")
            game = JoystickAIGameplay()
        else:
            print(f"\nAI agent使用threshold={threshold}的模型，human agent由键盘控制")
            game = HumanGameplay(threshold=threshold)
            
        game.run_game()
        
    except KeyboardInterrupt:
        print("\n游戏被中断")
        return False
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        return False
    finally:
        if threshold == 0 and 'game' in locals():
            game.cleanup()
    return True

def main():
    # 初始threshold列表
    thresholds = [0.2, 0.4, 0.6, 0.8, 0]
    # 随机打乱顺序
    random.shuffle(thresholds)
    
    print("\n=== 游戏将进行5轮，每轮使用不同的threshold值 ===")
    print("当前threshold序列:", thresholds)
    
    for i, threshold in enumerate(thresholds, 1):
        print(f"\n\n=== 第{i}/5轮游戏 ===")
        print(f"使用threshold值: {threshold}")
        
        if not run_game_session(threshold):
            break
            
        # 如果不是最后一轮，等待空格继续
        if i < len(thresholds):
            wait_for_space()
    
    print("\n=== 所有游戏结束 ===")

if __name__ == "__main__":
    main() 