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
                    self.current_ai_action = 0
                else:
                    self.current_ai_action = 2
            else:
                if y_axis > 0:
                    self.current_ai_action = 1
                else:
                    self.current_ai_action = 3
        elif button:
            self.current_ai_action = 4
            
        return self.current_ai_action

    def cleanup(self):
        pygame.quit()

def wait_for_space():
    print("\n\n\n\n\n\n\nPress space to continue...\n\n\n\n\n\n")
    while True:
        if keyboard.is_pressed('space'):
            while keyboard.is_pressed('space'):
                time.sleep(0.1)
            break
        time.sleep(0.1)

def run_game_session(threshold):
    try:
        if threshold == 0:
            game = JoystickAIGameplay()
        else:
            game = HumanGameplay(threshold=threshold)
            
        game.run_game()
        
    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False
    finally:
        if threshold == 0 and 'game' in locals():
            game.cleanup()
    return True

def main():
    thresholds = [0.2, 0.4, 0.6, 0.8, 0]
    random.shuffle(thresholds)

    
    for i, threshold in enumerate(thresholds, 1):
        print(f"\n\n===== Game {i}/5 =====")
        
        if not run_game_session(threshold):
            break

        if i < len(thresholds):
            wait_for_space()
    
    print("\n=== All Trial Ended ===")

if __name__ == "__main__":
    main() 