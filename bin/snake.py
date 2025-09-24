import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import pygame
import random
from enum import Enum
from typing import Tuple, List, Optional
import time


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeGameEngine:
    """
    A simple Snake game engine designed for RL agent training.
    
    Features:
    - Grid-based game world
    - Simple action space (4 directions)
    - Observation space (game state as array)
    - Reward system
    - Visualization with pygame
    - Reset functionality
    """
    
    def __init__(self, width: int = 10, height: int = 10, cell_size: int = 40, show_window: bool = False):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Game state
        self.snake = []
        self.food = None
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.max_steps = width * height * 2  # Prevent infinite games
        
        # Pygame setup for visualization
        pygame.init()
        flags = 0 if show_window else pygame.HIDDEN
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size), flags=flags)
        pygame.display.set_caption("Snake RL Environment")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state and return observation."""
        # Initialize snake in the center
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = [(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        
        # Place food
        self._place_food()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
            
        Returns:
            observation: Current game state
            reward: Reward for this step
            done: Whether game is over
            info: Additional information
        """
        self.steps += 1
        
        # Update direction (prevent immediate reverse)
        new_direction = Direction(action)
        if not self._is_opposite_direction(new_direction, self.direction):
            self.direction = new_direction
        
        # Move snake
        head_x, head_y = self.snake[0]
        
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        
        # Check collisions
        reward = 0
        done = False
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            done = True
            reward = -10
        
        # Self collision
        elif new_head in self.snake:
            done = True
            reward = -10
        
        # Food collision
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            reward = 10
            self._place_food()
            
            # Win condition (snake fills the board)
            if len(self.snake) == self.width * self.height:
                done = True
                reward += 50
        
        else:
            # Normal move
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = -0.1  # Small negative reward to encourage efficiency
        
        # Timeout
        if self.steps >= self.max_steps:
            done = True
            reward -= 5
        
        info = {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake)
        }
        
        return self._get_observation(), reward, done, info
    
    def _place_food(self):
        """Place food randomly on empty cells."""
        empty_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))
        
        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            self.food = None  # No empty cells (shouldn't happen in normal play)
    
    def _is_opposite_direction(self, new_dir: Direction, current_dir: Direction) -> bool:
        """Check if new direction is opposite to current direction."""
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        return opposites[current_dir] == new_dir
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current game state as observation.
        
        Returns:
            np.ndarray: Game state representation
            - 0: Empty cell
            - 1: Snake body
            - 2: Snake head
            - 3: Food
        """
        obs = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Snake body
        for x, y in self.snake[1:]:
            if 0 <= x < self.width and 0 <= y < self.height:
                obs[y, x] = 1
        
        # Snake head
        if self.snake:
            head_x, head_y = self.snake[0]
            if 0 <= head_x < self.width and 0 <= head_y < self.height:
                obs[head_y, head_x] = 2
        
        # Food
        if self.food:
            food_x, food_y = self.food
            obs[food_y, food_x] = 3
        
        return obs
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get a flattened state vector for simpler NN input.
        
        Returns:
            np.ndarray: Flattened game state + additional features
        """
        obs = self._get_observation().flatten()
        
        # Additional features
        head_x, head_y = self.snake[0] if self.snake else (0, 0)
        food_x, food_y = self.food if self.food else (0, 0)
        
        additional_features = np.array([
            head_x / self.width,  # Normalized head position
            head_y / self.height,
            food_x / self.width,  # Normalized food position
            food_y / self.height,
            len(self.snake) / (self.width * self.height),  # Snake length ratio
            self.direction.value / 3,  # Normalized direction
        ])
        
        return np.concatenate([obs, additional_features])
    
    def render(self, mode: str = 'human'):
        """Render the game state."""
        if mode == 'human':
            self._render_pygame()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_pygame(self):
        """Render using pygame."""
        self.screen.fill(self.BLACK)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = self.WHITE if i == 0 else self.GREEN  # Head is white, body is green
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        # Draw food
        if self.food:
            food_x, food_y = self.food
            rect = pygame.Rect(food_x * self.cell_size, food_y * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.RED, rect)
        
        pygame.display.flip()
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for headless operation."""
        # Create a simple RGB representation
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Snake body (green)
        for x, y in self.snake[1:]:
            if 0 <= x < self.width and 0 <= y < self.height:
                rgb[y, x] = [0, 255, 0]
        
        # Snake head (white)
        if self.snake:
            head_x, head_y = self.snake[0]
            if 0 <= head_x < self.width and 0 <= head_y < self.height:
                rgb[head_y, head_x] = [255, 255, 255]
        
        # Food (red)
        if self.food:
            food_x, food_y = self.food
            rgb[food_y, food_x] = [255, 0, 0]
        
        return rgb
    
    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
    
    def play_human(self, fps: int = 10):
        """
        Play the game with human input for testing.
        Arrow keys to control, ESC to quit.
        """
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    action = None
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_LEFT:
                        action = 3
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    
                    if action is not None:
                        obs, reward, done, info = self.step(action)
                        print(f"Score: {info['score']}, Reward: {reward:.1f}")
                        
                        if done:
                            print(f"Game Over! Final Score: {info['score']}")
                            self.reset()
            
            self.render()
            self.clock.tick(fps)
        
        self.close()



# Action / State Encoder & Decoder for Snake Game (component of DAT)
# --------------------------------------------------------------------------------------------------------------------------

class StateEncoder(nn.Module):
    
    def __init__(self, height=10, width=10, feature_dim=128):
        super(StateEncoder, self).__init__()
        self.height = height
        self.width = width
        self.feature_dim = feature_dim
        self.input_channels = 4
        
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) if min(height, width) >= 4 else nn.Identity()
        
        h_after_pool = height
        w_after_pool = width
        if min(height, width) >= 4:
            h_after_pool = h_after_pool // 2
            w_after_pool = w_after_pool // 2
            
        conv_output_size = 16 * h_after_pool * w_after_pool
        
        self.fc1 = nn.Linear(conv_output_size, feature_dim)
        
    def _preprocess_state(self, x):
        batch_size, height, width = x.shape
        channels = torch.zeros(batch_size, 4, height, width, device=x.device)
        channels[:, 0] = (x == 0).float()
        channels[:, 1] = (x == 1).float() 
        channels[:, 2] = (x == 2).float()
        channels[:, 3] = (x == 3).float()
        return channels
    
    def forward(self, x):
        if len(x.shape) == 3:
            pass
        elif len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        else:
            raise ValueError(f"Expected input shape (B, H, W) or (B, 1, H, W), got {x.shape}")
            
        x = self._preprocess_state(x)
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x


class StateDecoder(nn.Module):
    
    def __init__(self, height=10, width=10, feature_dim=128):
        super(StateDecoder, self).__init__()
        self.height = height
        self.width = width
        self.feature_dim = feature_dim
        
        h_after_pool = height
        w_after_pool = width
        if min(height, width) >= 4:
            h_after_pool = h_after_pool // 2
            w_after_pool = w_after_pool // 2
            
        self.h_after_pool = h_after_pool
        self.w_after_pool = w_after_pool
        conv_output_size = 16 * h_after_pool * w_after_pool
        
        self.fc1 = nn.Linear(feature_dim, conv_output_size)
        self.upsample = nn.Upsample(size=(height, width), mode='nearest') if min(height, width) >= 4 else nn.Identity()
        self.deconv1 = nn.ConvTranspose2d(16, 4, kernel_size=3, padding=1)
        
    def forward(self, x, target): 
        batch_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, 16, self.h_after_pool, self.w_after_pool)
        x = self.upsample(x)
        x = self.deconv1(x)
        loss = F.cross_entropy(x, target.long())
        return loss
        

    def generate(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = x.view(batch_size, 16, self.h_after_pool, self.w_after_pool)
        x = self.upsample(x)
        x = self.deconv1(x) 
        x = self._postprocess_state(x)
        return x
    
    def _postprocess_state(self, x):
        probs = F.softmax(x, dim=1)  
        state = torch.argmax(probs, dim=1) # interesting per-pixel token prediction
        return state.float()   


from model import CastedLinear

class ActionEncoder(nn.Module): # arguably exactly the same as WTE
    def __init__(self, action_size, feature_dim):
        super(ActionEncoder, self).__init__()
        self.action_size = action_size
        self.wte = nn.Embedding(action_size, feature_dim)
    
    def forward(self, x):
        if x is None: 
            return None
        return self.wte(x)


class ActionDecoder(nn.Module):
    def __init__(self, action_size, feature_dim):
        super(ActionDecoder, self).__init__()
        self.lm_head = CastedLinear(feature_dim, action_size)

    def generate(self, x): 
        logits = self.lm_head(x)
        probs = F.softmax(logits, dim=1)
        action = torch.argmax(probs, dim=1)
        return action

    def forward(self, x, target): 
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits, target.long())
        return loss 


class RandomAgent: 
    def __init__(self, env):
        self.env = env
    
    def act(self, obs):
        return random.randint(0, 3)

def collect_trajectories(env, agent,num_episodes=100, device="cuda"):
    trajectories = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        states,actions,rewards = [],[],[]
        states.append(torch.tensor(obs, dtype=torch.float32)) # initial state
        while True:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            states.append(torch.tensor(obs, dtype=torch.float32))
            actions.append(torch.tensor(action, dtype=torch.long))
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            
            total_reward += reward
            
            if done:
                print(f"Episode {episode}: Score={info['score']}, Reward={total_reward}")
                break
        
        trajectory = (torch.stack(states).to(device), torch.stack(actions).to(device), torch.stack(rewards).to(device))
        trajectories.append(trajectory)

    env.close()
    return trajectories