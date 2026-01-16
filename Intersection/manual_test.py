# --- manual_test.py ---
# Manual control test using IntersectionEnv

import pygame
import numpy as np
import random
import sys
import os

# Handle imports for both script execution and module import
try:
    # First try: import from installed package
    from Intersection import IntersectionEnv, DEFAULT_REWARD_CONFIG
    from Intersection.config import FPS, SCALE
except ImportError:
    try:
        # Second try: relative import (when used as a module)
        from .env import IntersectionEnv
        from .config import DEFAULT_REWARD_CONFIG, FPS, SCALE
    except ImportError:
        # Fall back: absolute import (when run as a script from source)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Intersection.env import IntersectionEnv
        from Intersection.config import DEFAULT_REWARD_CONFIG, FPS, SCALE


def main():
    """Manual control test with IntersectionEnv."""
    
    # Initialize environment (routes will be set dynamically)
    config = {
        'traffic_flow': False,  # No traffic flow for manual test
        'num_agents': 1,        # Single agentww
        'num_lanes': 2,
        'render_mode': 'human',
        'max_steps': 2000,
        'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
    }
    
    env = IntersectionEnv(config)

    def choose_random_route():
        if hasattr(env, 'traffic_routes') and len(env.traffic_routes) > 0:
            return random.choice(env.traffic_routes)
        # Fallback to default route if mapping is empty
        return ('IN_5', 'OUT_7')

    # Set initial random route and reset
    env.ego_routes = [choose_random_route()]
    obs, info = env.reset()
    
    print("=" * 60)
    print("Manual Control Test")
    print("=" * 60)
    print("Controls:")
    print("  UP/DOWN arrows: Throttle")
    print("  LEFT/RIGHT arrows: Steering")
    print("  R: Reset environment")
    print("  ESC/Q: Quit")
    print("=" * 60)
    
    total_reward = 0.0
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset environment with a new random route
                    env.ego_routes = [choose_random_route()]
                    obs, info = env.reset()
                    total_reward = 0.0
                    print(f"Environment reset! New route: {env.ego_routes[0]}")
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
        
        # Keyboard control
        keys = pygame.key.get_pressed()
        throttle = 0.0
        steer = 0.0
        
        if keys[pygame.K_UP]:
            throttle = 0.3
        if keys[pygame.K_DOWN]:
            throttle = -0.5
        if keys[pygame.K_LEFT]:
            steer = 1.0
        if keys[pygame.K_RIGHT]:
            steer = -1.0
        
        action = np.array([throttle, steer])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        # Handle both single agent (scalar) and multi-agent (list) reward formats
        if isinstance(reward, (list, np.ndarray)) and len(reward) > 0:
            reward = reward[0]  # Take first agent's reward
        total_reward += reward
        done = terminated or truncated
        
        if done:
            print(f"Episode ended: {info['collisions']}, Total Reward: {total_reward:.4f}")
            # Auto-reset for continuous play
            obs, info = env.reset()
            total_reward = 0.0
        
        # Custom rendering with lane IDs and lidar
        if env.render_mode == 'human' and env.pygame_initialized:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Clear screen
            env.screen.fill((0, 0, 0))
            
            # Draw road with lane IDs
            env.road.draw(env.screen, show_lane_ids=True)
            
            # Draw agents
            for agent in env.agents:
                if agent.alive():
                    # Debug: draw a red circle at agent position
                    pygame.draw.circle(env.screen, (255, 0, 0), 
                                     (int(agent.pos_x), int(agent.pos_y)), 10)
                    # Draw the car
                    env.screen.blit(agent.image, agent.rect)
                    # Draw lidar
                    agent.lidar.draw(env.screen)
            
            # === Visualization of navigation path ===
            if len(env.agents) > 0 and env.agents[0].alive():
                player_car = env.agents[0]
                
                # 1. Draw target lookahead point (yellow dot)
                if hasattr(player_car, 'path') and len(player_car.path) > 0:
                    lookahead = 10
                    idx = min(player_car.path_index + lookahead, len(player_car.path)-1)
                    target_pt = player_car.path[idx]
                    pygame.draw.circle(env.screen, (255, 255, 0), 
                                     (int(target_pt[0]), int(target_pt[1])), 5)
                    
                    # 2. Draw complete trajectory line (cyan line)
                    if len(player_car.path) > 1:
                        pygame.draw.lines(env.screen, (0, 255, 255), False, 
                                         player_car.path, 1)
            
            # Status text
            status_text = f"Step: {env.step_count} | Reward: {total_reward:.2f}"
            if len(env.agents) > 0 and env.agents[0].alive():
                speed_ms = (env.agents[0].speed * FPS) / SCALE
                status_text += f" | Speed: {speed_ms:.1f} m/s"
            
            # Check collision status
            collision_info = info.get('collisions', {})
            if collision_info:
                agent_id = id(env.agents[0])
                if agent_id in collision_info:
                    status = collision_info[agent_id]
                    color = (255, 0, 0) if status != "ALIVE" else (0, 255, 0)
                    status_text += f" | Status: {status}"
                else:
                    color = (0, 255, 0)
            else:
                color = (0, 255, 0)
            
            txt = env.font.render(status_text, True, color)
            env.screen.blit(txt, (10, 10))
            
            # Hint text
            hint_text = "Path is Cyan. Target is Yellow."
            hint_surf = env.font.render(hint_text, True, (255, 255, 255))
            env.screen.blit(hint_surf, (10, 40))
            
            if done:
                reset_text = "Press R to Reset"
                reset_surf = env.font.render(reset_text, True, (255, 255, 0))
                env.screen.blit(reset_surf, (10, 70))
            
            pygame.display.flip()
            env.clock.tick(FPS)
    
    env.close()
    print("Test completed!")

if __name__ == '__main__':
    main()
