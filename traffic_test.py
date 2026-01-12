# --- traffic_flow_test.py ---
# Test script for traffic flow with lane ID rendering

import pygame
import numpy as np
from env import IntersectionEnv
from config import DEFAULT_REWARD_CONFIG, FPS, SCALE

def main():
    """Test traffic flow with manual control."""
    
    # Initialize environment with traffic flow
    config = {
        'traffic_flow': True,  # Single agent with traffic flow
        'traffic_density': 5,  # Moderate traffic density
        'render_mode': 'human',
        'max_steps': 2000,
        'reward_config': DEFAULT_REWARD_CONFIG['reward_config'],
        'ego_routes': [('IN_6', 'OUT_2')]  # South to North straight
    }
    
    env = IntersectionEnv(config)
    
    # Reset environment
    obs, info = env.reset()
    
    print("=" * 60)
    print("Traffic Flow Test")
    print("=" * 60)
    print("Controls:")
    print("  UP/DOWN arrows: Throttle")
    print("  LEFT/RIGHT arrows: Steering")
    print("  R: Reset environment")
    print("  ESC/Q: Quit")
    print("=" * 60)
    print(f"Traffic density: {config['traffic_density']}")
    print(f"Traffic flow enabled: {env.traffic_flow}")
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
                    # Reset environment
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("Environment reset!")
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
        
        # Keyboard control
        keys = pygame.key.get_pressed()
        throttle = 0.0
        steer = 0.0
        
        if keys[pygame.K_UP]:
            throttle = 0.5
        if keys[pygame.K_DOWN]:
            throttle = -0.5
        if keys[pygame.K_LEFT]:
            steer = 1.0
        if keys[pygame.K_RIGHT]:
            steer = -1.0
        
        action = np.array([throttle, steer])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        if done:
            print(f"Episode ended: {info['collisions']}, Total Reward: {total_reward:.4f}")
            # Auto-reset for continuous play
            obs, info = env.reset()
            total_reward = 0.0
        
        # Custom rendering with lane IDs (no lidar)
        if env.render_mode == 'human' and env.pygame_initialized:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Clear screen
            env.screen.fill((0, 0, 0))
            
            # Draw road with lane IDs
            env.road.draw(env.screen, show_lane_ids=True)
            
            # Draw traffic NPCs
            if env.traffic_flow:
                env.traffic_cars.draw(env.screen)
            
            # Draw agents (ego vehicle) with lidar
            for agent in env.agents:
                if agent.alive():
                    env.screen.blit(agent.image, agent.rect)
                    # Draw lidar only for ego vehicle (not for NPCs)
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
            
            # Traffic info
            if env.traffic_flow:
                status_text += f" | Traffic: {len(env.traffic_cars)}"
            
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
            hint_text = "Traffic Flow Test - Path: Cyan, Target: Yellow"
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

