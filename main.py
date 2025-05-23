import argparse
import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pygame
import torch

from bots.programmed_bot import ProgrammedBot
from bots.example_bot import MyBot
from components.character import Character
from example_bot_1 import RainbowDQNAgent
from Environment import Env


# --- these variables shouldn't be touched ---
WORLD_WIDTH = 1280
WORLD_HEIGHT = 1280
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 800

# --- config for the training of the model ---
CONFIG = {
    "frame_skip": 4,
    "tick_limit": 2400,
    "num_epochs": 100,
    "action_size": 56,
    "hyperparameters": {
        "double_dqn": True,
        "learning_rate": 0.0001,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_decay": 0.9999,
    }
}

# --- changes number of obstacles ---
CURRICULUM_STAGES = [
    {"n_obstacles": 10, "duration": 100},
    {"n_obstacles": 15, "duration": 200},
    {"n_obstacles": 20, "duration": 300}
]


def run_game(env, players, bots):
    """Runs the game in display mode for human viewing"""
    env.reset(randomize_objects=True)
    env.steps = 0

    for bot in bots:
        bot.reset_for_new_episode()

    env.set_players_bots_objects(players, bots)

    env.last_damage_tracker = {player.username: 0 for player in players}

    running = True
    while running:
        # handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        # make sure all events are processed
        pygame.event.pump()

        # step the environment
        finished, _ = env.step(debugging=False)
        
        if finished:
            print("Game finished!")
            # wait a moment to show the final state
            pygame.time.delay(3000)  # 3 seconds delay
            break


def train_single_episode(env, players, bots, config, current_stage):
    """Trains a single episode in one environment"""
    env.reset(randomize_objects=True)
    env.steps = 0

    for bot in bots:
        bot.reset_for_new_episode()

    episode_metrics = {
        "rewards": {player.username: 0 for player in players},
        "kills": {player.username: 0 for player in players},
        "damage_dealt": {player.username: 0 for player in players},
        "survival_time": {player.username: 0 for player in players},
        "epsilon": {player.username: 0 for player in players}
    }

    # Initialize last_damage_tracker unconditionally
    env.last_damage_tracker = {player.username: 0 for player in players}

    previous_info = {}

    for i, player in enumerate(players):
        player_info = player.get_info()
        player_info["shot_fired"] = False
        player_info["closest_opponent"] = players[1].get_info()["location"] if i == 0 else players[0].get_info()["location"]
        previous_info[player.username] = player_info

    while env.steps < config["tick_limit"]:
        finished, info = env.step(debugging=False)

        for player, bot in zip(players, bots):
            reward = env.calculate_reward(info, player.username, previous_info)
            reward *= 1.0 - (current_stage * 0.1)  # scale by curriculum stage
            episode_metrics["rewards"][player.username] += reward

            player_info = info["players_info"][player.username]
            episode_metrics["kills"][player.username] = player_info.get("kills", 0)

            current_damage = player_info.get("damage_dealt", 0)
            damage_delta = current_damage - env.last_damage_tracker.get(player.username, 0)
            episode_metrics["damage_dealt"][player.username] += max(0, damage_delta)
            env.last_damage_tracker[player.username] = current_damage

            if player_info.get("alive", False):
                episode_metrics["survival_time"][player.username] += 1

            next_info = player.get_info()
            if 'closest_opponent' not in next_info:
                next_info['closest_opponent'] = env.find_closest_opponent(player)
            bot.remember(reward, next_info, finished)

            episode_metrics["epsilon"][player.username] = bot.epsilon

        previous_info = info["players_info"]
        if finished:
            break

    return episode_metrics


# --- CLI arguments ---

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=str2bool,
        default=True,
        help='A boolean flag to activate training mode (default: True)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)'
    )
    return parser.parse_args()



def main():
    args = parse_args()
    training_mode = args.train
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    # --- setup output directory using time ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"training_runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    os.makedirs(f"{run_dir}/plots", exist_ok=True)

    # --- save config to file for debugging purposes ---
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)

    # --- create environment ---
    env = Env(
        training=training_mode,
        use_game_ui=False,
        world_width=WORLD_WIDTH,
        world_height=WORLD_HEIGHT,
        display_width=DISPLAY_WIDTH,
        display_height=DISPLAY_HEIGHT,
        n_of_obstacles=CURRICULUM_STAGES[0]["n_obstacles"],
        frame_skip=CONFIG["frame_skip"]
    )

    world_bounds = env.get_world_bounds()

    # --- setup players and bots ---
    players = [
        Character(starting_pos=(world_bounds[2] - 100, world_bounds[3] - 100),
                  screen=env.world_surface, boundaries=world_bounds, username="Ninja"),
        Character(starting_pos=(world_bounds[0] + 10, world_bounds[1] + 10),
                  screen=env.world_surface, boundaries=world_bounds, username="Von"),
    ]

    bots = []
    # Choose between training mode and display mode
    if training_mode:
        for _ in players:
            bot = RainbowDQNAgent(action_size=CONFIG["action_size"])
            bot.use_double_dqn = CONFIG["hyperparameters"]["double_dqn"]
            bot.learning_rate = CONFIG["hyperparameters"]["learning_rate"]
            bot.batch_size = CONFIG["hyperparameters"]["batch_size"]
            bot.gamma = CONFIG["hyperparameters"]["gamma"]
            bot.epsilon_decay = CONFIG["hyperparameters"]["epsilon_decay"]
            bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=bot.learning_rate)
            bots.append(bot)
        
        # --- link players and bots to environment ---
        env.set_players_bots_objects(players, bots)

        all_rewards = {player.username: [] for player in players}

        # --- training Loop ---
        for epoch in range(CONFIG["num_epochs"]):
            print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")

            # determine current curriculum stage
            total_epochs = 0
            for i, stage in enumerate(CURRICULUM_STAGES):
                total_epochs += stage["duration"]
                if epoch < total_epochs:
                    current_stage = i
                    break

            env.n_of_obstacles = CURRICULUM_STAGES[current_stage]["n_obstacles"]

            metrics = train_single_episode(env, players, bots, CONFIG, current_stage)

            for idx, bot in enumerate(bots):
                torch.save(bot.model.state_dict(), f"{run_dir}/models/bot_model_{idx}_epoch_{epoch + 1}.pth")

            for player in players:
                username = player.username
                all_rewards[username].append(metrics["rewards"][username])
                print(f"{username} - Reward: {metrics['rewards'][username]:.2f}, "
                      f"Kills: {metrics['kills'][username]}, "
                      f"Damage: {metrics['damage_dealt'][username]}, "
                      f"Epsilon: {metrics['epsilon'][username]:.4f}")

        # --- plot training rewards ---
        for username, rewards in all_rewards.items():
            plt.plot(rewards, label=username)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{run_dir}/plots/rewards_plot.png")
        plt.close()

    else:
        # Display mode - run the game for human viewing
        trained_models = [
            ("experiments/Diego/005/bot_model_0_epoch_13.pth", "new")
        ]
        for idx, model in enumerate(trained_models):
            (model_path, architecture) = model
            if architecture == "old":
                bot = MyBot(action_size=CONFIG["action_size"])
                bot.model.load_state_dict(torch.load(model_path, map_location=bot.device))
                bot.target_model.load_state_dict(bot.model.state_dict())  # Keep target model in sync
                bot.epsilon = 0.0  # Full exploitation during display
                
            else:
                bot = RainbowDQNAgent(action_size=CONFIG["action_size"])
                bot.model.load_state_dict(torch.load(model_path, map_location=bot.device))
                bot.target_model.load_state_dict(bot.model.state_dict())  # Keep target model in sync
                bot.epsilon = 0.0  # Full exploitation during display

            bots.append(bot)

        best_bot = ProgrammedBot()
        bots.append(best_bot)

        # --- link players and bots to environment ---
        env.set_players_bots_objects(players, bots)

        run_game(env, players, bots)


if __name__ == "__main__":
    main()
