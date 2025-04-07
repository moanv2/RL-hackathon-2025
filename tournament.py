import os
import torch
import itertools

from bots.example_bot import MyBot
from components.character import Character
from Environment import Env
from main import run_game


def find_all_bot_paths(root_dir):
    bot_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pth"):
                bot_paths.append(os.path.join(root, file))
    return bot_paths


def create_bot_from_path(path, device):
    bot = MyBot()
    bot.model.load_state_dict(torch.load(path, map_location=device))
    bot.target_model.load_state_dict(bot.model.state_dict())
    bot.epsilon = 0.0
    return bot


def run_match(env, bot1_path, bot2_path, device):
    bot1 = create_bot_from_path(bot1_path, device)
    bot2 = create_bot_from_path(bot2_path, device)
    bots = [bot1, bot2]

    world_bounds = env.get_world_bounds()
    players = [
        Character(starting_pos=(world_bounds[2] - 100, world_bounds[3] - 100),
                  screen=env.world_surface, boundaries=world_bounds, username=bot1_path),
        Character(starting_pos=(world_bounds[0] + 10, world_bounds[1] + 10),
                  screen=env.world_surface, boundaries=world_bounds, username=bot2_path),
    ]

    # --- link players and bots to environment ---
    env.set_players_bots_objects(players, bots)

    run_game(env, players, bots)


def main():
    experiments_root = "experiments"
    all_bot_paths = find_all_bot_paths(experiments_root)

    print(f"Found {len(all_bot_paths)} bots:")
    for path in all_bot_paths:
        print(f" - {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Env(
        training=False,
        use_game_ui=True,
        world_width=1280,
        world_height=1280,
        display_width=800,
        display_height=800,
        n_of_obstacles=5,
        frame_skip=2
    )

    # All possible unique 1v1 matchups (no self-play)
    for bot1_path, bot2_path in itertools.combinations(all_bot_paths, 2):
        print(f"\n▶ Match: {bot1_path} vs {bot2_path}")
        run_match(env, bot1_path, bot2_path, device)


if __name__ == "__main__":
    main()
