import logging
import math
import os

import pygame

from components.advanced_UI import game_UI
from components.world_gen import spawn_objects


class Env:
    def __init__(self, training=False, use_game_ui=True, world_width=1280, world_height=1280, display_width=640,
                 display_height=640, n_of_obstacles=10, frame_skip=4):
        pygame.init()

        self.training_mode = training

        # ONLY FOR DISPLAY
        # create display window with desired display dimensions
        self.display_width = display_width
        self.display_height = display_height
        # only create a window if not in training mode
        if not self.training_mode:
            self.screen = pygame.display.set_mode((display_width, display_height))
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Disable actual video output
            pygame.display.set_mode((1, 1))  # Minimal display

            self.screen = pygame.Surface((display_width, display_height))

        # REAL WORLD DIMENSIONS
        # create an off-screen surface for the game world
        self.world_width = world_width
        self.world_height = world_height
        self.world_surface = pygame.Surface((world_width, world_height))

        self.clock = pygame.time.Clock()
        self.running = True

        self.use_advanced_UI = use_game_ui
        if self.use_advanced_UI:
            self.advanced_UI = game_UI(self.world_surface, self.world_width, self.world_height)

        if not self.training_mode and self.use_advanced_UI:
            self.advanced_UI.display_opening_screen()

        self.n_of_obstacles = n_of_obstacles
        self.min_obstacle_size = (50, 50)
        self.max_obstacle_size = (100, 100)

        # frame skip for training acceleration
        self.frame_skip = frame_skip if training else 1

        # INIT SOME VARIABLES
        self.OG_bots = None
        self.OG_players = None
        self.OG_obstacles = None

        self.bots = None
        self.players = None
        self.obstacles = None

        """REWARD VARIABLES"""
        self.last_positions = {}
        self.last_damage = {}
        self.last_kills = {}
        self.last_health = {}
        self.visited_areas = {}
        self.learning_metrics = {}          # learning rate tracking

        self.visited_areas.clear()
        self.last_positions.clear()
        self.last_health.clear()
        self.last_kills.clear()
        self.last_damage.clear()

        self.steps = 0

    def set_players_bots_objects(self, players, bots, obstacles=None):
        self.OG_players = players
        self.OG_bots = bots
        self.OG_obstacles = obstacles

        self.reset()

    def get_world_bounds(self):
        return (0, 0, self.world_width, self.world_height)

    def find_closest_opponent(self, player):
        """Find the position of the closest opponent for a given player"""
        closest_dist = float('inf')
        closest_pos = None

        for other in self.players:
            if other != player and other.alive:
                dist = math.dist(player.rect.center, other.rect.center)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = other.rect.center

        # return default position if no opponents found
        if closest_pos is None:
            return player.rect.center

        return closest_pos

    def reset(self, randomize_objects=False, randomize_players=False):
        self.running = True
        if not self.training_mode:
            if not self.use_advanced_UI:
                self.screen.fill("green")
                pygame.display.flip()
                self.clock.tick(1)  # 1 frame per second for 1 second = 1 frame
            else:
                self.advanced_UI.display_reset_screen()

        else:
            self.screen.fill("green")

        self.last_positions = {}
        self.last_damage = {}
        self.last_kills = {}
        self.last_health = {}
        self.visited_areas = {}
        self.learning_metrics = {}  # to reset learning metrics

        self.steps = 0

        # TODO: add variables for parameters
        if self.use_advanced_UI:
            self.obstacles = self.advanced_UI.obstacles
        else:
            if randomize_objects or self.OG_obstacles is None:
                self.OG_obstacles = spawn_objects(
                    (0, 0, self.world_width, self.world_height),
                    self.max_obstacle_size,
                    self.min_obstacle_size,
                    self.n_of_obstacles
                )
            self.obstacles = self.OG_obstacles

        self.players = self.OG_players.copy()
        self.bots = self.OG_bots
        if randomize_players:
            self.bots = self.bots.shuffle()
            for index in range(len(self.players)):
                self.players[index].related_bot = self.bots[index]  # ensuring bots change location

        else:
            for index in range(len(self.players)):
                self.players[index].related_bot = self.bots[index]

        for player in self.players:
            player.reset()
            temp = self.players.copy()
            temp.remove(player)
            player.players = temp  # Other players
            player.objects = self.obstacles

    def step(self, debugging=False):
        # only render if not in training mode
        if not self.training_mode:
            if self.use_advanced_UI:
                # use the background from game_UI
                self.world_surface.blit(self.advanced_UI.background, (0, 0))
            else:
                self.world_surface.fill("purple")

        # frame skipping for training acceleration
        skip_count = self.frame_skip if self.training_mode else 1

        # track if any frame resulted in game over
        game_over = False
        final_info = None

        # get actions once and reuse them for all skipped frames
        player_actions = {}
        if self.training_mode:
            for player in self.players:
                if player.alive:
                    # update player info with closest opponent data before action
                    player_info = player.get_info()
                    player_info['closest_opponent'] = self.find_closest_opponent(player)
                    player_actions[player.username] = player.related_bot.act(player_info)

        # process multiple frames if frame skipping is enabled
        for _ in range(skip_count):
            if game_over:
                break

            self.steps += 1

            players_info = {}
            alive_players = []

            for player in self.players:
                player.update_tick()

                # use stored actions if in training mode with frame skipping
                if self.training_mode and skip_count > 1:
                    actions = player_actions.get(player.username, {})
                else:
                    # update info with closest opponent before getting action
                    player_info = player.get_info()
                    player_info['closest_opponent'] = self.find_closest_opponent(player)
                    actions = player.related_bot.act(player_info)

                if player.alive:
                    alive_players.append(player)
                    player.reload()

                    # skip drawing in training mode for better performance
                    if not self.training_mode:
                        player.draw(self.world_surface)

                    if debugging:
                        print("Bot would like to do:", actions)
                    if actions.get("forward", False):
                        player.move_in_direction("forward")
                    if actions.get("right", False):
                        player.move_in_direction("right")
                    if actions.get("down", False):
                        player.move_in_direction("down")
                    if actions.get("left", False):
                        player.move_in_direction("left")
                    if actions.get("rotate", 0):
                        player.add_rotate(actions["rotate"])
                    if actions.get("shoot", False):
                        player.shoot()

                    if not self.training_mode:
                        # store position for trail
                        if not hasattr(player, 'previous_positions'):
                            player.previous_positions = []
                        player.previous_positions.append(player.rect.center)
                        if len(player.previous_positions) > 10:
                            player.previous_positions.pop(0)

                player_info = player.get_info()
                player_info["shot_fired"] = actions.get("shoot", False)
                player_info["closest_opponent"] = self.find_closest_opponent(player)
                players_info[player.username] = player_info

            new_dic = {
                "general_info": {
                    "total_players": len(self.players),
                    "alive_players": len(alive_players)
                },
                "players_info": players_info
            }

            # store the final state
            final_info = new_dic

            # check if game is over
            if len(alive_players) == 1:
                print("Game Over, winner is:", alive_players[0].username)
                if not self.training_mode:
                    if self.use_advanced_UI:
                        self.advanced_UI.display_winner_screen(alive_players)
                    else:
                        self.screen.fill("green")

                game_over = True
                break

        # skip all rendering operations in training mode for better performance
        if not self.training_mode:
            if self.use_advanced_UI:
                self.advanced_UI.draw_everything(final_info, self.players, self.obstacles)
            else:
                # draw obstacles manually if not using advanced UI
                for obstacle in self.obstacles:
                    obstacle.draw(self.world_surface)

            # scale and display the world surface
            scaled_surface = pygame.transform.scale(self.world_surface, (self.display_width, self.display_height))
            self.screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()

        # in training mode, use a high tick rate but not unreasonably high
        if not self.training_mode:
            self.clock.tick(120)  # normal gameplay speed
        else:
            # skip the clock tick entirely in training mode for maximum speed
            pass  # no tick limiting in training mode for maximum speed

        # return the final state
        if game_over:
            print("Total steps:", self.steps)
            return True, final_info  # Game is over
        else:
            # return the final state from the last frame
            return False, final_info

    def bot_exploration(self, info):
        """
        Track and reward exploration within the env
        :param info: Access to player information specifically location and previously visited positions
        :return: Dictionary with exploration metrics
        """
        location = info.get("location")
        username = info.get("username")

        if username not in self.visited_areas:
            # Calculate grid size
            grid_cell_size = int(self.world_width * 0.05)  # 5% of world width
            grid_width = int(self.world_width / grid_cell_size)
            grid_height = int(self.world_height / grid_cell_size)

            self.visited_areas[username] = {
                "grid": [[0 for _ in range(grid_height)] for _ in range(grid_width)],
                "total_cells": grid_width * grid_height,
                "grid_cell_size": grid_cell_size  # Store this for future reference
            }

        # Move this section outside the if block so it runs every time
        # Get the stored grid cell size
        grid_cell_size = self.visited_areas[username]["grid_cell_size"]

        # Convert location to grid coordinates
        grid_x = min(int(location[0] / grid_cell_size), len(self.visited_areas[username]["grid"]) - 1)
        grid_y = min(int(location[1] / grid_cell_size), len(self.visited_areas[username]["grid"][0]) - 1)

        # Mark as visited
        self.visited_areas[username]["grid"][grid_x][grid_y] = 1

        # Calculate exploration score (percentage of map explored)
        visited_count = sum(sum(row) for row in self.visited_areas[username]["grid"])
        exploration_score = visited_count / self.visited_areas[username]["total_cells"]

        return {
            "exploration_score": exploration_score,
            "visited_count": visited_count,
            "total_cells": self.visited_areas[username]["total_cells"]
        }

    def track_learning_progress(self, bot_username, current_reward):
        """
        Track learning progress for each bot based on reward improvements
        :param bot_username: The bot to track
        :param current_reward: Current episode reward
        :return: Dictionary with learning metrics
        """
        # Initialize tracking for this bot if not already done
        if bot_username not in self.learning_metrics:
            self.learning_metrics[bot_username] = {
                "previous_avg_reward": 0,
                "reward_history": [],
                "learning_rate_score": 0
            }

        # Add current reward to history
        self.learning_metrics[bot_username]["reward_history"].append(current_reward)

        # Keep only last 10 rewards for moving average
        if len(self.learning_metrics[bot_username]["reward_history"]) > 10:
            self.learning_metrics[bot_username]["reward_history"].pop(0)

        # Calculate recent average reward
        recent_avg_reward = sum(self.learning_metrics[bot_username]["reward_history"]) / len(
            self.learning_metrics[bot_username]["reward_history"])
        previous_avg_reward = self.learning_metrics[bot_username]["previous_avg_reward"]

        # Calculate improvement as learning rate score
        learning_rate_score = 0
        if previous_avg_reward != 0:
            improvement = (recent_avg_reward - previous_avg_reward) / abs(previous_avg_reward)
            # Normalize improvement to 0-1 range, cap at 1.0
            learning_rate_score = min(1.0, max(0, improvement))

        # Update previous average reward
        self.learning_metrics[bot_username]["previous_avg_reward"] = recent_avg_reward
        self.learning_metrics[bot_username]["learning_rate_score"] = learning_rate_score

        return {
            "learning_rate_score": learning_rate_score,
            "avg_reward": recent_avg_reward
        }

    def _get_distance_to_opponent(self, info):
        """
        Euclidian distance between current bot's location and its opponent
        """
        dx = info["location"][0] - info["closest_opponent"][0]
        dy = info["location"][1] - info["closest_opponent"][1]
        
        return (dx ** 2 + dy ** 2) ** 0.5

    def accuracy_precision(self, info):
        """
        :param info: access into the players information
        :return: Dictionary containing accuracy metrics
        """
        all_rays = info.get("rays")
        shot_fired = info.get("shot_fired")
        middle_ray = all_rays[2]

        # Get the ray detection type from the middle ray
        ray_detection_type = middle_ray[-1]

        # Fixed variable name to match return dictionary
        is_targeting_player = ray_detection_type == "player"

        shot_accuracy = 0.0

        if is_targeting_player:
            shot_accuracy = 0.10 if shot_fired else 0.05
        else:
            shot_accuracy = 0.0 if not shot_fired else -0.05

        return {
            "is_targeting_player": is_targeting_player,
            "shot_accuracy": shot_accuracy
        }

    def calculate_movement_efficiency(self, info):
        """
        Calculate how efficiently the bot is moving through the environment
        :param info: Bot information dictionary
        :return: Dictionary with movement efficiency metrics
        """
        meters_moved = info.get("meters_moved", 0)
        total_rotation = info.get("total_rotation", 0)

        # Calculate movement efficiency (distance covered per unit of rotation)
        rotation_in_full_turns = max(1, total_rotation / 360)  # Avoid division by zero
        movement_efficiency = meters_moved / rotation_in_full_turns

        # Normalize to a 0-1 scale for the reward function
        max_expected_efficiency = 10  # Adjust based on your environment
        normalized_efficiency = min(1.0, movement_efficiency / max_expected_efficiency)

        return {
            "movement_efficiency": movement_efficiency,
            "normalized_efficiency": normalized_efficiency
        }

    def calculate_reward(self, info_dictionary, bot_username, previous_info):
        """
        Calculate reward for a given step of the environment and for a given bot

        """
        reward = 0

        health_penalty_weight = 0.2
        damage_reward_weight = 0.2
        kill_reward_weight = 50
        movement_reward_weight = 0.05
        rotation_penalty_weight = 0.005
        ammo_penalty_weight = 0.1
        approach_reward_weight = 0.02
        death_penalty = 75


        players_info = info_dictionary.get("players_info", {})
        player_current_info = players_info.get(bot_username)
        player_previous_info = previous_info.get(bot_username)

        assert player_current_info is not None
        assert player_previous_info is not None

        # Penalize health loss
        health_diff = player_current_info["health"] - player_previous_info["health"]
        if health_diff < 0:
            reward += health_diff * health_penalty_weight
        
        # Reward hitting enemy
        damage_diff = player_current_info["damage_dealt"] - player_previous_info["damage_dealt"]
        reward += damage_diff * damage_reward_weight

        # Reward kills
        kill_diff = player_current_info["kills"] - player_previous_info["kills"]
        reward += kill_diff * kill_reward_weight

        # Reward exploration (movement)
        distance_moved = player_current_info["meters_moved"] - player_previous_info["meters_moved"]
        reward += distance_moved * movement_reward_weight

        # Penalize excessive spinning
        rotation_diff = player_current_info["total_rotation"] - player_previous_info["total_rotation"]
        reward -= abs(rotation_diff) * rotation_penalty_weight

        # Penalize ammo wasting
        ammo_diff = player_current_info["current_ammo"] - player_previous_info["current_ammo"]
        if ammo_diff < 0:
            reward += ammo_diff * ammo_penalty_weight

        # Reward getting closer to enemy
        prev_dist = self._get_distance_to_opponent(player_previous_info)
        curr_dist = self._get_distance_to_opponent(player_current_info)
        dist_diff = prev_dist - curr_dist
        reward += dist_diff * approach_reward_weight

        # Penalize death
        if not player_current_info["alive"] and player_previous_info["alive"]:
            reward -= death_penalty
        
        # Reward when it is looking at opponent
        ray_reward = 0
        ray_reward_weight = 2.0   
        center_ray_bonus_weight = 5.0

        rays = player_current_info.get("rays", [])
        for i, ray in enumerate(rays):
            hit_type = ray[-1]
            if hit_type == 'player':
                if i == 2:  # middle ray (the one that shoots)
                    ray_reward += center_ray_bonus_weight
                else:
                    ray_reward += ray_reward_weight

        reward += ray_reward

        return reward