import logging

def calculate_reward(self, info_dictionary, bot_username, previous_info):
    """
    Calculate reward for a given step of the environment and for a given bot
    """
    reward = 0

    # Reward weights
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
        delta = health_diff * health_penalty_weight
        reward += delta
        logging.debug(f"[{bot_username}] Health penalty: {delta:.2f}")

    # Reward hitting enemy
    damage_diff = player_current_info["damage_dealt"] - player_previous_info["damage_dealt"]
    delta = damage_diff * damage_reward_weight
    reward += delta
    logging.debug(f"[{bot_username}] Damage reward: {delta:.2f}")

    # Reward kills
    kill_diff = player_current_info["kills"] - player_previous_info["kills"]
    delta = kill_diff * kill_reward_weight
    reward += delta
    logging.debug(f"[{bot_username}] Kill reward: {delta:.2f}")

    # Reward exploration (movement)
    distance_moved = player_current_info["meters_moved"] - player_previous_info["meters_moved"]
    delta = distance_moved * movement_reward_weight
    reward += delta
    logging.debug(f"[{bot_username}] Movement reward: {delta:.2f}")

    # Penalize excessive spinning
    rotation_diff = player_current_info["total_rotation"] - player_previous_info["total_rotation"]
    delta = -abs(rotation_diff) * rotation_penalty_weight
    reward += delta
    logging.debug(f"[{bot_username}] Rotation penalty: {delta:.2f}")

    # Penalize ammo wasting
    ammo_diff = player_current_info["current_ammo"] - player_previous_info["current_ammo"]
    if ammo_diff < 0:
        delta = ammo_diff * ammo_penalty_weight
        reward += delta
        logging.debug(f"[{bot_username}] Ammo penalty: {delta:.2f}")

    # Reward getting closer to enemy
    prev_dist = self._get_distance_to_opponent(player_previous_info)
    curr_dist = self._get_distance_to_opponent(player_current_info)
    dist_diff = prev_dist - curr_dist
    delta = dist_diff * approach_reward_weight
    reward += delta
    logging.debug(f"[{bot_username}] Approach reward: {delta:.2f}")

    # Penalize death
    if not player_current_info["alive"] and player_previous_info["alive"]:
        reward -= death_penalty
        logging.debug(f"[{bot_username}] Death penalty: {-death_penalty:.2f}")

    # Reward when looking at opponent
    ray_reward = 0
    ray_reward_weight = 2.0   
    center_ray_bonus_weight = 5.0

    rays = player_current_info.get("rays", [])
    for i, ray in enumerate(rays):
        hit_type = ray[-1]
        if hit_type == 'player':
            if i == 2:
                ray_reward += center_ray_bonus_weight
                logging.debug(f"[{bot_username}] Center ray sees opponent: +{center_ray_bonus_weight}")
            else:
                ray_reward += ray_reward_weight
                logging.debug(f"[{bot_username}] Side ray {i} sees opponent: +{ray_reward_weight}")

    reward += ray_reward

    return reward