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

    return reward