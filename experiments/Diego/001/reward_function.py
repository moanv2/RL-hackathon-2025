def calculate_reward(self, info_dictionary, bot_username, previous_info):
    """
    Calculate reward for a given step of the environment and for a given bot

    Reward vars
    R = w₁S + w₂D + w₃K + w₄H + w₅E + w₆A
    """
    reward = 0

    players_info = info_dictionary.get("players_info", {})
    player_current_info = players_info.get(bot_username)
    player_previous_info = previous_info.get(bot_username)

    assert player_current_info is not None
    assert player_previous_info is not None

    # --- Rewarding closing distance
    current_distance_to_opponent = self._get_distance_to_opponent(player_current_info)
    previous_distance_to_opponent = self._get_distance_to_opponent(player_previous_info)

    if current_distance_to_opponent < 20:
        reward += (20 - current_distance_to_opponent) * 0.5

    # --- Encourage retreating / penalize not retreating when taking damage
    current_health = player_current_info.get("health")
    previous_health = player_previous_info.get("health")

    if current_health < previous_health:
        if current_distance_to_opponent <= previous_distance_to_opponent:
            reward -= (previous_distance_to_opponent - current_distance_to_opponent) * 0.5
        else:  # Reward Bot for retreating
            reward += (current_distance_to_opponent - previous_distance_to_opponent) * 0.5

    # bro did not answer the whatsapp so i will pseudocode the 2nd var for the calc_reward func

    # NOTE: This is not to hard code strategy, rather just make it and let the RL algo choose
    # at least from what i understand, tomo we can clarify via call.

    # -----------------------------
    # Scanning opponents with ray (Variable 2: Accuracy/Precision)
    # Goal: Encourage agent to be more precise and have better accuracy since it always shoots from the middle ray

    all_rays = player_current_info.get("rays")
    middle_ray = all_rays[2]

    ray_detection_type = middle_ray[-1]
    if ray_detection_type == "player":
        reward += 5
    else:
        reward -= 0.001

    # Reward accuracy with middle ray
    # Excellent - middle ray detected an opponent (reward)

    # If bot shot AND hit with middle ray, major reward for accuracy
    # Additional reward if damage was dealt this epoch
    # Shot was successful in dealing damage

    # Penalize slightly for not shooting when middle ray detects enemy
    # Missed opportunity for precise shot

    # Check if any rays detected opponents (even if not middle ray)
    # Some enemies detected, but not by middle ray (small reward cuz it detected)
    # If bot shot but middle ray didn't hit, penalize slightly

    # Penalize for shooting when no enemies are detected
    # penalize (idk 0.2 we can test it out)

    return reward