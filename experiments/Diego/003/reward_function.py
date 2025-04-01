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
        reward += (20 - current_distance_to_opponent) * 0.3

    # --- Encourage retreating / penalize not retreating when taking damage
    current_health = player_current_info.get("health")
    previous_health = player_previous_info.get("health")

    if current_health < previous_health:
        health_diff = previous_health - current_health
        # cap diff to prevent HUGE spikes (due to two unsuccessful runs)
        health_diff = min(health_diff, 20)

        if current_distance_to_opponent <= previous_distance_to_opponent:
            reward -= (previous_distance_to_opponent - current_distance_to_opponent) * 0.2
        else:  # Reward Bot for retreating
            reward += (current_distance_to_opponent - previous_distance_to_opponent) * 0.1

    # -----------------------------
    # Scanning opponents with ray (Variable 2: Accuracy/Precision)
    # Goal: Encourage agent to be more precise and have better accuracy since it always shoots from the middle ray

    accuracy_metrics = self.accuracy_precision(player_current_info)
    # Fixed variable name to match accuracy_precision function
    is_targeting_player = accuracy_metrics["is_targeting_player"]
    shot_accuracy = accuracy_metrics["shot_accuracy"]

    shot_fired = player_current_info.get("shot_fired")

    reward += shot_accuracy * 0.2

    if is_targeting_player and shot_fired:
        current_damage = player_current_info.get("damage_dealt", 0)
        previous_damage = player_previous_info.get("damage_dealt", 0)

        if current_damage > previous_damage:
            damage_diff = current_damage - previous_damage
            # same as above capping limit
            damage_diff = min(damage_diff, 20)
            reward += damage_diff * 0.2  # reward for hitting

    # Exploration
    exploration_metrics = self.bot_exploration(player_current_info)
    exploration_score = exploration_metrics["exploration_score"]
    reward += exploration_score * 0.15

    # Movement efficiency
    efficiency_metrics = self.calculate_movement_efficiency(player_current_info)
    movement_efficiency = efficiency_metrics["normalized_efficiency"]
    reward += movement_efficiency * 0.12

    # Track and get learning rate metrics
    learning_metrics = self.track_learning_progress(bot_username, reward)
    learning_rate_score = learning_metrics["learning_rate_score"]
    reward += learning_rate_score * 0.1

    return reward