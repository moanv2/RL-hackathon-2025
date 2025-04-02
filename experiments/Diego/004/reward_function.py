def calculate_reward(self, info_dictionary, bot_username, previous_info):
    """
    Enhanced reward function optimized for Rainbow DQN learning

    Reward structure:
    R = w₁*Combat + w₂*Positioning + w₃*Exploration + w₄*Survival + w₅*Learning
    """
    reward = 0

    players_info = info_dictionary.get("players_info", {})
    player_current_info = players_info.get(bot_username)
    player_previous_info = previous_info.get(bot_username)

    if player_current_info is None or player_previous_info is None:
        return 0  # Safety check

    # --- COMBAT REWARDS ---
    # Get metrics for accuracy and precision
    accuracy_metrics = self.accuracy_precision(player_current_info)
    is_targeting_player = accuracy_metrics["is_targeting_player"]
    shot_fired = player_current_info.get("shot_fired")

    # 1. Reward for targeting enemy with middle ray
    if is_targeting_player:
        reward += 0.2  # Base reward for aiming correctly

        # 2. Bigger reward for hitting enemy (damage dealt)
        current_damage = player_current_info.get("damage_dealt", 0)
        previous_damage = player_previous_info.get("damage_dealt", 0)

        if current_damage > previous_damage:
            damage_diff = min(current_damage - previous_damage, 30)  # Cap to prevent spikes
            reward += damage_diff * 0.5  # Substantial reward for successful hits

            # Extra bonus for kills
            current_kills = player_current_info.get("kills", 0)
            previous_kills = player_previous_info.get("kills", 0)
            if current_kills > previous_kills:
                reward += 10.0  # Large reward for a kill

    # 3. Penalize for wasting ammo (shooting when not targeting enemy)
    elif shot_fired:
        reward -= 0.15  # Small penalty for wasting shots

    # --- POSITIONING REWARDS ---
    # Calculate distances to opponent
    current_distance = self._get_distance_to_opponent(player_current_info)
    previous_distance = self._get_distance_to_opponent(player_previous_info)

    # 4. Optimal distance management (based on health state)
    current_health = player_current_info.get("health", 100)
    previous_health = player_previous_info.get("health", 100)
    optimal_distance = 250  # Default optimal distance

    # When health is low, prefer greater distance for safety
    if current_health < 40:
        optimal_distance = 400
    # When health is high, prefer closer distance for aggression
    elif current_health > 70:
        optimal_distance = 150

    # Reward for moving toward optimal distance
    distance_to_optimal_current = abs(current_distance - optimal_distance)
    distance_to_optimal_previous = abs(previous_distance - optimal_distance)

    if distance_to_optimal_current < distance_to_optimal_previous:
        # Moving toward optimal distance
        reward += 0.3 * (distance_to_optimal_previous - distance_to_optimal_current) / optimal_distance

    # 5. Dynamic positioning based on health changes
    if current_health < previous_health:
        health_diff = previous_health - current_health

        # Reward retreating when taking damage
        if current_distance > previous_distance:
            retreat_reward = min(0.2 * health_diff, 1.0)
            reward += retreat_reward
        # Penalize staying too close when taking damage
        else:
            reward -= 0.1 * health_diff

    # --- EXPLORATION REWARDS ---
    # 6. Encourage map exploration
    exploration_metrics = self.bot_exploration(player_current_info)
    exploration_score = exploration_metrics["exploration_score"]

    # Calculate incremental exploration reward
    previous_exp = self.visited_areas.get(bot_username, {}).get("previous_score", 0)
    exploration_delta = max(0, exploration_score - previous_exp)
    # Store current score for next comparison
    if bot_username in self.visited_areas:
        self.visited_areas[bot_username]["previous_score"] = exploration_score

    # Higher reward for discovering new areas
    reward += exploration_delta * 5.0

    # 7. Movement efficiency - reward efficient movement patterns
    efficiency_metrics = self.calculate_movement_efficiency(player_current_info)
    normalized_efficiency = efficiency_metrics["normalized_efficiency"]
    reward += normalized_efficiency * 0.2

    # --- SURVIVAL REWARDS ---
    # 8. Time-based survival reward (tiny reward for staying alive)
    reward += 0.01

    # 9. Health preservation reward
    if current_health >= previous_health:
        reward += 0.05  # Small reward for maintaining health

    # --- LEARNING REWARDS ---
    # 10. Track learning progress and add bonus for improvement
    learning_metrics = self.track_learning_progress(bot_username, reward)
    learning_rate_score = learning_metrics["learning_rate_score"]
    reward += learning_rate_score * 0.5

    # Apply discount factor to final reward to help with temporal credit assignment
    return reward * 0.95