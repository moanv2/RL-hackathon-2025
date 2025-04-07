import math
import random


class BestBot():
    def __init__(self):
        self.name = "BestBot"
        self.previous_info = None
        self.step_counter = 0
        self.last_position = None
        self.stuck_counter = 0
        self.exploration_mode = False
        self.exploration_timer = 0
        self.exploration_direction = 0
        # World boundary knowledge
        self.world_width = 1280
        self.world_height = 1280
        # Efficient region tracking (divide world into 8x8 grid)
        self.grid_size = 160  # 1280/8 = 160
        self.visited_regions = set()
        self.current_target_region = None
        # Performance tracking
        self.damage_dealt_history = []
        self.health_history = []
        self.kills_history = []
        self.last_damage_dealt = 0
        self.last_health = 100
        # Adaptive parameters
        self.aggression_factor = 0.5  # Start with balanced aggression
        self.exploration_preference = 0.5  # Start with balanced exploration

    def reset_for_new_episode(self):
        self.previous_info = None
        self.step_counter = 0
        self.last_position = None
        self.stuck_counter = 0
        self.exploration_mode = False
        self.exploration_timer = 0
        self.exploration_direction = 0
        self.visited_regions = set()
        self.current_target_region = None
        # Reset performance tracking but maintain adaptive parameters
        self.damage_dealt_history = []
        self.health_history = []
        self.kills_history = []
        self.last_damage_dealt = 0
        self.last_health = 100
        # Keep adaptive parameters between episodes for continued learning

    def act(self, info):
        # Default safe action in case of any failures
        default_action = {
            "forward": True,
            "right": False,
            "down": False,
            "left": False,
            "rotate": random.randint(-5, 5),
            "shoot": False
        }

        try:
            # Increment step counter
            self.step_counter += 1

            # Extract key information efficiently
            try:
                my_pos = info["location"]
                opponent_pos = info["closest_opponent"]
                my_rot = info["rotation"]
                rays = info["rays"]
                my_health = info.get("health", 100)
                current_ammo = info.get("current_ammo", 10)
                damage_dealt = info.get("damage_dealt", 0)
                kills = info.get("kills", 0)
            except KeyError as e:
                # Log the missing key for debugging
                print(f"Missing key in info: {e}")
                return default_action

            # Update performance metrics
            self.update_performance_metrics(my_health, damage_dealt, kills)

            # Update visited regions (optimization tracking)
            try:
                current_region = (int(my_pos[0] // self.grid_size), int(my_pos[1] // self.grid_size))
                self.visited_regions.add(current_region)
            except (TypeError, ZeroDivisionError) as e:
                print(f"Error in region calculation: {e}")
                # Continue execution, this is non-critical

            # Check if we're stuck (optimization: movement efficiency)
            try:
                if self.last_position is not None:
                    distance_moved = math.sqrt((my_pos[0] - self.last_position[0]) ** 2 +
                                               (my_pos[1] - self.last_position[1]) ** 2)
                    if distance_moved < 0.1:
                        self.stuck_counter += 1
                    else:
                        self.stuck_counter = 0
            except (TypeError, ValueError) as e:
                print(f"Error in stuck detection: {e}")
                self.stuck_counter = 0  # Reset to avoid false positives

            # Store current position for next comparison
            self.last_position = my_pos

            # OPTIMIZATION: Boundary avoidance - don't waste time at edges
            try:
                at_boundary = self.check_boundary(my_pos)
                if at_boundary:
                    boundary_action = self.handle_boundary(my_pos, my_rot)
                    return boundary_action
            except Exception as e:
                print(f"Error in boundary handling: {e}")
                # Continue execution if boundary check fails

            # OPTIMIZATION: Ray analysis for precision targeting
            try:
                ray_analysis = self.analyze_rays(rays)
                opponent_detected = ray_analysis["opponent_detected"]
                opponent_ray_index = ray_analysis["opponent_ray_index"]
                fine_tune_rotation = ray_analysis["fine_tune_rotation"]
            except Exception as e:
                print(f"Error in ray analysis: {e}")
                opponent_detected = False
                opponent_ray_index = -1
                fine_tune_rotation = 0

            # Calculate distance to opponent (used in multiple decisions)
            try:
                distance_to_opponent = self.calculate_distance(my_pos, opponent_pos)
            except Exception as e:
                print(f"Error in distance calculation: {e}")
                distance_to_opponent = 100  # Default to a safe distance

            # OPTIMIZATION: Dynamic exploration mode based on performance
            try:
                if not self.exploration_mode:
                    should_explore = self.should_start_exploring(opponent_detected, distance_to_opponent)
                    if should_explore:
                        self.start_exploration()
                else:
                    # Update exploration state
                    should_continue = self.should_continue_exploring(opponent_detected, distance_to_opponent)
                    if not should_continue:
                        self.exploration_mode = False
            except Exception as e:
                print(f"Error in exploration logic: {e}")
                self.exploration_mode = False  # Default to combat mode

            # OPTIMIZATION: Targeting calculation
            try:
                angle_diff = self.calculate_targeting_angle(my_pos, opponent_pos, my_rot,
                                                            fine_tune_rotation, self.exploration_mode)
            except Exception as e:
                print(f"Error in targeting calculation: {e}")
                angle_diff = 0  # Default to no rotation

            # OPTIMIZATION: Shooting decision based on optimal conditions
            try:
                optimal_shooting_distance = 40  # Optimal range
                should_shoot = self.should_shoot(opponent_detected, opponent_ray_index, rays,
                                                 distance_to_opponent, optimal_shooting_distance,
                                                 current_ammo)
            except Exception as e:
                print(f"Error in shooting decision: {e}")
                should_shoot = False  # Default to not shooting

            # MOVEMENT OPTIMIZATION: Determine the best action based on current state

            # If we're stuck, try to break free (optimization: unstuck algorithm)
            if self.stuck_counter > 5:
                action = self.get_unstuck_action(should_shoot)
            # If we're exploring, use exploration behavior (optimization: explore efficiently)
            elif self.exploration_mode:
                action = self.get_exploration_action(angle_diff, should_shoot)
            # Normal combat movement (optimization: maximize damage while minimizing damage taken)
            else:
                action = self.get_combat_action(angle_diff, my_health, distance_to_opponent,
                                                opponent_detected, should_shoot)

            # Store current info for next step
            self.previous_info = info.copy()
            return action

        except Exception as e:
            print(f"Unhandled exception in act method: {e}")
            return default_action

    # Helper methods for optimization

    def get_next_exploration_target(self):
        """
        Select the next region to explore, prioritizing unvisited regions.
        Optimized for maximizing exploration coverage.
        """
        try:
            # Create all possible regions in the grid
            all_regions = [(x, y) for x in range(8) for y in range(8)]

            # Filter out visited regions
            unvisited_regions = [region for region in all_regions if region not in self.visited_regions]

            if unvisited_regions:
                # If there are unvisited regions, use a weighted selection
                # Prioritize regions that are closer to the center of the map
                center_x, center_y = 3.5, 3.5  # Center of 8x8 grid

                # Calculate weights based on distance from center (inverted so closer = higher weight)
                weights = []
                for region in unvisited_regions:
                    dist_from_center = math.sqrt((region[0] - center_x) ** 2 + (region[1] - center_y) ** 2)
                    # Invert and scale weight (closer to center = higher weight)
                    weight = 1.0 / (1.0 + dist_from_center)
                    weights.append(weight)

                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                else:
                    normalized_weights = [1.0 / len(weights)] * len(weights)

                # Weighted random selection
                return random.choices(unvisited_regions, weights=normalized_weights, k=1)[0]
            else:
                # If all regions visited, pick a random one
                return random.choice(all_regions)
        except Exception as e:
            print(f"Error in get_next_exploration_target: {e}")
            # Return a random region as fallback
            return (random.randint(0, 7), random.randint(0, 7))

    def update_performance_metrics(self, current_health, current_damage, current_kills):
        """Update performance tracking metrics and adapt bot behavior"""
        try:
            # Track damage dealt
            damage_diff = current_damage - self.last_damage_dealt
            if damage_diff > 0:
                self.damage_dealt_history.append(damage_diff)
                # Adapt aggression based on successful damage dealing
                if len(self.damage_dealt_history) > 5:
                    recent_damage = sum(self.damage_dealt_history[-5:])
                    # Increase aggression if dealing good damage
                    if recent_damage > 50:
                        self.aggression_factor = min(0.9, self.aggression_factor + 0.05)
                    # Decrease aggression if not dealing much damage
                    elif recent_damage < 10:
                        self.aggression_factor = max(0.2, self.aggression_factor - 0.05)

            # Track health lost
            health_diff = self.last_health - current_health
            if health_diff > 0:
                self.health_history.append(health_diff)
                # Adapt exploration based on damage taken
                if len(self.health_history) > 5:
                    recent_damage_taken = sum(self.health_history[-5:])
                    # Increase exploration if taking too much damage
                    if recent_damage_taken > 40:
                        self.exploration_preference = min(0.8, self.exploration_preference + 0.1)
                    # Decrease exploration if not taking much damage
                    elif recent_damage_taken < 10 and self.aggression_factor > 0.6:
                        self.exploration_preference = max(0.2, self.exploration_preference - 0.05)

            # Track kills
            if current_kills > len(self.kills_history):
                self.kills_history.append(self.step_counter)
                # Boost aggression on kill
                self.aggression_factor = min(0.9, self.aggression_factor + 0.1)

            # Update last values
            self.last_damage_dealt = current_damage
            self.last_health = current_health
        except Exception as e:
            print(f"Error in update_performance_metrics: {e}")
            # Non-critical function, continue execution

    def check_boundary(self, position):
        """Check if position is near world boundary"""
        try:
            edge_buffer = 50
            return (position[0] < edge_buffer or
                    position[0] > self.world_width - edge_buffer or
                    position[1] < edge_buffer or
                    position[1] > self.world_height - edge_buffer)
        except TypeError as e:
            print(f"Error in check_boundary: {e}")
            return False  # Default to not at boundary

    def handle_boundary(self, position, rotation):
        """Handle movement when near boundary"""
        try:
            edge_buffer = 50
            boundary_turn_direction = 0

            # Determine which boundary we're near
            if position[0] < edge_buffer:  # Too close to left edge
                boundary_turn_direction = 90  # Turn right
            elif position[0] > self.world_width - edge_buffer:  # Too close to right edge
                boundary_turn_direction = 270  # Turn left
            elif position[1] < edge_buffer:  # Too close to top edge
                boundary_turn_direction = 180  # Turn down
            elif position[1] > self.world_height - edge_buffer:  # Too close to bottom edge
                boundary_turn_direction = 0  # Turn up

            # Calculate optimal rotation
            angle_diff = ((boundary_turn_direction - rotation + 180) % 360) - 180
            rotation_amount = min(max(angle_diff, -15), 15)  # Limit rotation speed

            return {
                "forward": True,
                "right": False,
                "down": False,
                "left": False,
                "rotate": rotation_amount,
                "shoot": False
            }
        except Exception as e:
            print(f"Error in handle_boundary: {e}")
            # Return a default safe action
            return {
                "forward": True,
                "right": False,
                "down": False,
                "left": False,
                "rotate": 10,
                "shoot": False
            }

    def analyze_rays(self, rays):
        """Analyze ray data for opponent detection"""
        try:
            opponent_detected = False
            opponent_ray_index = -1
            fine_tune_rotation = 0

            # Scan rays for opponent
            for i, ray in enumerate(rays):
                if ray[-1] == "player":
                    opponent_detected = True
                    opponent_ray_index = i
                    break

            # Calculate fine-tune rotation if opponent detected
            if opponent_detected:
                center_index = len(rays) // 2
                ray_diff = opponent_ray_index - center_index
                fine_tune_rotation = ray_diff * 2  # Scale based on how off-center

            return {
                "opponent_detected": opponent_detected,
                "opponent_ray_index": opponent_ray_index,
                "fine_tune_rotation": fine_tune_rotation
            }
        except Exception as e:
            print(f"Error in analyze_rays: {e}")
            return {
                "opponent_detected": False,
                "opponent_ray_index": -1,
                "fine_tune_rotation": 0
            }

    def should_start_exploring(self, opponent_detected, distance_to_opponent):
        """Determine if bot should enter exploration mode"""
        try:
            # Base exploration on dynamic exploration preference
            base_probability = self.exploration_preference * 0.04

            # Adjust based on opponent proximity
            if opponent_detected or distance_to_opponent < 50:
                exploration_probability = base_probability * 0.25  # Reduce when opponent near
            else:
                exploration_probability = base_probability

            # Random chance weighted by exploration preference
            return random.random() < exploration_probability
        except Exception as e:
            print(f"Error in should_start_exploring: {e}")
            return False  # Default to not exploring

    def start_exploration(self):
        """Initialize exploration mode"""
        try:
            self.exploration_mode = True
            self.exploration_timer = random.randint(30, 60)  # Longer periods with higher preference
            self.exploration_direction = random.choice([-1, 1])

            # Update target region if needed
            if not self.current_target_region:
                self.current_target_region = self.get_next_exploration_target()
        except Exception as e:
            print(f"Error in start_exploration: {e}")
            self.exploration_mode = False  # Reset if something goes wrong

    def should_continue_exploring(self, opponent_detected, distance_to_opponent):
        """Determine if bot should continue exploring"""
        try:
            # Update timer
            self.exploration_timer -= 1

            # Exit conditions
            if self.exploration_timer <= 0:
                return False

            # Exit if opponent detected nearby and aggression is high
            if opponent_detected and distance_to_opponent < 30 and self.aggression_factor > 0.6:
                return False

            return True
        except Exception as e:
            print(f"Error in should_continue_exploring: {e}")
            return False  # Default to exiting exploration

    def calculate_targeting_angle(self, my_pos, target_pos, my_rot, fine_tune, exploring):
        """Calculate optimal rotation angle"""
        try:
            if exploring and self.current_target_region:
                # Target the region center during exploration
                target_x = (self.current_target_region[0] + 0.5) * self.grid_size
                target_y = (self.current_target_region[1] + 0.5) * self.grid_size

                dx = target_x - my_pos[0]
                dy = target_y - my_pos[1]
            else:
                # Target the opponent during combat
                dx = target_pos[0] - my_pos[0]
                dy = target_pos[1] - my_pos[1]

            # Calculate target angle
            target_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
            target_angle = (450 - target_angle) % 360

            # Calculate difference with current rotation
            angle_diff = ((target_angle - my_rot + 180) % 360) - 180

            # Add fine-tuning from ray detection when not exploring
            if not exploring:
                angle_diff += fine_tune

            return angle_diff
        except Exception as e:
            print(f"Error in calculate_targeting_angle: {e}")
            return 0  # Default to no rotation

    def should_shoot(self, opponent_detected, opponent_ray_index, rays, distance, optimal_distance, ammo):
        """Determine if the bot should shoot"""
        try:
            if not opponent_detected or ammo <= 0:
                return False

            # Perfect alignment (middle ray)
            if opponent_ray_index == len(rays) // 2:
                return True

            # Calculate probability based on alignment and distance
            center_index = len(rays) // 2
            ray_diff = abs(opponent_ray_index - center_index)

            # Distance factor (1.0 = optimal distance, 0.0 = far from optimal)
            distance_factor = max(0, 1 - abs(distance - optimal_distance) / 30)

            # Alignment factor (1.0 = perfect alignment, 0.0 = poor alignment)
            alignment_factor = max(0, 1 - ray_diff / 3)

            # Adjust weights based on aggression factor
            align_weight = 0.7 - (self.aggression_factor * 0.2)  # More aggressive = less concerned with alignment
            dist_weight = 0.3 + (
                        self.aggression_factor * 0.2)  # More aggressive = more willing to shoot at any distance

            # Combined probability
            shoot_probability = align_weight * alignment_factor + dist_weight * distance_factor

            # Apply aggression multiplier
            shoot_probability *= (0.5 + self.aggression_factor)

            return random.random() < shoot_probability
        except Exception as e:
            print(f"Error in should_shoot: {e}")
            return False  # Default to not shooting

    def get_unstuck_action(self, should_shoot):
        """Generate action to get unstuck"""
        try:
            # More aggressive pattern to break free
            return {
                "forward": self.step_counter % 2 == 0,
                "right": self.step_counter % 3 == 0,
                "down": self.step_counter % 5 == 0,
                "left": self.step_counter % 7 == 0,
                "rotate": 15,  # Rotate aggressively
                "shoot": should_shoot
            }
        except Exception as e:
            print(f"Error in get_unstuck_action: {e}")
            # Default unstuck action
            return {
                "forward": True,
                "right": True,
                "down": False,
                "left": False,
                "rotate": 15,
                "shoot": False
            }

    def get_exploration_action(self, angle_diff, should_shoot):
        """Generate action for exploration mode"""
        try:
            # Calculate rotation amount for exploration target
            rotation_amount = min(max(angle_diff, -10), 10)

            # Mix in some randomness to avoid local minima
            if random.random() < 0.1:
                rotation_amount += random.uniform(-5, 5)

            # Exploration movement optimized for covering distance
            return {
                "forward": True,
                "right": random.random() < 0.05,
                "down": False,
                "left": random.random() < 0.05,
                "rotate": rotation_amount,
                "shoot": False  # Don't waste ammo while exploring
            }
        except Exception as e:
            print(f"Error in get_exploration_action: {e}")
            # Default exploration action
            return {
                "forward": True,
                "right": False,
                "down": False,
                "left": False,
                "rotate": random.randint(-5, 5),
                "shoot": False
            }

    def get_combat_action(self, angle_diff, health, distance, opponent_detected, should_shoot):
        """Generate optimal combat action"""
        try:
            # Calculate rotation
            rotation_amount = 0
            if abs(angle_diff) > 5:  # Dead zone
                rotation_amount = min(max(angle_diff, -10), 10)

            # Distance-based tactics
            too_close = distance < 20
            too_far = distance > 60

            # Health-based tactics adjusted by aggression factor
            health_retreat_threshold = 30 * (
                        1 - self.aggression_factor * 0.5)  # More aggressive = lower retreat threshold
            retreating = health < health_retreat_threshold or (too_close and health < 60)
            advancing = too_far and health > 40

            # Dynamic strafing
            should_strafe = opponent_detected and distance < 40
            strafe_right = should_strafe and self.step_counter % 8 < 4

            return {
                "forward": advancing and not retreating,
                "right": strafe_right,
                "down": retreating,
                "left": should_strafe and not strafe_right,
                "rotate": rotation_amount,
                "shoot": should_shoot
            }
        except Exception as e:
            print(f"Error in get_combat_action: {e}")
            # Default combat action
            return {
                "forward": True,
                "right": False,
                "down": False,
                "left": False,
                "rotate": angle_diff,
                "shoot": should_shoot
            }

    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        try:
            return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        except (TypeError, ValueError) as e:
            print(f"Error in calculate_distance: {e}")
            return 100  # Default to a safe distance