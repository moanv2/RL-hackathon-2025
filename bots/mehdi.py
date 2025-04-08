import math


class MehdiBot():
    def __init__(self):
        self.name = "Mehdi"

    def reset_for_new_episode(self):
        print(f"{self.name} reset for new episode, aka do nothing")

    def get_angle_between_vectors(self, a, b):
        # a and b are (x, y) vectors
        dot = a[0]*b[0] + a[1]*b[1]
        norm_a = math.hypot(*a)
        norm_b = math.hypot(*b)

        if norm_a == 0 or norm_b == 0:
            return 999  # big number so you can ignore degenerate cases

        # Clamp to avoid math domain errors
        cos_theta = max(min(dot / (norm_a * norm_b), 1.0), -1.0)
        angle_rad = math.acos(cos_theta)
        return math.degrees(angle_rad)
    
    def get_order_direction(self, rays, to_enemy):
        """
        Check external rays angle with to_enemy. the closest one means going to that direction

        Returns
            int: 1 (for clockwise rotation) or -1 (for anti clockwise rotation)
        """
        
        left_ray = rays[0]
        left_ray_start, left_ray_end = left_ray[0]
        left_view_dir = (left_ray_end[0] - left_ray_start[0], left_ray_end[1] - left_ray_start[1])

        right_ray = rays[-1]
        right_ray_start, right_ray_end = right_ray[0]
        right_view_dir = (right_ray_end[0] - right_ray_start[0], right_ray_end[1] - right_ray_start[1])

        left_angle = self.get_angle_between_vectors(left_view_dir, to_enemy)
        right_angle = self.get_angle_between_vectors(right_view_dir, to_enemy)

        if left_angle < right_angle:
            return -1
        
        return 1
    
    def get_rotation_speed(self):
        return 1

    def act(self, info):
        # From bot to enemy
        bot_pos = info["location"]
        enemy_pos = info["closest_opponent"]
        to_enemy = (enemy_pos[0] - bot_pos[0], enemy_pos[1] - bot_pos[1])

        # Viewing direction vector (from center ray)
        ray = info["rays"][2]  # assuming index 2 is center
        ray_start, ray_end = ray[0]
        view_dir = (ray_end[0] - ray_start[0], ray_end[1] - ray_start[1])

        angle = self.get_angle_between_vectors(view_dir, to_enemy)
        print(f"Angle between view and enemy: {angle:.2f}°")

        if angle < 1:  # within 5 degrees
            shoot = False
            if ray[-1] == "player":
                print("Aligned with enemy! SHOOT")
                shoot = True
            else:
                print("Looking at enemy, but there is an obstacle in front. I probably need to move")

            return {
                "rotate": 0, 
                "shoot": shoot, 
                "forward": False, 
                "left": False,
                "right": False, 
                "down": False
            }
        else:
            rotation_speed = self.get_rotation_speed()
            rotation_direction = self.get_order_direction(info["rays"], to_enemy)
            
            return {
                "forward": False,
                "right": False,
                "down": False,
                "left": False,
                "rotate": rotation_speed * rotation_direction,
                "shoot": False,
            }
        
        return {
            "rotate": 0, 
            "shoot": False, 
            "forward": False, 
            "left": False, 
            "right": False, 
            "down": False}

        