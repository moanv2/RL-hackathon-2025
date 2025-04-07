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

        if angle < 5:  # within 5 degrees
            print("Aligned with enemy! SHOOT")
            return {"rotate": 0, "shoot": True, "forward": False, "left": False, "right": False, "down": False}
        else:
            # Always rotate right
            print("Not aligned → rotating right")
            return {
                "forward": False,
                "right": False,
                "down": False,
                "left": False,
                "rotate": 5,  # clockwise rotation
                "shoot": False,
            }
        
        return {
            "rotate": 0, 
            "shoot": False, 
            "forward": False, 
            "left": False, 
            "right": False, 
            "down": False}

        