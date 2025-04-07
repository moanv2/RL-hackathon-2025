import math


class BestBot():
    def __init__(self):
        self.name = "BestBot"

    def reset_for_new_episode(self):
        print(f"{self.name} reset for new episode, aka do nothing")

    def act(self, info):
        try:
            my_pos = info["location"]
            my_rot = info["rotation"]  # 0–360
            rays = info["rays"]
            opponent_pos = info["closest_opponent"]

            # --- 1. Shoot if middle ray hits player ---
            if rays[2][-1] == "player":
                return {
                    "forward": False,
                    "right": False,
                    "down": False,
                    "left": False,
                    "rotate": 0,
                    "shoot": True,
                }

            # --- 2. Rotate toward opponent ---
            def get_target_angle(bot_pos, enemy_pos):
                dx = enemy_pos[0] - bot_pos[0]
                dy = enemy_pos[1] - bot_pos[1]
                angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                angle = (450 - angle) % 360  # rotate coordinate system: 0° = up, clockwise
                return angle


            def get_signed_angle_diff(current, target):
                # Returns diff in range [-180, 180]
                diff = (target - current + 540) % 360 - 180
                return diff

            target_angle = get_target_angle(my_pos, opponent_pos)
            signed_diff = get_signed_angle_diff(my_rot, target_angle)

            rotate_speed = 5
            dead_zone = 5  # more generous dead zone

            print(f"Signed angle diff: {signed_diff:.2f}")

            if abs(signed_diff) > dead_zone:
                rotate = rotate_speed if signed_diff > 0 else -rotate_speed
                print("Rotating", "right" if signed_diff > 0 else "left")
                return {
                    "forward": False,
                    "right": False,
                    "down": False,
                    "left": False,
                    "rotate": rotate,
                    "shoot": False,
                }
            else:
                print("Within dead zone — aligned")
                return {
                    "forward": True,
                    "right": False,
                    "down": False,
                    "left": False,
                    "rotate": 0,
                    "shoot": False,
                }

        except Exception as e:
            print(f"Error in act: {e}")
            return {
                "forward": False,
                "right": False,
                "down": False,
                "left": False,
                "rotate": 0,
                "shoot": False,
            }
