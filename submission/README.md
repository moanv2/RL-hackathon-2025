# Submission for team uhh umm
Team members:
- Diego
- Mehdi


## Overview

The submission is basically just the `bot.py` file (VonBot class). We don't have model weights saved as .pth because we don't use any model for the decision making in the `act()` method of the bot.

## Steps to use our bot
You simply need to import it and append it to the list of bots, before running the game:
```
from submission.bot import VonBot

"""
Set up the env and the other stuff thats supposed to be in the main function
"""

players = [
    Character(starting_pos=(world_bounds[2] - 100, world_bounds[3] - 100),
                screen=env.world_surface, boundaries=world_bounds, username="Ninja"),
    Character(starting_pos=(world_bounds[0] + 10, world_bounds[1] + 10),
                screen=env.world_surface, boundaries=world_bounds, username="Von"),
]

bots = []
# Append the opponent bot to the list of bots here

# Then append our bot
best_bot = VonBot()
bots.append(best_bot)

# Then run the game in display mode
run_game(env, players, bots)
```
