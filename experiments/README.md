# Experiments

This directory will be used to keep track of all potential bot contestants.  
At some point, we will use every single bot here to make a giant tournament, and pick the best one out of all for the final submission.

Every single experiment inside this directory will contain:

- `model_bot_0.pth`: The trained weights of the model of bot 0.
- `model_bot_1.pth`: The trained weights of the model of bot 1.
- `bot.py`: File containing the Model class and the Bot Class (with its act method).
- `reward_function.py`: Will contain the calculate_reward() function that was used to train the model.
- `config.json`: The configuration that was used for training.
- `rewards_plot.png`: The history of the rewards for both bots (0 and 1).  
    
Only include bots that have been trained successfully (they learned how to maximize reward)