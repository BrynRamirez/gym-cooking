# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple
from itertools import product

import gym


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def initialize_agents(arglist):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    real_agent = RealAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                    real_agents.append(real_agent)

    return real_agents

def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    # game = GameVisualize(env)
    real_agents = initialize_agents(arglist=arglist)

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)

    while not env.done():
        action_dict = {}

        for agent in real_agents:
            action = agent.select_action(obs=obs)
            action_dict[agent.name] = action

        obs, reward, done, info = env.step(action_dict=action_dict)

        # Agents
        for agent in real_agents:
            agent.refresh_subtasks(world=env.world)

        # Saving info
        bag.add_status(cur_time=info['t'], real_agents=real_agents)


    # Saving final information before saving pkl file
    bag.set_collisions(collisions=env.collisions)
    bag.set_termination(termination_info=env.termination_info,
            successful=env.successful)

if __name__ == '__main__':
    # Define possible models
    models = ["bd", "up", "dc", "fb", "greedy"]

    # Iterate over all combinations of model1 and model2
    for model1, model2 in product(models, repeat=2):
        import argparse
        print(model1, model2)

        # Hardcoded argument values
        arglist = argparse.Namespace(
            level="open-divider_salad",
            num_agents=2,
            max_num_timesteps=100,
            max_num_subtasks=14,
            seed=1,
            with_image_obs=True,
            beta=1.3,
            alpha=0.01,
            tau=2,
            cap=75,
            main_cap=100,
            play=False,
            record=True,
            model1=model1,
            model2=model2,
            model3=None,
            model4=None
        )

        if arglist.play:
            env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
            env.reset()
            game = GamePlay(env.filename, env.world, env.sim_agents)
            game.on_execute()
        else:
            model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
            assert len(list(filter(lambda x: x is not None,
                                   model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
            fix_seed(seed=arglist.seed)
            main_loop(arglist=arglist)
