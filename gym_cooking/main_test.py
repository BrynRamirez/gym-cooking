# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
import torch

from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

from navigation_planner.planners.mat_action_selector_brtdp import MATActionSelector, MATBRTDP
from delegation_planner.mat_bayesian_delegator import MATTaskAllocator, MATBayesianDelegator

import numpy as np
import random
import argparse
from collections import namedtuple

import gym

# Paths to MAT models
task_allocator_model_path = "/task_allocator_model.pth"
action_selector_model_path = "/action_selector_model.pth"



def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def initialize_agents(arglist):
    real_agents = []

    # Load MAT Task Allocator model
    task_allocator_model = MATTaskAllocator(
        input_dim=state_dim, embed_dim=64, num_heads=4, num_layers=2, output_dim=num_subtasks
    )
    task_allocator_model.load_state_dict(torch.load("/task_allocator_model.pth"))
    task_allocator_model.eval()

    # Load MAT Action Selector model
    action_selector_model = MATActionSelector(
        input_dim=state_dim, embed_dim=64, num_heads=4, num_layers=2, action_dim=num_actions
    )
    action_selector_model.load_state_dict(torch.load("/action_selector_model.pth"))
    action_selector_model.eval()

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            elif phase == 2:  # Phase 2: Read recipe list
                recipes.append(globals()[line]())

            elif phase == 3:  # Phase 3: Initialize agents
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    real_agent = RealAgent(
                        arglist=arglist,
                        name='agent-' + str(len(real_agents) + 1),
                        id_color=COLORS[len(real_agents)],
                        recipes=recipes
                    )

                    # Assign MAT Delegator or Action Selector based on the model type
                    if getattr(arglist, f"model{len(real_agents) + 1}") == "bd":
                        real_agent.delegator = MATBayesianDelegator(
                            agent_name=real_agent.name,
                            all_agent_names=[f"agent-{i+1}" for i in range(arglist.num_agents)],
                            transformer_model=task_allocator_model,
                            none_action_prob=0.1
                        )
                    elif getattr(arglist, f"model{len(real_agents) + 1}") in ["up", "dc", "fb", "greedy"]:
                        real_agent.action_selector = MATBRTDP(transformer_model=action_selector_model)

                    real_agents.append(real_agent)

    return real_agents


def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    real_agents = initialize_agents(arglist=arglist)

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)

    while not env.done():
        action_dict = {}

        for agent in real_agents:
            if hasattr(agent, "delegator"):  # Use MAT Task Allocator if available
                agent.delegator.delegate(env=env, subtask_allocs=env.all_subtasks)
                action = agent.select_action(obs=obs)  # Action after task allocation
            elif hasattr(agent, "action_selector"):  # Use MAT Action Selector
                action = agent.action_selector.select_next_action(env=env)
            else:
                action = agent.select_action(obs=obs)  # Default behavior

            action_dict[agent.name] = action

        obs, reward, done, info = env.step(action_dict=action_dict)

        # Agents
        for agent in real_agents:
            agent.refresh_subtasks(world=env.world)

        # Saving info
        bag.add_status(cur_time=info['t'], real_agents=real_agents)

    # Saving final information before saving pkl file
    bag.set_collisions(collisions=env.collisions)
    bag.set_termination(termination_info=env.termination_info, successful=env.successful)


if __name__ == '__main__':
    # Hardcoded argument values
    import argparse
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
        model1="bd",
        model2="up",
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



