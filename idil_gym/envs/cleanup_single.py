import numpy as np
from idil_gym.envs.mdp_env.env_from_mdp import EnvFromMDP
from hcair_domains.cleanup_single.mdp import MDPCleanupSingle
from hcair_domains.cleanup_single.maps import MAP_SINGLE_V1
from hcair_domains.cleanup_single.policy import Policy_CleanupSingle
from hcair_domains.cleanup_single.agent import Agent_CleanupSingle
import gym
import argparse


class CleanupSingleEnv_v0(EnvFromMDP):

  def __init__(self):
    game_map = MAP_SINGLE_V1
    mdp = MDPCleanupSingle(**game_map)
    init_bstate = [0] * len(game_map["boxes"])
    init_pos = game_map["init_pos"]
    mdp.walls
    mdp.x_grid
    mdp.y_grid

    possible_init_states = []
    for x in range(mdp.x_grid):
      for y in range(mdp.y_grid):
        pt = (x, y)
        if pt not in mdp.walls and pt not in mdp.goals and pt not in mdp.boxes:
          sidx = mdp.conv_sim_states_to_mdp_sidx((init_bstate, pt))
          possible_init_states.append(sidx)

    # init_sidx = mdp.conv_sim_states_to_mdp_sidx((init_bstate, init_pos))
    # possible_init_states = [init_sidx]

    super().__init__(mdp, possible_init_states, use_central_action=True)


class CleanupSingleExpert:

  def __init__(self, temperature) -> None:
    mdp_task = MDPCleanupSingle(**MAP_SINGLE_V1)
    policy = Policy_CleanupSingle(mdp_task, temperature)
    self.agent = Agent_CleanupSingle(policy)

    self.PREV_LATNET = mdp_task.num_latents

    print(mdp_task.num_latents)
    print(mdp_task.num_states)

  def choose_action(self, state, prev_option, prev_action, sample=True):
    mdp = self.agent.agent_model.get_reference_mdp()
    sim_state = mdp.conv_mdp_sidx_to_sim_states(state)

    if prev_option == self.PREV_LATNET:
      self.agent.init_latent(sim_state)
    else:
      prev_latent = self.agent.conv_idx_to_latent(prev_option)
      self.agent.set_latent(prev_latent)
      self.agent.update_mental_state(None, (None, ), sim_state)

    latent = self.agent.agent_model.current_latent
    sim_action = self.agent.get_action(sim_state)
    action = self.agent.agent_model.policy_model.conv_action_to_idx(
        (sim_action, ))[0]

    return latent, action


if __name__ == "__main__":
  from collections import defaultdict
  import os
  import pickle

  parser = argparse.ArgumentParser()
  parser.add_argument("--save_dir", type=str, default="test_gen")
  parser.add_argument("--n_traj", type=int, default=50)
  parser.add_argument("--temperature", type=float, default=0.3, required=False)
  args = parser.parse_args()

  env = gym.make("CleanupSingle-v0")

  # expert agent (if you want to test any trained agent, load it here)
  TEMPERATURE = args.temperature if args.temperature is not None else 0.3
  agent = CleanupSingleExpert(TEMPERATURE)

  # generate data
  ############################################################################
  num_data = args.n_traj
  cur_dir = os.path.dirname(__file__)
  save_dir = os.path.join(cur_dir, args.save_dir)

  expert_trajs = defaultdict(list)
  for idx in range(num_data):
    state = env.reset()

    s_array = []
    a_array = []
    r_array = []
    x_array = []
    prev_latent = agent.PREV_LATNET
    done = False
    while not done:
      latent, action = agent.choose_action(state,
                                           prev_latent,
                                           None,
                                           sample=False)
      next_state, reward, done, info = env.step(action)
      s_array.append(state)
      a_array.append(action)
      r_array.append(reward)
      x_array.append(latent)

      state = next_state
      prev_latent = latent

    s_array.append(state)

    length = len(r_array)
    dones = [False] * length
    dones[-1] = done

    expert_trajs["states"].append(s_array[:-1])
    expert_trajs["next_states"].append(s_array[1:])
    expert_trajs["actions"].append(a_array)
    expert_trajs["latents"].append(x_array)
    expert_trajs["rewards"].append(r_array)
    expert_trajs["dones"].append(dones)
    expert_trajs["lengths"].append(length)

  rewards = [sum(r_arr) for r_arr in expert_trajs["rewards"]]
  print("mean reward:", np.mean(rewards))
  file_path = os.path.join(save_dir, f"CleanupSingle-v0_{num_data}.pkl")
  with open(file_path, 'wb') as f:
    pickle.dump(expert_trajs, f)
