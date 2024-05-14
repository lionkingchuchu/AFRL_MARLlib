# noqa: D212, D415
"""
# Simple Adversary

```{figure} mpe_simple_adversary.gif
:width: 140px
:name: simple_adversary
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_adversary_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete/Continuous                              |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= [adversary_0, agent_0,agent_1]`         |
| Agents             | 3                                                |
| Action Shape       | (5)                                              |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5))                   |
| Observation Shape  | (8),(10)                                         |
| Observation Values | (-inf,inf)                                       |
| State Shape        | (28,)                                            |
| State Values       | (-inf,inf)                                       |


In this environment, there is 1 adversary (red), N good agents (green), N landmarks (default N=2). All agents observe the position of landmarks and other agents. One landmark is the 'target landmark' (colored green). Good agents are rewarded based on how close the closest one of them is to the
target landmark, but negatively rewarded based on how close the adversary is to the target landmark. The adversary is rewarded based on distance to the target, but it doesn't know which landmark is the target landmark. All rewards are unscaled Euclidean distance (see main MPE documentation for
average distance). This means good agents have to learn to 'split up' and cover all landmarks to deceive the adversary.

Agent observation space: `[goal_rel_position, landmark_rel_position, other_agent_rel_positions]`

Adversary observation space: `[landmark_rel_position, other_agents_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False)
```



`N`:  number of good agents and landmarks

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

"""
MPEv2 Note:
In simple_adversary, there is no need to add more adversary
Keep in mind that goal landmark is defined at the initializing step.
Each agent has one goal landmark, so there is no need to kill goal landmark
If you want to kill goal landmark and create new one, you have to implement (1) assigning one landmark as a goal landmark (2) allocating goal landmark to each agents
"""

import numpy as np
from gym.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(self, N=2, max_cycles=25, continuous_actions=False, render_mode=None):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_adversary_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N + 1
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = num_agents - 1

        world.agent_num = 0 # keep created agent num for naming agent
        world.landmark_num = 0 # keep created landmark num for naming landmark

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
            world.agent_num += 1
        world.agent_num -= 1 # minus one adversary
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        return world
    
    def add_new_agent(self,world,np_random):
        agent = Agent()
        agent.adversary = False
        agent.name = f"agent_{world.agent_num}"  # naming agent based on created agent num
        agent.collide = False
        agent.silent = True
        agent.size = 0.15
        world.agent_num += 1

        agent.color = np.array([0.35, 0.35, 0.85])
        agent.goal_a = world.goal
        agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)

        world.new_agent_que.append(agent)
    
    def add_new_landmark(self,world,np_random):
        landmark = Landmark()
        landmark.name = "landmark %d" % world.landmark_num # naming agent based on created agent num
        landmark.collide = False
        landmark.movable = False
        landmark.size = 0.08
        world.landmark_num += 1

        landmark.color = np.array([0.15, 0.15, 0.15])
        landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)
        world.new_landmark_que.append(landmark)

    def kill_agent(self, world, agent):
        world.del_agent_que.add(agent.name)
    
    def kill_landmark(self, world, landmark):
        world.del_landmark_que.add(landmark.name)

    def update_que(self, world, simpleenv, np_random):
        # This method will be called right after world.step, you can fill world.add_agent_que or world.del_agent_que to add delete agent
        # You can get information of current world / environment through world / simpleenv args.

        #ex:
        for agent in world.agents:
            if agent.state.p_pos[0] < -100 and agent.state_p_pos[1] < -100: # under certain condition, delete that agent
                self.kill_agent(world,agent)
        
        #for testing
        if simpleenv.steps == 4:
            self.add_new_agent(world, np_random)
            self.add_new_agent(world, np_random)
            self.add_new_landmark(world, np_random)
        
        if simpleenv.steps == 9:
            self.kill_agent(world, world.agents[1])
            # self.kill_landmark(world, world.landmarks[0]) # killing goal landmark raises error

    def reset_world(self, world, np_random):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = np_random.choice(world.landmarks)
        world.goal = goal
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.index_map = {entity.name: idx for idx, entity in enumerate(world.entities)}
        world.calc_distmat()

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return world.get_distance(agent, agent.goal_a)
        else:
            dists = []
            for lm in world.landmarks:
                dists.append(world.get_distance(agent, lm))
            dists.append(
                world.get_distance(agent, agent.goal_a)
            )
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum(
                world.get_distance(a,a.goal_a)
                for a in adversary_agents
            )
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if (
                    world.get_distance(a, a.goal_a)
                    < 2 * a.goal_a.size
                ):
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                world.get_distance(a, a.goal_a)
                for a in good_agents
            )
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if (
                min(
                    world.get_distance(a, a.goal_a)
                    for a in good_agents
                )
                < 2 * agent.goal_a.size
            ):
                pos_rew += 5
            pos_rew -= min(
                world.get_distance(a, a.goal_a)
                for a in good_agents
            )
        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -world.get_distance(agent, agent.goal_a)
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if (
                world.get_distance(agent, agent.goal_a)
                < 2 * agent.goal_a.size
            ):
                adv_rew += 5
            return adv_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(world.get_relpos(entity, agent))
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(world.get_relpos(other, agent))

        if not agent.adversary:
            return np.concatenate(
                [world.get_relpos(agent.goal_a, agent)] + entity_pos + other_pos
            )
        else:
            return np.concatenate(entity_pos + other_pos)

    def spawn_position(self, mode, np_random, **kwargs):
        """
        mode: 'deterministic', 'uniform', 'gaussian'
        deterministic -> x: pos_x, y: pos_y
        uniform -> xlim: [minimum pos_x, maximum pos_x] ylim: [minimum pos_y, maximum pos_y]
        gaussian -> x: pos_x (mu), y: pos_y (mu), std: standard deviation
        """
        if mode == 'deterministic':
            return np.array([kwargs['x'],kwargs['y']])
        elif mode == 'uniform':
            return np_random.uniform(np.array([kwargs['xlim'][0], kwargs['ylim'][0]]), np.array([kwargs['xlim'][1], kwargs['ylim'][1]]))
        elif mode == 'gaussian':
            return np_random.normal(np.array([kwargs['x'],kwargs['y']]), kwargs['std'])
