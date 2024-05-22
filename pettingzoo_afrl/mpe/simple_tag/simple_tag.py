# noqa: D212, D415
"""
# Simple Tag

```{figure} mpe_simple_tag.gif
:width: 140px
:name: simple_tag
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_tag_v3`                 |
|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [adversary_0, adversary_1, adversary_2, agent_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |


This is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By
default, there is 1 good agent, 3 adversaries and 2 obstacles.

So that good agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_tag_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles

        world.agent_num = 0 # keep created agent num for naming agent
        world.adversary_num = 0 # keep created agent num for naming agent
        world.landmark_num = 0 # keep created landmark num for naming landmark
        
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            if agent.adversary: world.adversary_num += 1
            else: world.agent_num += 1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
            world.landmark_num += 1
        return world

    def add_new_agent(self,world,np_random, adv):
        agent = Agent()
        agent.adversary = True if adv else False
        agent.name = f"adversary_{world.adversary_num}" if adv else f"agent_{world.agent_num}"  # naming agent based on created agent num
        agent.collide = True
        agent.silent = True
        agent.size = 0.075 if adv else 0.05
        agent.accel = 3.0 if adv else 4.0
        agent.max_speed = 1.0 if adv else 1.3
        if adv: world.adversary_num += 1
        else: world.agent_num += 1

        agent.color = np.array([0.85, 0.35, 0.35]) if adv else np.array([0.35, 0.85, 0.35])
        agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)

        world.new_agent_que.append(agent)
    
    def add_new_landmark(self,world,np_random):
        landmark = Landmark()
        landmark.name = "landmark %d" % world.landmark_num # naming agent based on created agent num
        landmark.collide = True
        landmark.movable = False
        landmark.size = 0.2
        landmark.boundary = False
        world.landmark_num += 1
        landmark.color = np.array([0.25, 0.25, 0.25])
        landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
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
            self.add_new_agent(world, np_random, 0)
            self.add_new_agent(world, np_random, 1)
            self.add_new_landmark(world, np_random)
        
        if simpleenv.steps == 9:
            self.kill_agent(world, world.agents[1])
            self.kill_landmark(world, world.landmarks[0])

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        
        world.index_map = {entity.name: idx for idx, entity in enumerate(world.entities)}
        world.calc_distmat()


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent, world):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2, world):
        dist = np.sqrt(np.sum(agent1.state.p_pos - agent2.state.p_pos)**2)
        #dist = world.get_distance(agent1, agent2)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(agent.state.p_pos - adv.state.p_pos)**2)
                # rew += 0.1 * world.get_distance(agent,adv)
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent, world):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(a.state.p_pos - adv.state.p_pos)**2)
                    # world.get_distance(a,adv)
                    for a in agents
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv, world):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        landmark_rpos = []
        for entity in world.landmarks:
            if not entity.boundary:
                landmark_rpos.append(entity.state.p_pos - agent.state.p_pos)
                # entity_pos.append(world.get_relpos(entity, agent))
        ally_rpos = []
        ally_vel = []
        enemy_rpos = []
        enemy_vel = []

        for other in world.agents:
            if other is agent:
                continue
            if agent.adversary:
                if not other.adversary:
                    enemy_rpos.append(other.state.p_pos - agent.state.p_pos)
                    enemy_vel.append(other.state.p_vel)
                else:
                    ally_rpos.append(other.state.p_pos - agent.state.p_pos)
                    ally_vel.append(other.state.p_vel)
            else:
                if other.adversary:
                    enemy_rpos.append(other.state.p_pos - agent.state.p_pos)
                    enemy_vel.append(other.state.p_vel)
                else:
                    ally_rpos.append(other.state.p_pos - agent.state.p_pos)
                    ally_vel.append(other.state.p_vel)

        return np.concatenate(
            [agent.state.p_pos]
            + [agent.state.p_vel]
            + landmark_rpos
            + ally_rpos
            + ally_vel
            + enemy_rpos
            + enemy_vel

        )
    
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
