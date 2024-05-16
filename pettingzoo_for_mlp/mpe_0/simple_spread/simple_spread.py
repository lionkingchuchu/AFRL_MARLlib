# noqa: D212, D415
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v3` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |


This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""
"""
MPEv2 Note:
In simple spread, optimal solution is possible only if agent num == landmark num
"""
import numpy as np
from gym.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
import heapq

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=3):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        world.collaborative = True

        world.agent_num = 0 # keep created agent num for naming agent
        world.landmark_num = 0 # keep created landmark num for naming landmark

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            world.agent_num += 1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            world.landmark_num += 1
        return world
    
    def add_new_agent(self,world,np_random):
        agent = Agent()
        agent.name = f"agent_{world.agent_num}"  # naming agent based on created agent num
        agent.collide = True
        agent.silent = True
        agent.size = 0.15
        world.agent_num += 1

        agent.color = np.array([0.35, 0.35, 0.85])
        agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)

        world.new_agent_que.append(agent)

    def add_new_landmark(self,world,np_random):
        landmark = Landmark()
        landmark.name = "landmark %d" % world.landmark_num # naming agent based on created agent num
        landmark.collide = False
        landmark.movable = False
        world.landmark_num += 1

        landmark.color = np.array([0.25, 0.25, 0.25])
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
            self.kill_landmark(world, world.landmarks[0])

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
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
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                # world.get_distance(a,lm)
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent, world):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2, world):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist = world.get_distance(agent1, agent2)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                rew -= 1.0 * (self.is_collision(a, agent, world) and a != agent)
        return rew

    def global_reward(self, world):
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                # world.get_distance(a,lm)
                for a in world.agents
            ]
            rew -= min(dists)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # relpos of other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            # other_pos.heapq.heappush(other_pos,other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # other_pos.append(world.get_relpos(other, agent))
        
        # relpos of other landmarks
        landmark_rpos = []
        for entity in world.landmarks:  # world.entities:
            # landmark_rpos.heapq.heappush(landmark_rpos,entity.state.p_pos - agent.state.p_pos)
            landmark_rpos.append(entity.state.p_pos - agent.state.p_pos)
            # entity_pos.append(world.get_relpos(entity, agent))
        
        return np.concatenate(
            [agent.state.p_pos] + [agent.state.p_vel] + other_pos +  landmark_rpos
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