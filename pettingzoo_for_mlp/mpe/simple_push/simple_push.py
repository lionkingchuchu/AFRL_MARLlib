# noqa: D212, D415
"""
# Simple Push

```{figure} mpe_simple_push.gif
:width: 140px
:name: simple_push
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_push_v3` |
|--------------------|---------------------------------------------|
| Actions            | Discrete/Continuous                         |
| Parallel API       | Yes                                         |
| Manual Control     | No                                          |
| Agents             | `agents= [adversary_0, agent_0]`            |
| Agents             | 2                                           |
| Action Shape       | (5)                                         |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5,))             |
| Observation Shape  | (8),(19)                                    |
| Observation Values | (-inf,inf)                                  |
| State Shape        | (27,)                                       |
| State Values       | (-inf,inf)                                  |


This environment has 1 good agent, 1 adversary, and 1 landmark. The good agent is rewarded based on the distance to the landmark. The adversary is rewarded if it is close to the landmark, and if the agent is far from the landmark (the difference of the distances). Thus the adversary must learn to
push the good agent away from the landmark.

Agent observation space: `[self_vel, goal_rel_position, goal_landmark_id, all_landmark_rel_positions, landmark_ids, other_agent_rel_positions]`

Adversary observation space: `[self_vel, all_landmark_rel_positions, other_agent_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_push_v3.env(max_cycles=25, continuous_actions=False)
```



`max_cycles`:  number of frames (a step for each agent) until game terminates

"""

"""
MPEv2 Note:
In simple_push, more than three landmarks is prohibited
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
    def __init__(self, max_cycles=25, continuous_actions=False, render_mode=None):
        EzPickle.__init__(
            self,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world()
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_push_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2

        world.agent_num = 0 # keep created agent num for naming agent
        world.adversary_num = 0 # keep created agent num for naming agent
        
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            if agent.adversary:
                world.adversary_num += 1
            else:
                world.agent_num += 1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def add_new_agent(self,world,np_random, adv):
        agent = Agent()
        agent.adversary = True if adv else False
        agent.name = f"adversary_{world.adversary_num}" if adv else f"agent_{world.agent_num}"  # naming agent based on created agent num
        agent.collide = True
        agent.silent = True
        if adv: world.adversary_num += 1
        else: world.agent_num += 1

        agent.goal_a = world.goal
        agent.color = np.array([0.25, 0.25, 0.25])
        if adv:
            agent.color = np.array([0.75, 0.25, 0.25])
        #else:
        #    j = world.goal.index
        #    agent.color[j+1] += 0.5
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
        

    def reset_world(self, world, np_random):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.5, 0.1])
            landmark.index = i
        # set goal landmark
        goal = np_random.choice(world.landmarks)
        goal.color = np.array([0.1,0.1,0.8])
        world.goal = goal
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            #else:
            #    j = goal.index
            #    agent.color[j + 1] += 0.5
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

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def agent_reward(self, agent, world):
        # the distance to the goal
        return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
        # return world.get_distance(agent,agent.goal_a)

    def adversary_reward(self, agent, world):
        # keep the nearest good agents away from the goal
        agent_dist = [
            np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
            #world.get_distance(agent,a)
            for a in world.agents
            if not a.adversary
        ]
        pos_rew = min(agent_dist)
        # nearest_agent = world.good_agents[np.argmin(agent_dist)]
        # neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
        neg_rew = np.sqrt(
            np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos))
        )
        #neg_rew = world.get_distance(agent,agent.goal_a)
        #np.sqrt(
        #    np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos))
        #)
        # neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
        return pos_rew - neg_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        landmark_rpos = []
        for entity in world.landmarks:  # world.entities:
            if entity == world.goal: continue
            landmark_rpos.append(entity.state.p_pos - agent.state.p_pos)
            # entity_pos.append(world.get_relpos(entity,agent))

        ally_rpos = []
        enemy_rpos = []
        for other in world.agents:
            if other is agent:
                continue
            #other_pos.append(world.get_relpos(other,agent))
            if agent.adversary:
                if other.adversary: # ally
                    ally_rpos.append(other.state.p_pos - agent.state.p_pos)
                else: # enemy
                    enemy_rpos.append(other.state.p_pos - agent.state.p_pos)
            else:
                if other.adversary: # enemy
                    enemy_rpos.append(other.state.p_pos - agent.state.p_pos)
                else: # ally
                    ally_rpos.append(other.state.p_pos - agent.state.p_pos)

        if not agent.adversary:
            return np.concatenate(
                [agent.state.p_pos] +
                [agent.state.p_vel]
                + [agent.goal_a.state.p_pos - agent.state.p_pos]
                #+ [world.get_relpos(agent.goal_a,agent)]
                + ally_rpos
                + landmark_rpos
                + enemy_rpos
            )
        else:
            return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + ally_rpos + landmark_rpos + enemy_rpos)

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