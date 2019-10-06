from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

class ProtossAgent(base_agent.BaseAgent):

    def __init__(self):
        super(ProtossAgent, self).__init__()
        self.attack_coords = None
        self.switch_gateway = False
        self.probe_gas = False


    def step(self, obs):
        super(ProtossAgent, self).step(obs)
        free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
        #get attacking positions based on our position in the map, this is only done once
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()
            if xmean <= 31 and ymean <= 31:
                self.attack_coords = (52, 51)
            else:
                self.attack_coords = (10, 16)


        zealots = self.get_units_by_type(obs, units.Protoss.Zealot)
        #we are going to attack everytime there are more than 5 zealots
        if len(zealots) > 5:
            #we first check if they are selected, if not, then we select all the army to attack
            if self.unit_type_is_selected(obs, units.Protoss.Zealot) or self.unit_type_is_selected(obs, units.Protoss.Stalker):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now", self.attack_coords)
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")


        pylons = self.get_units_by_type(obs, units.Protoss.Pylon)
        #we will build a total of 2 pylons for the rush, we do not need more since
        #we will only send groups of 5 zealots
        if len(pylons) < 2:
            #we only have to check if a probe is selected, if not, we select one
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
                    x = abs(random.randint(0, 83))
                    y = abs(random.randint(0, 83))
                    if x > 83 or y > 83:
                        return actions.FUNCTIONS.no_op()
                    return actions.FUNCTIONS.Build_Pylon_screen("now", (x, y))
            return self.select_probe(obs)


        gateways = self.get_units_by_type(obs, units.Protoss.Gateway)
        #for the rush we will build a total of 3 gateways to maximize the train time of the zealots
        if len(gateways) < 3:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                #if we have a selected probe we can build a gateway, if not we select a probe
                if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
                    x = abs(random.randint(0, 83))
                    y = abs(random.randint(0, 83))
                    if x > 83 or y > 83:
                        return actions.FUNCTIONS.no_op()
                    return actions.FUNCTIONS.Build_Gateway_screen("now", (x, y))
            return self.select_probe(obs)
        else:
            #Once we have 3 gateways built or being built we can start trying to train zealots
            if self.unit_type_is_selected(obs, units.Protoss.Gateway) and not self.switch_gateway:
                #If a gateway is selected then we check if we can train a zealot
                if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
                    #If we can train a zealot then we train it and we set self.switch_gateway
                    #to true so that next iteration we choose a different gateway so that we
                    #use all of them and not just one
                    self.switch_gateway = True
                    return actions.FUNCTIONS.Train_Zealot_quick("now")
            else:
                #If there is no gateway chosen or we previously trained a zealot we want to choose
                #a random gateway to train a zealot
                gateway = random.choice(gateways)
                self.switch_gateway = False
                return actions.FUNCTIONS.select_point("select", (abs(gateway.x), abs(gateway.y)))


        return actions.FUNCTIONS.no_op()

    #This function is used to optimally choose a probe, it first checks if there is
    #and idle probe to choose it instead of the ones that are working.
    #If there is no idle worker then we choose one randomly
    def select_probe(self, obs):
        self.probe_gas = False
        if actions.FUNCTIONS.select_idle_worker.id in obs.observation.available_actions:
            return actions.FUNCTIONS.select_idle_worker("select")
        else:
            probes = self.get_units_by_type(obs,units.Protoss.Probe)
            if len(probes) > 0:
                probe = random.choice(probes)
                return actions.FUNCTIONS.select_point("select", (abs(probe.x), abs(probe.y)))


    def unit_type_is_selected(self, obs, unit_type):
        if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type:
            return True
        if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type:
            return True
        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions


def main(unused_argv):
    agent = ProtossAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="AbyssalReef",
                #map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.protoss),
                        sc2_env.Bot(sc2_env.Race.terran,
                        sc2_env.Difficulty.hard)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=25,
                game_steps_per_episode=0,
                visualize=False) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
