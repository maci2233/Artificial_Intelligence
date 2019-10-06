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
        self.zealots = 0
        self.zealot_limit = 6

    def step(self, obs):
        super(ProtossAgent, self).step(obs)
        free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()
            if xmean <= 31 and ymean <= 31:
                self.attack_coords = (52, 51)
            else:
                self.attack_coords = (10, 16)


        cybers = self.get_units_by_type(obs, units.Protoss.CyberneticsCore)
        zealots = self.get_units_by_type(obs, units.Protoss.Zealot)
        stalkers = self.get_units_by_type(obs, units.Protoss.Stalker)
        if len(zealots) + len(stalkers) > 9:
            if self.unit_type_is_selected(obs, units.Protoss.Zealot) or self.unit_type_is_selected(obs, units.Protoss.Stalker):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now", self.attack_coords)
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
        elif self.zealots == self.zealot_limit and len(cybers) == 0:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_CyberneticsCore_screen.id):
                    x = abs(random.randint(0, 83))
                    y = abs(random.randint(0, 83))
                    if x > 83 or y > 83:
                        return actions.FUNCTIONS.no_op()
                    return actions.FUNCTIONS.Build_CyberneticsCore_screen("now", (x, y))
            return self.select_probe(obs)


        pylons = self.get_units_by_type(obs, units.Protoss.Pylon)
        if pylons == 0 or free_supply < 5:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
                    x = abs(random.randint(0, 83))
                    y = abs(random.randint(0, 83))
                    if x > 83 or y > 83:
                        return actions.FUNCTIONS.no_op()
                    return actions.FUNCTIONS.Build_Pylon_screen("now", (x, y))
            #SELECTING PROBE
            return self.select_probe(obs)

        assims = self.get_units_by_type(obs, units.Protoss.Assimilator)
        if len(assims) == 0:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Assimilator_screen.id):
                    vespenes = self.get_units_by_type(obs, units.Neutral.VespeneGeyser)
                    vesp = random.choice(vespenes)
                    return actions.FUNCTIONS.Build_Assimilator_screen("now", (vesp.x, vesp.y))
            #SELECTING PROBE
            return self.select_probe(obs)
        else:
            assim = random.choice(assims)
            if assim['assigned_harvesters'] < 3 and assim.build_progress == 100:
                if self.unit_type_is_selected(obs, units.Protoss.Probe):
                    if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id) and not self.probe_gas:
                        self.probe_gas = True
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", (assim.x, assim.y))
                #SELECTING PROBE
                return self.select_probe(obs)


        gateways = self.get_units_by_type(obs, units.Protoss.Gateway)
        if len(gateways) < 2:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
                    x = abs(random.randint(0, 83))
                    y = abs(random.randint(0, 83))
                    if x > 83 or y > 83:
                        return actions.FUNCTIONS.no_op()
                    return actions.FUNCTIONS.Build_Gateway_screen("now", (x, y))
            #SELECTING PROBE
            return self.select_probe(obs)
        else:
            if self.unit_type_is_selected(obs, units.Protoss.Gateway) and not self.switch_gateway:
                #self.switch_gateway = True
                if self.can_do(obs, actions.FUNCTIONS.Train_Stalker_quick.id):
                    self.switch_gateway = True
                    return actions.FUNCTIONS.Train_Stalker_quick("now")
                if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id) and self.zealots < self.zealot_limit:
                    self.zealots += 1
                    self.switch_gateway = True
                    return actions.FUNCTIONS.Train_Zealot_quick("now")
            else:
                gateway = random.choice(gateways)
                self.switch_gateway = False
                return actions.FUNCTIONS.select_point("select", (abs(gateway.x), abs(gateway.y)))


        return actions.FUNCTIONS.no_op()


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
                        sc2_env.Difficulty.medium)],
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
