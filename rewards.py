from QLearning import QLearning


def throughput_reward(self: QLearning, simulation_step):
    if simulation_step == 0:
        return 0
    highest_throughput = (self.configuration['mission_size'] - 1) * (
            simulation_step / self.configuration['sensor_generation_frequency'])
    return self.environment.ground_station.packets / highest_throughput


def delivery_packets_reward(self: QLearning, simulation_step):
    gs_packets = self.environment.ground_station.packets
    if simulation_step == 0:
        score = 0
    elif gs_packets - self.last_gs_packets > 0:
        generated_packets = (self.configuration['mission_size'] - 1) * \
                            (simulation_step / self.configuration['sensor_generation_frequency'])
        score = gs_packets - self.last_gs_packets / generated_packets
    else:
        score = 0
    self.last_gs_packets = gs_packets
    self.last_step = simulation_step
    return score


def delivery_reward(self: QLearning, simulation_step):
    gs_packets = self.environment.ground_station.packets
    if simulation_step == 0:
        score = 0
    elif gs_packets - self.last_gs_packets > 0:
        score = 10
    else:
        score = 0
    self.last_gs_packets = gs_packets
    return score


def movement_reward(self: QLearning, simulation_step):
    return sum(agent.position for agent in self.environment.agents) / (len(self.environment.agents) * self.configuration['mission_size'])

