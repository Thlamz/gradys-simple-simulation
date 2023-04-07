from QLearning import QLearning


def throughput_reward(self: QLearning, simulation_step):
    if simulation_step == 0:
        return 0
    highest_throughput = (self.configuration['mission_size'] - 1) * (
            simulation_step / self.configuration['sensor_generation_frequency'])
    return self.environment.ground_station.packets / highest_throughput


def delivery_reward(self: QLearning, simulation_step):
    gs_packets = self.environment.ground_station.packets
    if simulation_step == 0:
        self.last_gs_packets = self.environment.ground_station.packets
        return 0

    if gs_packets - self.last_gs_packets:
        return 1
    else:
        return 0