from QLearning import QLearning


def throughput_reward(self: QLearning, simulation_step):
    if simulation_step == 0:
        return 0
    return self.environment.ground_station.packets / simulation_step


def delivery_packets_reward(self: QLearning, simulation_step):
    gs_packets = self.environment.ground_station.packets
    if simulation_step == 0:
        score = 0
    elif gs_packets - self.last_gs_packets > 0:
        score = gs_packets - self.last_gs_packets
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
    score = 0
    for agent in self.environment.agents:
        if agent.position == 5:
            score += 1
    return score


def gs_reward(self: QLearning, simulation_step):
    gs_packets = self.environment.ground_station.packets
    return gs_packets


def delivery_and_pickup(self: QLearning, simulation_step):
    if simulation_step == 0:
        self.last_packets = [sensor.count_packets() for sensor in self.environment.sensors]
        self.last_gs = self.environment.ground_station.packets
        return 0

    score = 0

    # Pickup score
    packets = [agent.packets for agent in self.environment.agents]
    for packet, old_packet in zip(packets, self.last_packets):
        if packet > old_packet:
            score += 1

    # Delivery score
    gs = self.environment.ground_station.packets
    if gs > self.last_gs:
        score += (gs - self.last_gs) * 5

    self.last_packets = packets
    self.last_gs = gs
    return score


def unique_packets(self: QLearning, simulation_step):
    if simulation_step == 0:
        self.last_agent_sources = [agent.sources.copy() for agent in self.environment.agents]
        return 0

    score = 0
    for index, agent in enumerate(self.environment.agents):
        if agent.position == 0 and len(self.last_agent_sources[index]) == len(self.environment.sensors):
            score += 1

    self.last_agent_sources = [agent.sources.copy() for agent in self.environment.agents]
    return score
