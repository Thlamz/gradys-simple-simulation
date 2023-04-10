import heapq
import math
from collections import deque
from typing import TypedDict, List, Tuple, Deque


class GroundStation:
    packets: int

    def __init__(self):
        self.packets = 0


class Sensor:
    _lifecycle_packets: Deque[int]
    _packets: int

    def __init__(self):
        self._lifecycle_packets = deque()
        self._packets = 0

    def _validate_packets(self, simulation_step, lifecycle):
        while len(self._lifecycle_packets) > 0 and (simulation_step - self._lifecycle_packets[0]) > lifecycle:
            deque.popleft(self._lifecycle_packets)

    def add_packet(self, simulation_step, lifecycle):
        if math.isinf(lifecycle):
            self._packets += 1
        else:
            self._validate_packets(simulation_step, lifecycle)
            self._lifecycle_packets.append(simulation_step)

    def num_packets(self, simulation_step, lifecycle):
        if math.isinf(lifecycle):
            return self._packets
        else:
            self._validate_packets(simulation_step, lifecycle)
            return len(self._lifecycle_packets)

    def clear_packets(self):
        self._lifecycle_packets.clear()
        self._packets = 0


class Agent:
    position: int
    packets: int
    reversed: bool

    def __init__(self):
        self.packets = 0
        self.position = 0
        self.reversed = False
