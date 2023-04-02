from typing import TypedDict


class GroundStation:
    packets: int

    def __init__(self):
        self.packets = 0


class LifecyclePacket(TypedDict):
    created_at: int


class Sensor:
    lifecycle_packets: list[LifecyclePacket]

    def __init__(self):
        self.lifecycle_packets = []

    @property
    def packets(self):
        return len(self.lifecycle_packets)


class Agent:
    position: int
    packets: int

    def __init__(self):
        self.packets = 0
        self.position = 0