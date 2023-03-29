from typing import TypedDict


class Node:
    """
    Simple class representing a node in the simulation. The nodes simply record how many
    packets they are carrying
    """
    packets: int

    def __init__(self):
        self.packets = 0


class LifecyclePacket(TypedDict):
    created_at: int


class LifecycleNode(Node):
    lifecycle_packets: list[LifecyclePacket]

    def __init__(self):
        self.lifecycle_packets = []

    @property
    def packets(self):
        return len(self.lifecycle_packets)
