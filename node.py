from typing import TypedDict


class Node(TypedDict):
    """
    Simple class representing a node in the simulation. The nodes simply record how many
    packets they are carrying
    """
    packets: int