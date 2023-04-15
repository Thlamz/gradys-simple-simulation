import itertools
import unittest

from base_serializer import tuple_to_base_id, base_id_to_tuple
from state import MobilityState, SignedMobilityState, CommunicationMobilityState


class UtilityTestCase(unittest.TestCase):
    def test_base_transformation(self):
        last_base = -1
        for test in itertools.product((i for i in range(3)), repeat=3):
            test = test[::-1]
            base = tuple_to_base_id(test, 3)
            self.assertEqual(base, last_base + 1)
            last_base = base
            self.assertEqual(test, base_id_to_tuple(base, 3, 3))


class StateSerializationTestCase(unittest.TestCase):
    def test_mobility_serialization(self):
        state = MobilityState([1, 2, 3])
        serialized = state.serialize()
        self.assertEqual(MobilityState.deserialize(serialized), state)

    def test_signed_mobility_serialization(self):
        state = SignedMobilityState([1, 20, 3])
        serialized = state.serialize()
        self.assertEqual(SignedMobilityState.deserialize(serialized), state)

    def test_communication_mobility_serialization(self):
        state = CommunicationMobilityState([1, 2, 3], [1, 2, 3])
        serialized = state.serialize()
        self.assertEqual(CommunicationMobilityState.deserialize(serialized), state)


if __name__ == '__main__':
    unittest.main()
