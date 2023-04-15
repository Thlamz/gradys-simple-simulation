
import unittest

from state import MobilityState, SignedMobilityState, CommunicationMobilityState


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
