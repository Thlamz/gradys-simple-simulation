import unittest
from pathlib import Path

from QLearning import SparseQTable
from control import Control, MobilityCommand
from environment import Environment
from campaign import get_default_configuration
from state_test import MobilityState


class QTableTestCase(unittest.TestCase):
    def test_sparse_qtable(self):
        config = get_default_configuration()
        q_table_file = Path("./__sparse_qtable.json")
        config['model_file'] = q_table_file
        env = Environment(config)

        q_table = SparseQTable(config, env)
        state = MobilityState([0])
        control = Control((MobilityCommand.FORWARDS,))
        q_table.set_q_value(state, control, 0)
        self.assertEqual(q_table.get_q_value(state, control), 0)
        control2 = Control((MobilityCommand.REVERSE,))
        q_table.set_q_value(state, control2, 1)
        self.assertEqual(q_table.get_optimal_control(state), control2)

        state2 = MobilityState([1])
        # Getting non existanting q_value returns initialization value
        self.assertEqual(q_table.get_q_value(state2, control), config['qtable_initialization_value'])
        # Optimal control for non existanting state is random
        self.assertEqual(q_table.get_q_value(state2, q_table.get_optimal_control(state2)),
                         config['qtable_initialization_value'])

        q_table.export_qtable()

        loaded_q_table = SparseQTable(config, env)
        self.assertEqual(q_table.q_table, loaded_q_table.q_table)

        q_table_file.unlink()


if __name__ == '__main__':
    unittest.main()
