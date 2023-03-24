import unittest

from environment import tuple_to_base_id, base_id_to_tuple


class MyTestCase(unittest.TestCase):
    def test_base_transformation(self):
        data = (0, 5, 3, 1)
        base = tuple_to_base_id(data, 6)

        self.assertEqual(data, base_id_to_tuple(base, 6, 4))


if __name__ == '__main__':
    unittest.main()
