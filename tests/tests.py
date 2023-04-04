import itertools
import unittest

from base_serializer import tuple_to_base_id, base_id_to_tuple


class MyTestCase(unittest.TestCase):
    def test_base_transformation(self):
        last_base = -1
        for test in itertools.product((i for i in range(3)),  repeat=3):
            test = test[::-1]
            base = tuple_to_base_id(test, 3)
            self.assertEqual(base, last_base + 1)
            last_base = base
            self.assertEqual(test, base_id_to_tuple(base, 3, 3))


if __name__ == '__main__':
    unittest.main()
