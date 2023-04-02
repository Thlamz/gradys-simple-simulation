def tuple_to_base_id(t: tuple, base: int) -> int:
    """
    Auxiliary function that is used to serialize a tuple with a known max value to an integer
    :param t: Tuple of integers
    :param base: Known max value of the tuple
    :return: Integer representing the tuple
    """
    result = 0
    for index, data in enumerate(t):
        result += base ** index * data
    return result


def base_id_to_tuple(n: int, base: int, size: int) -> tuple:
    """
    Auxiliary functions that deserializes an integer to a tuple with a known max value and size
    :param n: Number to deserialzie
    :param base: Known max value of the tuple
    :param size: Size of the tuple
    :return: Deserialized tuple
    """
    result = [0 for _ in range(size)]
    digit = 0
    while n >= base:
        remainder = n % base
        result[digit] = int(remainder)
        n = (n - remainder) / base
        digit += 1
    result[digit] = int(n)
    return tuple(result)