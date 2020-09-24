import math
import numpy as np
from sympy import sieve

def _num_primes(num_classes):
    assert num_classes >= 2
    return int(math.floor(math.log2(num_classes - 1)) + 1)

class FizzBuzzExtended:

    # Onomatopoeia words from http://onomatopoeialist.com/
    words = ["Fizz", "Buzz", "Whizz", "Rizzz", "Zzzz", "Hiss",
             "Burr", "Chirr", "Purr", "Whirr", "Aha", "Blah",
             "Duh", "Huh", "Shuh", "Crow", "Meow", "Neow", "Yeow",
             "Achoo","Aroo", "Boo", "Choo", "Ooh", "Shoo", "Whoo"]

    def __init__(self, max_number, num_words):
        assert max_number <= 0x7FFFFFFF
        assert num_words > 0
        self.max_number = max_number
        self.num_words = num_words
        self._num_classes = 2 ** num_words

        sieve.extend_to_no(num_words + 1) # 2 is omited
        self._primes = sieve._list[1:2+num_words]
        self._num_digit = num_words

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_digit(self):
        return self._num_digit

    def binary_encode(self, number):
        """For neural network input"""
        assert 0<=number<0x7FFFFFFF, "should be int32"
        return np.array([number >> d & 1 for d in range(self._num_digits)], dtype=np.uint8)

    def sparse_encode(self, number):
        """For neural network label, but for using in sparse_softmax_cross_entropy_with_logits"""
        assert 0 <= number <= self.max_number
        if number == 0:
            return 0
        ret = 0
        for i, p in enumerate(self._primes):
            ret |= ((number % p) == 0) << i
        return ret

    def decode(self, number, class_index) -> str:
        """For final output"""
        if class_index == 0:
            return str(number)
        else:
            ret = [FizzBuzzExtended.words[i] for i in range(self._num_digit) if (class_index >> i) & 1]
            return "".join(ret)



if __name__ == "__main__":
    fbe = FizzBuzzExtended(max_number=100, num_words=2)
    assert fbe.num_classes == 4
    assert fbe.num_digit == 2
    assert fbe.sparse_encode(0) == 0
    assert fbe.sparse_encode(1) == 0
    assert fbe.sparse_encode(2) == 0
    assert fbe.sparse_encode(3) == 1
    assert fbe.sparse_encode(4) == 0
    assert fbe.sparse_encode(5) == 2
    assert fbe.sparse_encode(6) == 1
    assert fbe.sparse_encode(15) == 3
    assert fbe.decode(11, 0) == "11"
    assert fbe.decode(3, 1) == "Fizz"
    assert fbe.decode(5, 2) == "Buzz"
    assert fbe.decode(14, 3) == "FizzBuzz"

    fbe = FizzBuzzExtended(max_number=200, num_words=3)
    assert fbe.num_classes == 8
    assert fbe.num_digit == 3
    assert fbe.sparse_encode(1) == 0
    assert fbe.sparse_encode(3) == 1
    assert fbe.sparse_encode(5) == 2
    assert fbe.sparse_encode(7) == 4
    assert fbe.sparse_encode(9) == 1
    assert fbe.sparse_encode(10) == 2
    assert fbe.sparse_encode(13) == 0
    assert fbe.sparse_encode(15) == 3
    assert fbe.sparse_encode(21) == 5
    assert fbe.sparse_encode(35) == 6
    assert fbe.sparse_encode(105) == 7
    assert fbe.decode(11, 0) == "11"
    assert fbe.decode(3, 1) == "Fizz"
    assert fbe.decode(5, 2) == "Buzz"
    assert fbe.decode(14, 3) == "FizzBuzz"
    assert fbe.decode(7, 4) == "Whizz"
    assert fbe.decode(0, 4) == "Whizz"
    assert fbe.decode(21, 5) == "FizzWhizz"
    assert fbe.decode(35, 6) == "BuzzWhizz"
    assert fbe.decode(105, 7) == "FizzBuzzWhizz"
