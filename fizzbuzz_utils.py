import math
import numpy as np
from sympy import sieve


class FizzBuzzExtended:

    # onomatopoeic words from http://onomatopoeialist.com/
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
        self._num_input_digits = int(math.log2(max_number - 1) + 1)
        self._num_output_digits = num_words

    @property
    def num_input_digits(self):
        return self._num_input_digits

    @property
    def num_output_digits(self):
        return self._num_output_digits

    @property
    def num_output_classes(self):
        return self._num_classes

    def binary_encode(self, number):
        """For neural network input"""
        if isinstance(number, np.ndarray):
            assert (number>=0).all() and (number<=0x7FFFFFFF).all(), "should be in range 0<=number<0x7FFFFFFF"
        else:
            assert 0<=number<=0x7FFFFFFF, "should be in range 0<=number<0x7FFFFFFF"

        return np.array([number >> d & 1 for d in range(self._num_input_digits)], dtype=np.uint8)

    def sparse_encode(self, number):
        """For neural network label, but for using in sparse_softmax_cross_entropy_with_logits"""
        assert 0 <= number <= self.max_number
        if number == 0:
            return 0
        ret = 0
        for i in range(self._num_output_digits):
            p = self._primes[i]
            ret |= ((number % p) == 0) << i
        return ret

    def decode(self, number, class_index) -> str:
        """For final output"""
        if class_index == 0:
            return str(number)
        else:
            ret = [FizzBuzzExtended.words[i] for i in range(self._num_output_digits) if (class_index >> i) & 1]
            return "".join(ret)



if __name__ == "__main__":
    fbe = FizzBuzzExtended(max_number=120, num_words=2)
    assert fbe.num_output_classes == 4
    assert fbe.num_input_digits == 7 # 0~127
    assert fbe.num_output_digits == 2
    assert (fbe.binary_encode(0) == np.array([0,0,0,0,0,0,0])).all()
    assert fbe.sparse_encode(0) == 0
    assert fbe.sparse_encode(1) == 0
    assert fbe.sparse_encode(2) == 0
    assert fbe.sparse_encode(3) == 1
    assert fbe.sparse_encode(4) == 0
    assert fbe.sparse_encode(5) == 2
    assert fbe.sparse_encode(6) == 1
    assert fbe.sparse_encode(15) == 3
    assert fbe.sparse_encode(105) == 3
    assert fbe.decode(11, 0) == "11"
    assert fbe.decode(3, 1) == "Fizz"
    assert fbe.decode(5, 2) == "Buzz"
    assert fbe.decode(14, 3) == "FizzBuzz"

    fbe = FizzBuzzExtended(max_number=200, num_words=3)
    assert fbe.num_output_classes == 8
    assert fbe.num_input_digits == 8 # 0~255
    assert fbe.num_output_digits == 3
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

    fbe = FizzBuzzExtended(max_number=200, num_words=4)
    assert fbe.decode(0, 1) == "Fizz"
    assert fbe.decode(0, 2) == "Buzz"
    assert fbe.decode(0, 4) == "Whizz"
    assert fbe.decode(0, 8) == "Rizzz"
    assert fbe.decode(0, 3) == "FizzBuzz"
    assert fbe.decode(0, 7) == "FizzBuzzWhizz"
    assert fbe.decode(0, 15) == "FizzBuzzWhizzRizzz"
