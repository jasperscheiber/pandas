import pytest
import numpy as np

from pandas import Categorical, Index, Series
import pandas._testing as tm


@pytest.mark.parametrize("input,expected", [(0, 0), (5, 4), (2.5, 2)])
def test_series(input, expected):
    ser = Series([1, 2, 3, 4])
    exp = ser.searchsorted(input)
    assert exp == expected


@pytest.mark.parametrize("input,expected", [(0, 0), (11, 5), (5.4, 2)])
def test_index(input, expected):
    ind = Index([2, 4, 6, 8, 10])
    exp = ind.searchsorted(input)
    assert exp == expected


@pytest.mark.parametrize("input,expected", [(4, 4), (6, 6)])
def test_index_right(input, expected):
    ind = Index([2, 4, 4, 4, 6, 6])
    exp = ind.searchsorted(input, side='right')
    assert exp == expected


@pytest.mark.parametrize("input,expected", [([0, 7], [0, 6]), ([1.1, 1.2, 3.5], [1, 1, 3])])
def test_multiple(input, expected):
    ind = Index([1, 2, 3, 4, 5, 6])
    exp = ind.searchsorted(input)
    tm.assert_numpy_array_equal(exp, np.array(expected))


@pytest.mark.parametrize("input,expected", [("Alex", 0), ("Edward", 5), ("Callie", 2)])
def test_categorical(input, expected):
    names = ["Alex", "Brenda", "Callie", "Callie", "David", "Edward"]
    cat = Categorical(names, ordered=True)
    exp = cat.searchsorted(input)
    assert exp == expected


@pytest.mark.parametrize("input,sorter,expected", [(3, [3, 1, 2, 4, 0], 1)])
def test_sorter(input, sorter, expected):
    ind = Index([10, 4, 6, 2, 8])
    exp = ind.searchsorted(3, sorter=sorter)
    assert exp == expected
