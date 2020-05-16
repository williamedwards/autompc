# Created by William Edwards (wre2@illinois.edu)

from enum import Enum
from abc import ABC, abstractproperty

class HyperType(Enum):
    """
    Enumeration for hyperparameter types.
    """
    boolean = 1
    int_range = 2
    float_range = 3
    choice = 4
    multi_choice = 5

class Hyperparam(ABC):
    @abstractproperty
    def type(self):
        pass

    @abstractproperty
    def options(self):
        pass

class BooleanHyperparam(Hyperparam):
    def __init__(self, default_value=False):
        self._value = default_value

    @property
    def type(self):
        return HyperType.boolean

    @property
    def options(self):
        return None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if not isinstance(val, bool):
            raise ValueError("Hyperparameter value is wrong type.")
        self._value = val

class IntRangeHyperparam(Hyperparam):
    def __init__(self, bounds, default_value=None):
        self.lower = bounds[0]
        self.upper = bounds[1]
        if default_value is None:
            self._value = bounds[0]
        else:
            self._value = default_value

    @property
    def type(self):
        return HyperType.int_range

    @property
    def options(self):
        return (self.lower, self.upper)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if not isinstance(val, int):
            raise ValueError("Hyperparameter value is wrong type.")
        if self.lower <= val < self.upper:
            self._value = val
        else:
            raise ValueError("Hyperparameter value out of range")

class FloatRangeHyperparam(Hyperparam):
    def __init__(self, bounds, default_value=None):
        self.lower = bounds[0]
        self.upper = bounds[1]
        if default_value is None:
            self._value = bounds[0]
        else:
            self._value = default_value

    @property
    def type(self):
        return HyperType.float_range

    @property
    def options(self):
        return (self.lower, self.upper)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if not isinstance(val, float):
            raise ValueError("Hyperparameter value is wrong type.")
        if self.lower <= val < self.upper:
            self._value = val
        else:
            raise ValueError("Hyperparameter value out of range")

class ChoiceHyperparam(Hyperparam):
    def __init__(self, choices, default_value=None):
        self.choices = choices
        if default_value is None:
            self._value = choices[0]
        else:
            self._value = default_value

    @property
    def type(self):
        return HyperType.choice

    @property
    def options(self):
        return self.choices[:]

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if  val in self.choices: 
            self._value = val
        else:
            return ValueError("Hyperparameter value not in choices")

class MultiChoiceHyperparam(Hyperparam):
    def __init__(self, choices, default_value=None):
        self.choices = choices
        if default_value is None:
            self._value = set()
        else:
            self._value = default_value

    @property
    def type(self):
        return HyperType.multi_choice

    @property
    def options(self):
        return self.choices[:]

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if val <= set(self.choices):
            self._value = val
        else:
            return ValueError("Hyperparameter values not in choices")
