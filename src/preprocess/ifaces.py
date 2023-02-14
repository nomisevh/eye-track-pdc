import abc
import dataclasses
from abc import ABC
from typing import List, Any, Dict

import pandas as pd
from pandas import DataFrame


# --------------------------------------------------------

@dataclasses.dataclass
class MTSData(metaclass=abc.ABCMeta):
    x: pd.DataFrame = dataclasses.field(default_factory=lambda: pd.DataFrame())
    removed: bool = False
    _column_names: List = dataclasses.field(default_factory=lambda: [])

    def only_cols_(self, col_names: List[str]):
        self.x = self.x[col_names]

    @property
    def columns(self) -> List[str]:
        if len(self._column_names) > 0:
            return self._column_names
        return list(self.x.columns)

    @property
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


# --------------------------------------------------------


class Capturable(metaclass=abc.ABCMeta):
    """
    Generic interface for capturable instances, i.e. instances that can be saved and restored via state dicts.

    Methods
    -------
    state_dict
        Capture the state of the instance and return it in a python dict object.
    load_state_dict
        Restore the state of the instance and from the given state dict.
    """

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        raise NotImplementedError


class Operation(metaclass=abc.ABCMeta):
    """
    Generic interface for data processing classes.

    Methods
    -------
    __call__(inp)
        Invoke the operation for the given input.
    """

    @abc.abstractmethod
    def __call__(self, inp: Any) -> Any:
        raise NotImplementedError


class AtomicOperation(Operation):
    """
    Generic placeholder for operations acting on a single MTS.

    Methods
    -------
    __call__(mts=pd.DataFrame())
        Invoke the atomic operation for the input dataframe.
    """

    @abc.abstractmethod
    def __call__(self, mts: MTSData) -> MTSData or False:
        """
        Parameters
        ----------
        mts: MTSData
             The MTS input stored as pandas dataframe.

        Returns
        -------
        MTSData
            Returns the processed dataframe or False if the dataframe is filtered-out by the operation.
        """
        raise NotImplementedError


class BatchOperation(Operation):
    """
    Generic placeholder for operations acting on a batch of MTS.

    Methods
    -------
    __call__(batch=[pd.DataFrame(), pd.DataFrame()])
        Invoke the batch operation(s) for the input list of dataframe objects.
    """

    @abc.abstractmethod
    def __call__(self, batch: List[MTSData or Any]) -> List[MTSData or Any]:
        """
        Parameters
        ----------
        batch: list of MTSData
             The batch input comprising a list of pandas dataframe (or other) objects.

        Returns
        -------
        list of MTSData
            Returns the processed batch of dataframe (or other) objects.
        """
        raise NotImplementedError


# --------------------------------------------------------

class Scissor(AtomicOperation, Capturable):

    @abc.abstractmethod
    def __call__(self, frame: MTSData, **kwargs) -> MTSData or List[MTSData]:
        """
        Segments a multivariate time series (MTS)
        :param frame: A dataframe holding the MTS
        :param kwargs: Any processor-specific arguments
        :return: A list holding the segmented MTS
        """
        raise NotImplementedError
