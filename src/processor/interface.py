import abc
from abc import ABC
from typing import List, Tuple

from numpy import ndarray
from pandas import DataFrame


class MainProcessor(ABC):

    @abc.abstractmethod
    def __call__(self, frames: List[DataFrame], **kwargs) -> Tuple[List[List[DataFrame]], ndarray]:
        """
        Processes and segments a list of multivariate time series (MTS)

        :param frames: A list containing dataframes which hold the MTS
        :param kwargs: Any processor-specific arguments
        :return: A list of lists holding all the processed and segmented MTS and a numpy array holding a mask signifying
        which frames were kept during any sanitation.
        """
        raise NotImplementedError


class BatchProcessor(ABC):

    @abc.abstractmethod
    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        """
        Processes a list of multivariate time series (MTS)

        :param frames: A list containing dataframes which hold the MTS
        :param kwargs: Any processor-specific arguments
        :return: A list holding all the processed MTS
        """
        raise NotImplementedError


class AtomicProcessor(ABC):

    @abc.abstractmethod
    def __call__(self, frames: DataFrame, **kwargs) -> DataFrame:
        """
        Processes a single multivariate time series (MTS)

        :param frames: A dataframes holding the MTS segment
        :param kwargs: Any processor-specific arguments
        :return: The processed MTS segment
        """
        raise NotImplementedError


class Scissor(ABC):

    @abc.abstractmethod
    def __call__(self, frame: DataFrame, **kwargs) -> List[DataFrame]:
        """
        Segments a multivariate time series (MTS)

        :param frame: A dataframe holding the MTS
        :param kwargs: Any processor-specific arguments
        :return: A list holding the segmented MTS
        """
        raise NotImplementedError
