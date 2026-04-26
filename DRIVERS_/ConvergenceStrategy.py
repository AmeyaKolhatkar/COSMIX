# Convergenge Strategy Base Class

from abc import ABC, abstractmethod

class ConvergenceStrategy(ABC):
    """
    Defined how MCMC is run and when it stops
    """

    @abstractmethod
    def run(self):
        """
        returns:
            Either MCMCResults or MultiChainResults
        """
        pass

    @abstractmethod
    def is_converged(self):
        """
        returns:
            bool
        """
        pass

    @abstractmethod
    def summary(self):
        """
        returns a dict of metadata summary
        """
        return {
            "mode": self.__class__.__name__,
            "converged": self.is_converged()
        }

