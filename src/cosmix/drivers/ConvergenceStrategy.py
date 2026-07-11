"""ConvergenceStrategy — abstract base class for all run-and-convergence strategies.

Defines the single required interface:

    run() → MCMCResults | MultiChainResults

Concrete strategies (SingleChainStrategy, MultiFixedStrategy,
MultiAutoConvergence) implement the run logic and convergence criteria
appropriate for their sampler type.
"""

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

