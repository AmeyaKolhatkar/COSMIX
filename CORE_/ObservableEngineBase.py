"""ObservableEngineBase — abstract base class for all observable-computing engines.

Each engine (BackgroundKinematics, GrowthKinematics, …) advertises a
``capabilities`` frozenset of observable names.  The EngineResolver inspects
this set to decide which engine to invoke for each requested observable.

Concrete engines must set the class attribute::

    capabilities = {"H", "dL", ...}

and implement a method for each capability name.
"""

class ObservableEngineBase:
    """
    Base Class for kinematics / physics engines that produce observables
    """
    capabilities = set()

    def provides(self, name):
        return name in self.capabilities