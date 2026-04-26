# Observable Engine Base Class

class ObservableEngineBase:
    """
    Base Class for kinematics / physics engines that produce observables
    """
    capabilities = set()

    def provides(self, name):
        return name in self.capabilities