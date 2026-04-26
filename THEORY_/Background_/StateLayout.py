# state layout class

class StateLayout:
    """
    Defines the structure of the ODE state vector 'y'

    Maps variable names -> indices and locks ordering

    Used by:
        - Equation System
        - Augmentors
        - Builders
    """
    def __init__(self):
        self._index = {}
        self._names = []
        self._frozen = False

    def add(self, name):
        """
        Register a state variable; must be called before locking
        """
        if self._frozen:
            raise RuntimeError("StateLayout is frozen; cannot add state variables")
        
        if name in self._index:
            raise ValueError(f"State variable '{name}' already defined.")
        
        self._index[name] = len(self._names)
        self._names.append(name)

    def add_many(self, names):
        for n in names:
            self.add(n)

    def freeze(self):
        """
        Prevents further modification.
        Must be called after layout construction
        """
        if not self._names:
            raise RuntimeError("Cannot lock an empty StateLayout")
        self._frozen = True

    def index(self, name):
        """
        return index of a state variable
        """
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"State variable '{name}' not found in layout.")
        
    def name(self, idx):
        return self._names[idx]
    
    @property
    def size(self):
        return len(self._names)
    
    @property
    def names(self):
        return list(self._names)
    

    def extract(self, y, name):
        """
        Extract the variable 'name' from a state vector 'y'
        """
        return y[self.index(name)]
    
    def assign(self, y, name, value):
        """
        Assign a value 'value' to the variable 'name' of the state vector 'y'
        """
        y[self.index(name)] = value

    def summary(self):
        print("State Laout:")
        for n, i in self._index.items():
            print(f"    {i:.2d}  -  {n}")
