"""REGISTRIES FOR MODELS AND LIKELIHOODS.

Useful for automatic Registration of models/likelihoods
"""
from cosmix.core.CosmologyModelBase import CosmologyModelBase
from cosmix.core.LikelihoodBase_ import LikelihoodBase
from cosmix.samplers.SamplerBase import SamplerBase
class Registry:

    def __init__(self): 
        self.models = {}
        self.likelihoods = {}
        self.samplers = {}

    #------------ Models ------------#
    def register_model(self, name):
        def wrapper(cls):
            if issubclass(cls, CosmologyModelBase):
                self.models[name] = cls
            else:
                raise TypeError(f"[Registry] {cls.__name__} must inherit from CosmologyModelBase")
            return cls
        return wrapper
    
    def build_model(self, name, pm):
        return self.models[name](pm)
    
    #------------ Likelihoods ------------#
    def register_likelihood(self, name):
        def wrapper(cls):
            if issubclass(cls, LikelihoodBase):
                self.likelihoods[name] = cls
            else:
                raise TypeError(f"{cls.__name__} must inherit from LikelihoodBase")
            return cls
        return wrapper
    
    def build_likelihood(self, name, pm):
        return self.likelihoods[name](pm)
    
    #------------ Samplers ------------#
    def register_sampler(self, name):
        def wrapper(cls):
            if issubclass(cls, SamplerBase):
                self.samplers[name] = cls
            else:
                raise TypeError(f"{cls.__name__} must inherit from SamplerBase")
            return cls
        return wrapper

    def build_sampler(self, name, lnpost, **kwargs):
        return self.samplers[name](lnpost=lnpost, **kwargs)
    
# Instantiating a global Registry object
cosmix_registry = Registry()
            


        

