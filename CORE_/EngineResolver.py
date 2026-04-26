"""EngineResolver — maps observable names to the engine that computes them.

Each engine (BackgroundKinematics, GrowthKinematics, …) advertises a set
of capability strings.  fill_theory_cache iterates over the requirements
dict produced by RequirementResolver, finds the right engine for each
observable, calls the corresponding method, and stores the result in the
TheoryCache.

The Requirement_map allows likelihoods to request aliased names
(e.g. "H_cc_cor" and "H_cc_unc" both map to the underlying "H" engine
method) without coupling the engine to likelihood-specific naming.
"""
Requirement_map = {
    "H": "H",
    "H_cc_cor": "H",
    "H_cc_unc": "H",
    "dL": "dL",
    "mu": "mu",
    "DM": "DM",
    "DH": "DH",
    "DV": "DV",
    "delta": "delta",
    "f": "f",
    "fsigma8": "fsigma8",
    "muG": "muG"
}

def resolve_engine(engines, observable_name):
    """
    Return the first engine that provides the observable.
    """
    physical_name = Requirement_map.get(observable_name, observable_name)
    for eng in engines:
        if physical_name in eng.capabilities:
            return eng, physical_name
    raise RuntimeError(f"No engine provides observable - {observable_name}")


def fill_theory_cache(theory, requirements, engines):
    """
    Populate TheoryCache using a list of engines
    """
    for name, req in requirements.items():
        z = req["z"]
        eng, physical_name = resolve_engine(engines, name)

        # call method name after observable
        method = getattr(eng, physical_name)
        values = method(z)
        if values is None:
            theory.mark_invalid()

            return theory

        theory.add(name, z, values)

    return theory