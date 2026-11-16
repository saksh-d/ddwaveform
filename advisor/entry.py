"""One advisor entry point, with distinct physical-model validation and workflows."""

def start_workflow(*,scenario,**settings):
    if scenario=='residual_cfo':
        from .cfo_workflow import start_cfo
        # Scaling bundles have no role in rectangular CFO qualification.
        for key in ('scaling_artifacts','scaling_bundle'): settings.pop(key,None)
        return start_cfo(**settings)
    if scenario=='dd_v1':
        from .configurable import start
        for key in ('artifacts','bundle'): settings.pop(key,None)
        return start(scenario=scenario,**settings)
    raise ValueError('SCENARIO must be dd_v1 or residual_cfo; no physical-model substitution')
