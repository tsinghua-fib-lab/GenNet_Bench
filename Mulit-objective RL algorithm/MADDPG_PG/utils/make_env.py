

def make_env(scenario_name, benchmark=False, discrete_action=False):
    
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    
    world = scenario.make_world()
    
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation,
                            discrete_action=discrete_action)
    return env
