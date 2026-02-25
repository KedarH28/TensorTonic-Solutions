def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    num_actions = len(transitions[0])
    
    new_values = [0.0] * num_states
    
    for s in range(num_states):
        action_values = []
        
        for a in range(num_actions):
            expected_value = rewards[s][a]
            
            for s_next in range(num_states):
                expected_value += gamma * transitions[s][a][s_next] * values[s_next]
            
            action_values.append(expected_value)
        
        new_values[s] = max(action_values)
    
    return new_values