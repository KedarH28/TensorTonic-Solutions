def value_iteration_step(values, transitions, rewards, gamma):
    num_states = len(values)
    num_actions = len(transitions[0])
    
    new_values = []
    
    for s in range(num_states):
        best_value = float('-inf')
        
        for a in range(num_actions):
            q = rewards[s][a]
            
            for s_next in range(num_states):
                q += gamma * transitions[s][a][s_next] * values[s_next]
            
            best_value = max(best_value, q)
        
        new_values.append(best_value)
    
    return new_values