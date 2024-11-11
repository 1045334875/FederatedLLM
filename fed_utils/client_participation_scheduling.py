import numpy as np


def client_selection(num_clients, client_selection_frac, client_selection_strategy, other_info=None):
    np.random.seed(other_info)
    if client_selection_strategy == "random":
        num_selected = max(int(client_selection_frac * num_clients), 1)
        selected_clients_set = set(np.random.choice(np.arange(num_clients), num_selected, replace=False))
    elif client_selection_strategy == "fix":
        num_selected = max(int(client_selection_frac * num_clients), 1)
        selected_clients_set = set(np.arange(num_clients))

    return selected_clients_set
