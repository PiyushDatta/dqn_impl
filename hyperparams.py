import os

curr_try = 1
hyperparams_dict = {
    # total eps, printing, and memory
    'total_episodes':  1000,
    'replay_memory_cap': 50000,
    'progress_per_iteration': 10,

    # learning rate
    'learning_rate':  0.001,
    'learning_rate_decay': 0.00,
    # 'learning_rate_decay':  0.001,

    # epsilon
    'epsilon': 1.0,
    'max_epsilon_episodes': 100,
    'min_epsilon': 0.01,
    'epsilon_decay': 0.00,
    # 'epsilon_decay': 0.995,

    # other factors
    'discount_factor': 0.99,
    'batch_size': 32,
    'copy_max_step': 15,
    'hidden_layer_size': 24,

    # weights save path
    'checkpoint_path': os.path.join(os.path.join(os.getcwd(), 'nn_saved_weights'), 'training_%s.pth' % curr_try),

    # results save path
    'saved_results_path': os.path.join(os.path.join(os.getcwd(), 'saved_results'), 'training_%s' % curr_try),

    # results save name
    'saved_results_name': '',
}
