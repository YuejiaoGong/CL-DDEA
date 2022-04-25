import numpy as np
import os
import sampling

def procs_params(kw):
    """update the configuration
    """
    # init configuration
    config = {
        "d": None,                    # dimension. If not provided, it will be got from `config["samples"]` or `config["pop"]`
        "n": None,                    # the number of offline data. If not provided, it will be got from `config["samples"]`. Default 11 * d
        "lower_bound": None,          # the lower bound of the decision variables
        "upper_bound": None,          # the upper bound of the decision variables
        "pc": 1,                      # probality of crossover
        "pm": None,                   # probality of mutation, default 1 / d
        "eta_c": 15,                  # distribution index of simulated binary crossover
        "eta_m": 15,                  # distribution index of polynomial mutation
        "max_iter": 100,              # maximum number of iterations
        "max_run": 20,                # maximum number of running
        "cmp_model": None,            # comparison-based model
        "reg_model": None,            # regression model
        "run": None,                  # the number of current running
        "iter": None,                 # the number of current iteration
        "samples": None,              # Training data, a tuple in the form of (samples_x, samples_y). If not provided, it will be generated by `config[samples_init]`, config["n"], config["d"] and `config["f"]`
        "samples_init": sampling.lhs, # offline data generation method
        "f": None,                    # fitness evaluation,
        "pop": None,                  # Initial population. If not provided, it will be generated by `config["pop_init"]`, `config["pop_size"]` and `config["d"]`
        "pop_size": 100,              # the size of population. If not provided, it will be got from `config["pop"]`
        "pop_init": sampling.lhs,     # method of population initialization
        "info_flag": True,            # whether output information like time costing,
    }
    
    config.update(kw)

    if (config["lower_bound"] is None) or (config["upper_bound"] is None):
        raise Exception("Undefined boundary")

    if config["samples"] is not None:
        x, y = config["samples"]
        # n of x, n of y
        data_n_x, data_n_y = len(x), len(y)
        # demension of x
        d_x = np.shape(x)[1]
    else:
        # x, y = None, None
        d_x = None
        data_n_x, data_n_y = None, None

    if config["pop"] is not None:
        pop_n, d_pop = config["pop"].shape
    else:
        pop_n, d_pop = None, None
    
    # check for inconsistencies of d
    d = config["d"] if config["d"] else None
    tmp = np.unique([x for x in [d, d_x, d_pop] if x])
    if tmp.size == 1:
        config["d"] = tmp[0]
    elif tmp.size == 0:
        raise Exception("No dimension provided")
    else:
        raise Exception("Inconsistency: dimension")

    # check for inconsistence of n
    n = config["n"] if config["n"] else None
    tmp = np.unique([x for x in [n, data_n_x, data_n_y] if x])
    if tmp.size == 1:
        config["n"] = tmp[0]
    elif tmp.size == 0:
        config["n"] = 11 * config["d"] # default 11 * d
    else:
        raise Exception("Inconsistency: number of offline data")

    # check for insistences of pop_size
    pop_size = config["pop_size"] if config["pop_size"] else None
    tmp = np.unique([x for x in [pop_size, pop_n] if x])
    if tmp.size == 1:
        config["pop_size"] = tmp[0]
    elif tmp.size == 0:
        raise Exception("No population size provided")
    else:
        raise Exception("Inconsistency: population size")

    if config["pm"] is None:
        config["pm"] = 1 / config["d"] # default 1 / d

    # generate offline data
    if config["samples"] is None:
        x = config["samples_init"](
            n=config["n"],
            d=config["d"],
            lower_bound=config["lower_bound"],
            upper_bound=config["upper_bound"]
        )
        y = config["f"](x)
        # add the samples to the config
        config["samples"] = (x, y)
    
    # initial population
    if config["pop"] is None:
        config["pop"] = config["pop_init"](
            n=config["pop_size"],
            d=config["d"],
            lower_bound=config["lower_bound"],
            upper_bound=config["upper_bound"]
        )

    return config
