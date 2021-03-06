from experiments.experiments import Experiment

def simple_experiment(dataset):
    import numpy as np
    exp = Experiment(dataset)
    reduction = np.random.choice([9], replace=False)
    exp.simple_experiment(reduction_models = reduction, attack="CW", drop_rate=0.07, tau="minRE", length=2000, T=5, metric="JSD")
    
def choose_team_each_jump_experiment(dataset):
    exp = Experiment(dataset)
    exp.choose_team_each_jump_experiment(jump=100, magnet=False, attack="BIM", drop_rate=0.1, T=15, metric="JSD", 
        tau="minRE", length=2000)

def all_cases_experiment(dataset):
    exp = Experiment(dataset)
    #attacks = ["FGSM", "BIM", "DEEPFOOL", "CW_0.0"]
    attacks = ["CW"]
    drop_rate = [0.001]
    reduction_models = [1,3,5,7,9]
    tau = ["RE", "minRE"]
    T = [5]
    metric = ["RE"]

    exp.all_cases_experiment([5], reduction_models, attacks, drop_rate, tau, T, metric)

# simple_experiment("CIFAR")
#all_cases_experiment("CIFAR")
# choose_team_each_jump_experiment("CIFAR")

exp = Experiment("CIFAR")

drop_rate = [0.1, 0.05, 0.07]
reduction_models = [7]
tau = ["minRE"]
T = [15, 10, 5, 1]
metric = ["JSD"]

exp.tuning_team_parameters("BIM", reduction_models, drop_rate, tau, metric, T)



