from experiments.experiments import Experiment

def simple_experiment(dataset):
    exp = Experiment(dataset)
    exp.simple_experiment(reduction_models=1, attack="FGSM", drop_rate=0.1, tau="minRE", length=2000, T=15)
    
def choose_team_each_jump_experiment(dataset):
    exp = Experiment(dataset)
    exp.choose_team_each_jump_experiment(n_experiments=5, reduction_models=3, attack="CW_10.0", drop_rate=0.01, 
        tau="RE", jump=50, length=2000)

def all_cases_experiment(dataset):
    exp = Experiment(dataset)
    # attacks = ["FGSM", "BIM", "DEEPFOOL", "CW_0.0"]
    attacks = ["FGSM"]
    drop_rate = [0.1]
    reduction_models = [1,3,5,7,9]
    tau = ["minRE"]
    T = [15]

    exp.all_cases_experiment([3], reduction_models, attacks, drop_rate, tau, T)

simple_experiment("CIFAR")

## executes chosen experiment
# all_cases_experiment("CIFAR")

