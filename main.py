from experiments.experiments import Experiment

def simple_experiment(dataset):
    exp = Experiment(dataset)
    exp.simple_experiment(attack="DEEPFOOL", drop_rate=0.07, tau="minRE", length=2000, T=1, metric="JSD")
    
def choose_team_each_jump_experiment(dataset):
    exp = Experiment(dataset)
    exp.choose_team_each_jump_experiment(jump=10, magnet=False, attack="CW_10.0", drop_rate=0.01, 
        tau="RE", length=2000)

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

#simple_experiment("CIFAR")
#all_cases_experiment("CIFAR")
choose_team_each_jump_experiment("MNIST")

