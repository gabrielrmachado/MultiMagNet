from experiments.experiments import Experiment

def simple_experiment(dataset):
    import numpy as np
    exp = Experiment(dataset)
    reduction = np.random.choice([1,3,5,7,9], replace=False)
    exp.simple_experiment(reduction_models = reduction, attack="DEEPFOOL", drop_rate=0.07, tau="minRE", length=2000, T=1, metric="JSD")
    
def choose_team_each_jump_experiment(dataset):
    exp = Experiment(dataset)
    exp.choose_team_each_jump_experiment(jump=1000, magnet=False, attack="DEEPFOOL", drop_rate=0.07, T=5, metric="JSD", 
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

simple_experiment("CIFAR")
#all_cases_experiment("CIFAR")
# choose_team_each_jump_experiment("CIFAR")

# exp = Experiment("MNIST")
# exp.create_adversarial_validation_images()



