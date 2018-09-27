from experiments.experiments import Experiment

def simple_experiment(dataset):
    exp = Experiment(dataset)
    exp.simple_experiment(reduction_models=3, attack="CW_0.0", drop_rate=0.01, tau="minRE", length=2000)
    
def choose_team_each_jump_experiment(dataset):
    exp = Experiment(dataset)
    exp.choose_team_each_jump_experiment(n_experiments=5, reduction_models=3, attack="CW_10.0", drop_rate=0.01, 
        tau="RE", jump=50, length=2000)

def all_cases_experiment(dataset):
    exp = Experiment(dataset)
    attacks = ["CW_40.0"]
    # attacks = ["FGSM", "BIM", "DEEPFOOL", "CW_0.0", "CW_10.0", "CW_20.0", "CW_30.0", "CW_40.0"]
    drop_rate = [0.001, 0.01, 0.05]
    reduction_models = [1, 3, 5, 7, 9]
    tau = ["RE", "minRE"]

    exp.all_cases_experiment([1], reduction_models, attacks, drop_rate, tau)

# executes chosen experiment
#all_cases_experiment("CIFAR")

exp = Experiment("CIFAR")
exp.testJSD(10, 1, "BIM", logits=True, T=10)
exp.simple_experiment(reduction_models=5, attack="DEEPFOOL", drop_rate=0.1, tau="minRE", length=2000)