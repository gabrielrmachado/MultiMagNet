from experiments.experiments import Experiment

def mnist_simple_experiment():
    exp = Experiment("MNIST")
    exp.simple_experiment(reduction_models=3, attack="CW_0.0", drop_rate=0.01, tau="minRE", length=2000)
    
def mnist_choose_team_each_jump_experiment():
    exp = Experiment("MNIST")
    exp.choose_team_each_jump_experiment(n_experiments=5, reduction_models=3, attack="CW_10.0", drop_rate=0.01, 
        tau="RE", jump=50, length=2000)

def mnist_all_cases_experiment():
    exp = Experiment("MNIST")
    attacks = ["FGSM", "BIM", "DEEPFOOL", "CW_0.0", "CW_10.0", "CW_20.0", "CW_30.0", "CW_40.0"]
    drop_rate = [0.001, 0.01, 0.05]
    reduction_models = [1, 3, 5, 7, 9]
    tau = ["RE", "minRE"]

    exp.all_cases_experiment([10], reduction_models, attacks, drop_rate, tau)

def cifar_simple_experiment():
    exp = Experiment("CIFAR")
    exp.simple_experiment(reduction_models=5, attack="FGSM", drop_rate=0.01, tau="minRE", length=2000)

# executes chosen experiment
cifar_simple_experiment()