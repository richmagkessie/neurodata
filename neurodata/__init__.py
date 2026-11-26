from .trainer import NeuroDataTrainer
from .frame import NeuroFrame
from .fidelity import evaluate, print_report


def fit(df, path, epochs=500, lr=1e-3, verbose=True,
        epsilon=None, delta=1e-5, password=None):
    """
    Train a model on df and save it to path.

    epsilon  : differential privacy budget (None = no DP)
    password : encrypt the .model file (None = no encryption)
    """
    trainer = NeuroDataTrainer(df, epochs=epochs, lr=lr, epsilon=epsilon, delta=delta)
    trainer.train(verbose=verbose)
    trainer.save(path, password=password)


def load(path, password=None) -> NeuroFrame:
    """Load a saved .model file and return a NeuroFrame."""
    return NeuroFrame(path, password=password)


def fidelity(original_df, neuro_frame):
    """Compare a NeuroFrame against the original DataFrame and print a report."""
    report = evaluate(original_df, neuro_frame.to_dataframe())
    print_report(report)
    return report
