# MCTS Multi-Agent Training Package

try:
    from .dual_net import DualNetwork
    from .mcts import MCTS
    from .train import MCTSTrainer
except ImportError:
    from dual_net import DualNetwork
    from mcts import MCTS
    from train import MCTSTrainer

__all__ = ['MCTS', 'DualNetwork', 'MCTSTrainer']
