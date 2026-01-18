# MCTS Multi-Agent Training Package

try:
    from .networks import DualNetwork
    from .mcts import MCTS, MCTSNode
    from .train import MCTSTrainer
except ImportError:
    from networks import DualNetwork
    from mcts import MCTS, MCTSNode
    from train import MCTSTrainer

__all__ = ['MCTS', 'MCTSNode', 'DualNetwork', 'MCTSTrainer']
