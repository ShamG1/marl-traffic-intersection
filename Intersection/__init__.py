# Intersection package
# This makes Intersection a proper Python package

try:
    from .env import IntersectionEnv
    from .config import DEFAULT_REWARD_CONFIG
    from .agent import Car
    from .sensor import Lidar
except ImportError:
    from env import IntersectionEnv
    from config import DEFAULT_REWARD_CONFIG
    from agent import Car
    from sensor import Lidar

__all__ = ['IntersectionEnv', 'DEFAULT_REWARD_CONFIG', 'Car', 'Lidar']
__version__ = '0.1.0'
