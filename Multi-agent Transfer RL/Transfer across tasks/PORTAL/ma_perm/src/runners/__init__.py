REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_debug import ParallelRunner as ParallelRunnerDebug
REGISTRY["parallel_debug"] = ParallelRunnerDebug
