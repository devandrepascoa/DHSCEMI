import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import ContainerManager, AVAILABLE_CONFIGS


def set_load(metrics, load):
    def _get_tokens_per_hour(self, model_name):
        return load
    metrics.get_tokens_per_hour = types.MethodType(_get_tokens_per_hour, metrics)


def test_scale_up_and_down_sequence(monkeypatch):
    manager = ContainerManager()
    manager.workload_metrics.scaling_cooldown = 0
    model = "test-model"

    async def spawn_stub(model_name, model_path, config):
        class Dummy:
            def __init__(self, cfg):
                self.config = cfg
                self._is_ready = True
                self.container_name = f"dummy-{str(cfg)}"
                self.active_requests = 0
                self.queue_start_times = {}
                self.last_scale_evaluation = 0.0

            async def estimate_processing_time(self, tokens=100):
                return 1.0

            async def get_load_score(self, tokens=100):
                return 1.0

            async def stop(self):
                pass

        instance = Dummy(config)
        manager.container_pools.setdefault(model_name, []).append(instance)
        return instance

    monkeypatch.setattr(manager, "spawn_container", spawn_stub)

    manager.container_pools[model] = []
    current = manager.workload_metrics.get_current_container_config(model)
    manager.workload_metrics.update_container_config(model, current)

    for idx in range(len(AVAILABLE_CONFIGS) - 1):
        config = manager.workload_metrics.get_current_container_config(model)
        assert str(config) == str(AVAILABLE_CONFIGS[idx])
        capacity = manager.workload_metrics.get_config_capacity_tokens_per_hour(config)
        set_load(manager.workload_metrics, capacity * 0.9)
        manager.workload_metrics.select_optimal_config(model)
        new_cfg = manager.workload_metrics.get_current_container_config(model)
        assert str(new_cfg) == str(AVAILABLE_CONFIGS[idx + 1])

    for idx in reversed(range(1, len(AVAILABLE_CONFIGS))):
        config = manager.workload_metrics.get_current_container_config(model)
        assert str(config) == str(AVAILABLE_CONFIGS[idx])
        capacity = manager.workload_metrics.get_config_capacity_tokens_per_hour(config)
        set_load(manager.workload_metrics, capacity * 0.1)
        manager.workload_metrics.select_optimal_config(model)
        new_cfg = manager.workload_metrics.get_current_container_config(model)
        assert str(new_cfg) == str(AVAILABLE_CONFIGS[idx - 1])
