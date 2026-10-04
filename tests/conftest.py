"""Independent multi-bean contact fixtures, outside the formal single-bean scope."""

import pytest

from feedingrobot.sim import model
from feedingrobot.sim.task import FeedingTask


@pytest.fixture(params=['panda', 'ur5e'])
def two_bean_task(request, monkeypatch):
    original = model.load_json
    def fixture_config(path):
        config = original(path)
        if path == 'configs/scene.json':
            config['beans']['count'] = 2
        return config
    monkeypatch.setattr(model, 'load_json', fixture_config)
    return FeedingTask(request.param)
