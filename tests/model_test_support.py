"""Scripted raw responses through the shared generation runner, never a real transport."""

from types import SimpleNamespace

from taim.generation import run_generation
from taim.models import Model


class ScriptedModel(Model):
    def __init__(self):
        pass

    cache = None
    reuse_unresolved = True
    model_id = "synthetic"
    reasoning_effort = None

    @property
    def configuration(self):
        return {
            "provider_id": self.provider_id,
            "model": self.model_id,
            "reasoning_effort": self.reasoning_effort,
        }

    def generate(self, request):
        transport = SimpleNamespace(provider_id=self.provider_id, generate=self.respond)
        return run_generation(
            request,
            transport,
            provider_config=self.configuration,
            cache=self.cache,
            reuse_unresolved=self.reuse_unresolved,
        )
