"""Sandbox infrastructure exception types."""


class SandboxInfrastructureError(RuntimeError):
    """A sandbox operation could not be completed because infrastructure failed."""


class SandboxTransportError(SandboxInfrastructureError):
    """A logical sandbox request exhausted all transport retries."""
