"""Exception hierarchy. Everything the agent can raise derives from AgentError."""


class AgentError(Exception):
    """Base class for all agent errors."""


class LLMError(AgentError):
    """The language model could not be reached or returned something unusable."""


class CodeValidationError(AgentError):
    """Generated code was rejected by the AST validator (before execution)."""


class SandboxError(AgentError):
    """Base class for errors raised while executing code in the sandbox."""


class ExecutionError(SandboxError):
    """Generated code raised an exception while running."""


class SandboxTimeout(SandboxError):
    """Generated code exceeded the time limit."""


class OutputPolicyError(SandboxError):
    """The result violated the aggregated-only output policy."""
