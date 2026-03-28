import re
from .openai_model import OpenaiModel


class MiniMaxModel(OpenaiModel):
    """
    MiniMax model wrapper using the OpenAI-compatible API.

    MiniMax provides large language models (MiniMax-M2.7, MiniMax-M2.7-highspeed,
    etc.) accessible through an OpenAI-compatible chat completions endpoint at
    ``https://api.minimax.io/v1``.

    Since ``MiniMaxModel`` inherits from :class:`OpenaiModel`, it is recognised
    by every ``isinstance(model, OpenaiModel)`` check throughout the framework
    (e.g. PAIR, TAP, IntrospectGeneration) without any modifications.

    Example
    -------
    >>> from easyjailbreak.models.minimax_model import MiniMaxModel
    >>> model = MiniMaxModel(api_keys='YOUR_MINIMAX_API_KEY')
    >>> response = model.generate('Hello, how are you?')
    """

    MINIMAX_BASE_URL = 'https://api.minimax.io/v1'
    DEFAULT_MODEL = 'MiniMax-M2.7'

    # Regex to strip <think>…</think> reasoning blocks that MiniMax M2.7 may
    # emit when chain-of-thought is enabled.
    _THINK_TAG_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

    def __init__(
        self,
        api_keys: str,
        model_name: str = DEFAULT_MODEL,
        generation_config=None,
    ):
        """
        Initializes the MiniMax model.

        :param str api_keys: MiniMax API key (``MINIMAX_API_KEY``).
        :param str model_name: Model identifier, defaults to ``'MiniMax-M2.7'``.
            Other options include ``'MiniMax-M2.7-highspeed'``,
            ``'MiniMax-M2.5'``, ``'MiniMax-M2.5-highspeed'``.
        :param dict generation_config: Extra generation parameters forwarded to
            the chat completions API.  Temperature values are automatically
            clamped to the MiniMax-supported range ``(0, 1]``.
        """
        if generation_config is not None:
            generation_config = self._clamp_temperature(dict(generation_config))
        super().__init__(
            model_name=model_name,
            api_keys=api_keys,
            generation_config=generation_config,
            base_url=self.MINIMAX_BASE_URL,
        )

    # ------------------------------------------------------------------
    # Temperature clamping
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_temperature(config: dict) -> dict:
        """Clamp ``temperature`` to the MiniMax-supported range ``(0, 1]``."""
        if 'temperature' in config:
            t = config['temperature']
            if t is not None:
                config['temperature'] = max(0.01, min(float(t), 1.0))
        return config

    # ------------------------------------------------------------------
    # Think-tag stripping
    # ------------------------------------------------------------------

    @classmethod
    def _strip_think_tags(cls, text: str | None) -> str | None:
        """Remove ``<think>…</think>`` blocks from model output."""
        if text is None:
            return None
        return cls._THINK_TAG_RE.sub('', text).strip()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def generate(self, messages, clear_old_history=True, **kwargs):
        kwargs = self._clamp_temperature(dict(kwargs))
        response = super().generate(messages, clear_old_history=clear_old_history, **kwargs)
        return self._strip_think_tags(response)
