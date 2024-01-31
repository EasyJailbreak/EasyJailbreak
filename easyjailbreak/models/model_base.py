"""
Defines base classes for black-box and white-box model interface standards.
The purpose is to unify models from various sources.
There should be no specific algorithm implementations here.
"""
class ModelBase:
    """
    Defines a common model interface.
    This base class is intended to provide a standardized interface for different types of models.
    """
    def generate(self, *args, **kwargs) -> str:
        """
        Generates output based on input arguments. This method must be implemented by subclasses.
        :return str: The generated output.
        """
        raise NotImplementedError

class WhiteBoxModelBase(ModelBase):
    """
    Defines the interface that white-box models should possess. Any user-defined white-box model should inherit from this class.
    These models could be Hugging Face models or custom models written in PyTorch/TensorFlow, etc.
    To maintain consistency with black-box models, this class integrates a tokenizer.
    """
    def __init__(self, model, tokenizer):
        """
        Initializes the white-box model with a model and a tokenizer.
        :param model: The underlying model for generation.
        :param tokenizer: The tokenizer used for processing input and output.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def instance2str(self, instance, *args, **kwargs):
        """
        Converts an instance to a string. This method must be implemented by subclasses.
        :param instance: The instance to be converted.
        :return: A string representation of the instance.
        """
        raise NotImplementedError

    @property
    def device(self):
        """
        Returns the device on which the model is running.
        :return: The device used by the model.
        """
        raise NotImplementedError

    @property
    def embed_layer(self):
        """
        Provides access to the embedding layer of the model.
        :return: The embedding layer of the model.
        """
        raise NotImplementedError

    @property
    def vocab_size(self):
        """
        Returns the vocabulary size of the model.
        :return: The size of the model's vocabulary.
        """
        raise NotImplementedError

    @property
    def bos_token_id(self):
        """
        Returns the Beginning-Of-Sequence token ID.
        :return: The BOS token ID.
        """
        raise NotImplementedError

    @property
    def eos_token_id(self):
        """
        Returns the End-Of-Sequence token ID.
        :return: The EOS token ID.
        """
        raise NotImplementedError

    @property
    def pad_token_id(self):
        """
        Returns the padding token ID.
        :return: The padding token ID.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Used to get logits, loss, and perform backpropagation, etc. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def batch_encode(self, *args, **kwargs):
        """
        Encodes a batch of inputs. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def batch_decode(self, *args, **kwargs):
        """
        Decodes a batch of outputs. This method must be implemented by subclasses.
        """
        raise NotImplementedError

class BlackBoxModelBase(ModelBase):
    """
    Defines the interface that black-box models should possess. Any user-defined black-box model should inherit from this class.
    These models could be like OpenAI's API or based on HTTP request services from third parties or self-built APIs.
    """
    def batch_generate(self, *args, **kwargs):
        """
        Uses asynchronous requests or multithreading to efficiently obtain batch responses. This method must be implemented by subclasses.
        """
        raise NotImplementedError
