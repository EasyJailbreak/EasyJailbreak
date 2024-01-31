from easyjailbreak.mutation import MutationBase
import random
import nltk
from nltk.corpus import wordnet

class ReplaceWordsWithSynonyms(MutationBase):
    r"""
    The 'ReplaceWordsWithSynonyms' class is designed to replace words in a sentence 
    with synonyms from the maintained momentum word dictionary. This mutation is used 
    in the AutoDAN-HGA algorithm.
    """
    def __init__(self, attr_name='jailbreak_prompt', word_dict={}):
        r"""
        Initializes the ReplaceWordsWithSynonyms instance.
        :param ~str attr_name: The attribute name in the instance that contains the text to be mutated.
        :param ~dict word_dict: The maintained momentum word dictionary
        """
        self.attr_name = attr_name
        self.word_dict = word_dict

    def update(self, word_dict=None):
        r"""
        Update the momentum word dictionary
        """
        self.word_dict = word_dict
    
    def _get_mutated_instance(self, instance, **kwargs):
        r"""
        Generates a mutated instance of the given object.
        :param instance: The instance to be mutated.
        :param kwargs: Additional keyword arguments.
        :return: A list containing the mutated instance.
        """
        seed = getattr(instance, self.attr_name)
        mutated_prompt = self.HGA_replace_words_with_synonyms(self.word_dict, seed)

        new_instance = instance.copy()
        setattr(new_instance, self.attr_name, mutated_prompt)
        return [new_instance]

    def HGA_replace_words_with_synonyms(self, word_dict, sentence):
        r"""
        Iterate over each word in sentence and searche for synonymous terms within
        the maintained word_dict that contains words and their associated scores. 
        If a synonym is found, a probabilistic decision based on the word's score 
        (compared to the total score of all synonyms) determines if the original 
        word in the sentence should be replaced by this synonym.
        """
        words = nltk.word_tokenize(sentence)
        for word in words:
            synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
            synonyms_scores = []
            syn_in_word_dict = []
            for s in synonyms:
                if s in word_dict.keys():
                    syn_in_word_dict.append(s)
                    synonyms_scores.append(word_dict[s])
            for s in syn_in_word_dict:
                if random.random() < word_dict[s] / sum(synonyms_scores):
                    sentence = sentence.replace(word, s, 1)
                    break
        return sentence
