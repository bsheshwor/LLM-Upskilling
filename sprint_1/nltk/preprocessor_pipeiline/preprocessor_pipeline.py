import re
from collections import Counter
from abc import ABC, abstractmethod
from typing import Union, List
import string

#importing nltk library
import nltk

#importing nltk libraries for stopwords and wordnet
from nltk.corpus import stopwords, wordnet

#importing nltk libraries for lemmatization and stemming
from nltk.stem import WordNetLemmatizer, PorterStemmer

#importing nltk libraries for word tokenization
from nltk.tokenize import word_tokenize

#importing nltk libraries for sentence tokenization
from nltk.tokenize import sent_tokenize

#importing textblob for spelling correction
from textblob import Word

class BasePreprocessor(ABC):
    @abstractmethod
    def preprocessor(self, text: str) -> Union[str, List[str]]:
        """
        Abstract method for text preprocessing.

        Args:
            text (str): Input text to be preprocessed.

        Returns:
            Union[str, List[str]]: Processed text or list of processed tokens.
        """
        pass

class DownloadRequirement:
    def download(self):
        """
        Downloads NLTK resources for text preprocessing.
        """
        download()
        download('stopwords')
        download('wordnet')

class LowerCase(BasePreprocessor):
    def preprocessor(self, text: str) -> str:
        """
        Converts the input text to lowercase.

        Args:
            text (str): Input text.

        Returns:
            str: Lowercased text.
        """
        return text.lower()

class RemoveTag(BasePreprocessor):
    def preprocessor(self, text: str) -> str:
        """
        Removes HTML tags from the input text.

        Args:
            text (str): Input text containing HTML tags.

        Returns:
            str: Text with HTML tags removed.
        """
        regex = re.compile(r'<[^>]+>')
        return regex.sub('', text)

class RemoveURL(BasePreprocessor):
    def preprocessor(self, text: str) -> str:
        """
        Removes URLs from the input text.

        Args:
            text (str): Input text containing URLs.

        Returns:
            str: Text with URLs removed.
        """
        url_search = re.search('http://\S+|https://\S+', text)
        if url_search:
            url_group = url_search.group(0)
            return text.replace(url_group, '')
        else:
            return text

class TokenizeWord(BasePreprocessor):
    def preprocessor(self, text: str) -> List[str]:
        """
        Tokenizes the input text into words.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of word tokens.
        """
        return word_tokenize(text)

class TokenizeSentence(BasePreprocessor):
    def preprocessor(self, text: str) -> List[str]:
        """
        Tokenizes the input text into sentences.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of sentence tokens.
        """
        return sent_tokenize(text)

class RemoveStopword(BasePreprocessor):
    def preprocessor(self, text: List[str]) -> List[str]:
        """
        Removes stopwords from a list of word tokens.

        Args:
            text (List[str]): List of word tokens.

        Returns:
            List[str]: List of word tokens with stopwords removed.
        """
        words = [w for w in text if w not in stopwords.words("english")]
        return words

class RemovePunctuation(BasePreprocessor):
    def preprocessor(self, text: str) -> str:
        """
        Removes punctuation from the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with punctuation removed.
        """
        pun_string = text.translate(str.maketrans('', '', string.punctuation))
        return pun_string

class RemoveFrequentWord(BasePreprocessor):
    def preprocessor(self, text: List[str]) -> List[str]:
        """
        Removes the most frequent words from a list of word tokens.

        Args:
            text (List[str]): List of word tokens.

        Returns:
            List[str]: List of word tokens with frequent words removed.
        """
        cnt = Counter()
        for word in text:
            cnt[word] += 1
        FREQWORD = set([w for (w, wc) in cnt.most_common(10)])
        return [word for word in text if word not in FREQWORD]

class RemoveRareWord(BasePreprocessor):
    def preprocessor(self, text: List[str]) -> List[str]:
        """
        Removes the rarest words from a list of word tokens.

        Args:
            text (List[str]): List of word tokens.

        Returns:
            List[str]: List of word tokens with rare words removed.
        """
        cnt = Counter()
        for word in text:
            cnt[word] += 1
        n_rare_words = 10
        RAREWORD = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
        return [word for word in text if word not in RAREWORD]

class CorrectSpelling(BasePreprocessor):
    def preprocessor(self, text: List[str]) -> List[str]:
        """
        Corrects the spelling of words in a list of word tokens.

        Args:
            text (List[str]): List of word tokens.

        Returns:
            List[str]: List of word tokens with corrected spelling.
        """
        temp = []
        for word in text:
            word = Word(word)
            result = word.correct()
            temp.append(result)
        return temp

class Lemmatizer(BasePreprocessor):
    def preprocessor(self, text: List[str]) -> str:
        """
        Lemmatizes a list of word tokens.

        Args:
            text (List[str]): List of word tokens.

        Returns:
            str: Text with lemmatized words.
        """
        lemmed = " ".join([WordNetLemmatizer().lemmatize(w) for w in text])
        return lemmed

class Stemmer(BasePreprocessor):
    def preprocessor(self, text: List[str]) -> str:
        """
        Stems a list of word tokens.

        Args:
            text (List[str]): List of word tokens.

        Returns:
            str: Text with stemmed words.
        """
        stemmed = " ".join([PorterStemmer().stem(w) for w in text])
        return stemmed
    
class TextPreprocessor:
    def __init__(self, text: str):
        """
        Initializes the TextPreprocessor instance.

        Args:
            text (str): The input text to be preprocessed.
        """
        self.text = text
        self.download_requirement = DownloadRequirement()
        self.download_requirement.download()
        self.lower_text: str = ""
        self.removed_tags: str = ""  # Variable name changed from 'self.removed_tags' to 'self.removed_tags'
        self.removed_urls: str = ""  # Variable name changed from 'self.removed_urls' to 'self.removed_urls'
        self.tokenized_words: List[str] = []  # Variable name changed from 'self.tokenized_words' to 'self.tokenized_words'
        self.tokenized: List[str] = []  # Variable name changed from 'self.tokenized' to 'self.tokenized'
        self.removed_punctuation: str = ""  # Variable name changed from 'self.removed_punctuation' to 'self.removed_punctuation'
        self.removed_frequent: List[str] = []
        self.removed_rare: List[str] = []
        self.spell_corrected: List[str] = []  # Variable name changed from 'self.spell_corrected' to 'self.spell_corrected'
        self.lemma: str = ""
        self.stem: str = ""
    
    def lower_case(self) -> str:
        """
        Converts the input text to lowercase.

        Returns:
            str: The input text in lowercase.
        """
        text_processor = LowerCase()
        self.lower_text = text_processor.preprocessor(self.text)
        return self.lower_text
    
    def remove_tag(self) -> str:
        """
        Removes HTML tags from the text.

        Returns:
            str: Text with HTML tags removed.
        """
        text_processor = RemoveTag()
        self.removed_tags = text_processor.preprocessor(self.lower_text)
        return self.removed_tags

    def remove_url(self) -> str:
        """
        Removes URLs from the text.

        Returns:
            str: Text with URLs removed.
        """
        text_processor = RemoveURL()
        self.removed_urls = text_processor.preprocessor(self.removed_tags)
        return self.removed_urls
    
    def tokenize_word(self) -> List[str]:
        """
        Tokenizes the text into words.

        Returns:
            List[str]: List of word tokens.
        """
        text_processor = TokenizeWord()
        self.tokenized_words = text_processor.preprocessor(self.removed_urls)
        return self.tokenized_words
    
    def remove_punctuation(self) -> str:
        """
        Removes punctuation from the text.

        Returns:
            str: Text with punctuation removed.
        """
        self.tokenized_words = " ".join(self.tokenized_words)
        text_processor = RemovePunctuation()
        self.removed_punctuation = text_processor.preprocessor(self.tokenized_words)
        return self.removed_punctuation
    
    def remove_frequent_word(self) -> List[str]:
        """
        Removes the most frequent words from the text.

        Returns:
            List[str]: List of words with frequent words removed.
        """
        text_processor = TokenizeWord()
        self.tokenized = text_processor.preprocessor(self.removed_punctuation)
        text_processor = RemoveFrequentWord()
        self.removed_frequent = text_processor.preprocessor(self.tokenized)
        return self.removed_frequent
    
    def remove_rare_word(self) -> List[str]:
        """
        Removes the rarest words from the text.

        Returns:
            List[str]: List of words with rare words removed.
        """
        text_processor = RemoveRareWord()
        self.removed_rare = text_processor.preprocessor(self.tokenized)
        return self.removed_rare
    
    def correct_spelling(self) -> List[str]:
        """
        Corrects the spelling of words in the text.

        Returns:
            List[str]: List of words with corrected spelling.
        """
        text_processor = CorrectSpelling()
        self.spell_corrected = text_processor.preprocessor(self.removed_frequent)
        return self.spell_corrected
    
    def lemmatizer(self) -> str:
        """
        Lemmatizes the text.

        Returns:
            str: Text with lemmatized words.
        """
        text_processor = Lemmatizer()
        self.lemma = text_processor.preprocessor(self.spell_corrected)
        return self.lemma
    
    def stemmer(self) -> str:
        """
        Stems the text.

        Returns:
            str: Text with stemmed words.
        """
        text_processor = Stemmer()
        self.stem = text_processor.preprocessor(self.spell_corrected)
        return self.stem

def main():
    text = "<html> @bshesh Dr. Smith graduated from the University of Washington. He later started an analytics firm called Lux, which catered to enterprise customers. https://www.bisheshwor.com.np"
    text_preprocessor = TextPreprocessor(text)
    print(text_preprocessor.lower_case())
    print(text_preprocessor.remove_tag())
    print(text_preprocessor.remove_url())
    print(text_preprocessor.tokenize_word())
    print(text_preprocessor.remove_punctuation())
    print(text_preprocessor.remove_frequent_word())
    print(text_preprocessor.remove_rare_word())
    print(text_preprocessor.correct_spelling())
    print(text_preprocessor.lemmatizer())
    print(text_preprocessor.stemmer())

if __name__ == '__main__':
    main()