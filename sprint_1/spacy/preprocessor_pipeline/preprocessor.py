import spacy
from nltk.tokenize import word_tokenize
from spacy import displacy

class TextPreprocessor:
    def __init__(self):
        # loading english language model of spacy
        self.nlp = spacy.load("en_core_web_sm")
        
    def lower_casing(self, text):
        """
        Accepts text as arguments and return text in lowercase

        Arguments:
        text: raw text

        Returns:
        text_to_lower: text converted to lower case
        """
        text_to_lower = text.lower()

        return text_to_lower

    def remove_stopwords(self, text):
        """
        Removes stopwords passed from the text passed as an arguments

        Arguments:
        text: raw text from where stopwords need to removed

        Returns:
        tokens_without_sw: concatenated tokens of raw text without stopwords
        """
        # getting list of default stop words in spaCy english model
        stopwords = self.nlp.Defaults.stop_words

        # tokenize text
        text_tokens = word_tokenize(text)

        # remove stop words:
        tokens_without_sw = " ".join([word for word in text_tokens if word not in stopwords])

        # return list of tokens with no stop words
        return tokens_without_sw
    
    def tokenize_word(self, text):
        """
        Tokenize the text passed as an arguments into a list of words(tokens)

        Arguments:
        text: raw text

        Returns:
        words: list containing tokens in text
        """
        # passing the text to nlp and initialize an object called 'doc'
        doc = self.nlp(text)

        # Tokenize the doc using token.text attribute
        words = [token.text for token in doc]

        # return list of tokens
        return words
    
    def tokenize_sentence(self, text):
        """
        Tokenize the text passed as an arguments into a list of sentence

        Arguments:
        text: raw text

        Returns:
        sentences: list of sentences
        """
        # passing the text to nlp and initialize an object called 'doc'
        doc = self.nlp(text)

        # tokenize the sentence using sents attributes
        sentences = list(doc.sents)

        # return tokenize sentence
        return sentences
    
    def remove_punctuation(self, text):
        """
        removes punctuation symbols present in the raw text passed as an arguments

        Arguments:
        text: raw text

        Returns: 
        not_punctuation: text without punctuation
        """
        # passing the text to nlp and initialize an object called 'doc'
        doc = self.nlp(text)

        not_punctuation = []
        # remove the puctuation
        for token in doc:
            if token.is_punct == False:
                not_punctuation.append(token)
        
        return " ".join([str(w) for w in not_punctuation])
    
    
    def lemmatization(self, text):
        """
        obtain the lemma of the each token in the text, append to the list, and returns the list

        Arguments:
        text: raw text

        Returns:
        token_lemma_list: list containing token with its lemma
        """

        # passing the text to nlp and initialize an object called 'doc'
        doc = self.nlp(text)

        token_lemma_list = []
        # Lemmatization
        for token in doc:
            token_lemma_list.append((token.text, token.lemma_))

        return token_lemma_list
    
    def pos_tagging(self, text):
        # passing the text to nlp and initialize an object called 'doc'
        doc = self.nlp(text)

        pos_list = []
        for token in doc:
            pos_list.append((token.text, token.pos_, token.tag_))
        return pos_list
    
    def named_entity_recognition(self, text):
        """
        returns entity_text and entity labels as a tuple

        Arguments:
        text: raw text

        Returns:
        entity_text_label: entity text and labels as a tuple
        """
        # passing the text to nlp and initialize an object called 'doc'
        doc = self.nlp(text)

        #named entity recogniton using doc.ents
        entity_text_label = []

        for entity in doc.ents:
            entity_text_label.append((entity.text, entity.label_))

        return entity_text_label

def main():
    sample_text = "Books, are on the table. I want to read a book ? "
    preprocessor = TextPreprocessor()
    lower_text = preprocessor.lower_casing(sample_text)
    print(lower_text)
    remove_stopword = preprocessor.remove_stopwords(lower_text)
    print(remove_stopword)
    token = preprocessor.tokenize_word(remove_stopword)
    print(token)
    sample_text2 =  "Oh man, this is pretty cool. We will do more such things."
    sentence = preprocessor.tokenize_sentence(sample_text2)
    print(sentence)
    not_punctuation = preprocessor.remove_punctuation(remove_stopword)
    print(not_punctuation)
    lemma = preprocessor.lemmatization(not_punctuation)
    print(lemma)

if __name__ == "__main__":
    main()