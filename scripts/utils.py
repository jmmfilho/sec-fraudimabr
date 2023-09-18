"""
Adicionar a referência ao lucas
"""

import emoji,string,re
import spacy
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, roc_auc_score

unicode_emoji = {}
for key, value in emoji.EMOJI_DATA.items():
    try:
        unicode_emoji[key] = value['pt']
    except:
        pass

# emojis and punctuation
emojis_list = list(unicode_emoji)
punct = list(string.punctuation)
emojis_punct = emojis_list + punct


def processEmojisPunctuation(text, remove_punct = True):
    """
    Put spaces between emojis. Removes punctuation.
    """
    # get all unique chars
    chars = set(text)
    # for each unique char in text, do:
    for c in chars:
        # remove punctuation
        if remove_punct:
            if c in emojis_list:
                text = text.replace(c, ' ' + c + ' ')
            if c in punct:
                text = text.replace(c, ' ')

        # put spaces between punctuation
        else:
            if c in emojis_punct:
                text = text.replace(c, ' ' + c + ' ')          

    text = text.replace('  ', ' ')
    return text

# stop words removal
stop_words = list(stopwords.words('portuguese'))
new_stopwords = ['aí','pra','vão','vou','onde','lá','aqui',
                 'tá','pode','pois','so','deu','agora','todo',
                 'nao','ja','vc', 'bom', 'ai','kkk','kkkk','ta', 'voce', 'alguem', 'ne', 'pq',
                 'cara','to','mim','la','vcs','tbm', 'tudo']
stop_words = stop_words + new_stopwords
final_stop_words = []
for sw in stop_words:
    sw = ' '+ sw + ' '
    final_stop_words.append(sw)


def removeStopwords(text):
    for sw in final_stop_words:
        text = text.replace(sw,' ')
    text = text.replace('  ',' ')
    return text


# lemmatization
nlp = spacy.load('pt_core_news_sm')


def lemmatization(text):
    doc = nlp(text)
    for token in doc:
        if token.text != token.lemma_:
            text = text.replace(token.text, token.lemma_)
    return text


def domainUrl(text):
    '''
    Substitutes an URL in a text for the domain of this URL
    Input: an string
    Output: the string with the modified URL
    '''    
    if 'http' in text:
        re_url = '[^\s]*https*://[^\s]*'
        matches = re.findall(re_url, text, flags=re.IGNORECASE)
        for m in matches:
            domain = m.split('//')
            domain = domain[1].split('/')[0]
            text = re.sub(re_url, domain, text, 1)
        return text
    else:
        return text 


def preprocess(text):
    text = text.lower().strip()
    text = domainUrl(text)
    text = processEmojisPunctuation(text)
    text = removeStopwords(text)
    text = lemmatization(text)
    return text


def getTestMetrics(y_test, y_pred, y_prob,full_metrics=True, print_charts=False):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    precision_neg = precision_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred)
    recall_neg = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred)
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    return (acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc)