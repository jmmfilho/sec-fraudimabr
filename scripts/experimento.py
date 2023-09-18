import time
import os
import pandas as pd
import pickle

# features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# cross_validation
from sklearn.model_selection import cross_validate

# data balancing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Utils
from utils import *

# Multicore
from multiprocessing import Process
from queue import Queue

data_dir = '../data'
preprocessed = False

# verificar se já houve processamento anterior
for filename in os.listdir(data_dir):
    if filename == 'processed_texts_telegram.p':
        preprocessed = True


def experimento_modelos(df):
    while not queue.empty():

        experiment = queue.get()

        # removendo duplicatas

        df = df.drop_duplicates(subset=['text_content'])
        texts = df['text_content']
        y = df['Golpe']

        # verifica se experimento ja esta com o texto pre-processado
        if 'processed' in experiment:
            # caso o texto ja esteja processado
            if preprocessed:
                pro_texts = pickle.load(open("../data/processed_texts_telegram.p", "rb"))
            else:
                pro_texts = [preprocess(t) for t in texts]
                pickle.dump(pro_texts, open("../data/processed_texts_telegram.p", "wb"))
        else:
            # apenas usa o diminutivo e separa emoji e pontuação
            pro_texts = [processEmojisPunctuation(t.lower(), remove_punct=False) for t in texts]

        # vetorizacao
        max_feat = 500
        vectorizer = None

        if 'tfidf' in experiment:

            if 'max_features' in experiment:
                vectorizer = TfidfVectorizer(max_features=max_feat)
            elif 'bigram' in experiment:
                vectorizer = TfidfVectorizer(ngram_range=(2, 2))
            elif 'trigram' in experiment:
                vectorizer = TfidfVectorizer(ngram_range=(3, 3))
            elif 'unibi_gram' in experiment:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            elif 'unibitri_gram' in experiment:
                vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            elif 'unibitriquad_gram' in experiment:
                vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            else:
                vectorizer = TfidfVectorizer()

        elif 'bow' in experiment:
            if 'max_features' in experiment:
                vectorizer = CountVectorizer(max_features=max_feat, binary=True)
            elif 'bigram' in experiment:
                vectorizer = CountVectorizer(binary=True, ngram_range=(2, 2))
            elif 'trigram' in experiment:
                vectorizer = CountVectorizer(binary=True, ngram_range=(3, 3))
            elif 'unibi_gram' in experiment:
                vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
            elif 'unibitri_gram' in experiment:
                vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))
            else:
                vectorizer = CountVectorizer(binary=True)

        vectorizer.fit(pro_texts)
        X = vectorizer.transform(pro_texts)

        if 'smote' in experiment:
            # oversampling with SMOTE
            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)
        elif 'undersampling' in experiment:
            rus = RandomUnderSampler(random_state=42)
            X, y = rus.fit_resample(X, y)
        elif 'random_oversampling' in experiment:
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)

        models = [LogisticRegression(), BernoulliNB(), MultinomialNB(), LinearSVC(dual=False),
                  KNeighborsClassifier(), SGDClassifier(), RandomForestClassifier(),
                  GradientBoostingClassifier(n_estimators=200),
                  MLPClassifier(max_iter=6, verbose=True, early_stopping=True)]

        resultados = []

        for model in models:
            resultado_linha = dict()
            resultado_linha['model'] = model.__class__.__name__
            metodos_scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

            cv_results = cross_validate(model, X, y, cv=5, scoring=metodos_scoring, return_train_score=True)

            for scoring in metodos_scoring:
                scores = cv_results[f'test_{scoring}']

                for i, score in enumerate(scores, start=1):
                    resultado_linha[f'{scoring}_{i}'] = score

                resultado_linha[f'{scoring}_avg'] = scores.mean()
                resultado_linha[f'{scoring}_std'] = scores.std()

            resultados.append(resultado_linha)

        df_metrics = pd.DataFrame(resultados)

        path_to_save = '../results_telegram/' + experiment + '.csv'

        df_metrics.to_csv(path_to_save, index=False)


# definir quais experimentos serão feitos
experiments = ['ml-tfidf-unibitri_gram-random_oversampling',
               'ml-tfidf-unibi_gram-random_oversampling',
               'ml-tfidf-unibitri_gram-processed-random_oversampling',
               'ml-bow-unibitri_gram-random_oversampling',
               'ml-bow-processed-random_oversampling',
               'ml-tfidf-random_oversampling',
               'ml-tfidf-processed-random_oversampling',
               'ml-bow-unibitri_gram-processed-random_oversampling',
               'ml-bow-unibi_gram-random_oversampling',
               'ml-bow-random_oversampling',
               'ml-tfidf-unibi_gram-processed-random_oversampling',
               'ml-bow-unibi_gram-processed-random_oversampling']

# usando fila para colocar as diferentes situacoes do experimento
queue = Queue()
for experiment_queue in experiments:
    queue.put(experiment_queue)


if __name__ == '__main__':

    filepath = '../data/dataset_rotulado_telegram.csv'
    # carregar os dados
    dataset = pd.read_csv(filepath)

    start_time = time.time()
    processos = []
    for i in range(3):
        print('Core Iniciado ', i)
        proc = Process(target=experimento_modelos, args=(dataset,))
        proc.start()
        processos.append(proc)

    for t in processos:
        t.join()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    print(f"Tempo total de execução: {elapsed_time:.2f} minutos")