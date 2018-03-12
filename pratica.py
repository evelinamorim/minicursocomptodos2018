# esse eh o algoritmo que usaremos para diminuir a dimensao 
#dos nossos vetores de 200 para 2
from sklearn.manifold import TSNE
# Matplotlib para fazer os graficos
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

#  seaborn deixa o grafico mais bunitin
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


from gensim.models import KeyedVectors

import pandas as pd
import numpy as np


# modelo treinado em ambiente com locale "en_US.UTF-8"

model_file = 'vectors_tweets.bin'
word_vectors = KeyedVectors.load_word2vec_format(model_file,binary=True,unicode_errors='ignore')


data_file = 'grad_students_tweets.csv'
df = pd.read_csv(data_file)

# nao temos classe nenhuma, entao vamos definir todo mundo como classe 0
#y = np.zeros(df.shape[0])
# limitei em 200 tweets, para mudar e fazer o grafico completo, comente esta linha e descomente a linha de cima
# e comente as linhas 56 e 57
y = np.zeros(200) 

# transformando meus tweets em embeddings através de soma
dim_vector = 200
X = []
for idx, r in df.iterrows():
    tweet = r['text'].replace('\'','')
    words = tweet.lower().split()

    v = np.zeros(dim_vector)
    for w in words:

        # se seu ambiente for outro locale, talvez voce deve pesquisar sua palavra no modelo como:
        # word_vectors[w.encode('utf-8')]
        try:
            v = np.add(v, word_vectors[w])
        except KeyError: # por enquanto vamos ignorar palavras nao existentes em nosso vocabulario
            pass
    X.append(v)
    # limitei o tamanho da quantidade de tweets no grafico, para desenhar todos, comente es
    if len(X) == 200:
        break

def scatter(x, colors):
    # funcao levemente adaptada do tutorial: https://github.com/oreillymedia/t-SNE-tutorial

    # We choose a color palette with seaborn.
    # palette = np.array(sns.color_palette("hls", 21))
    palette = np.array(sns.color_palette("hls", 1))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    i = 0
    # while i <= 10:
    while i <= 3:
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)   
        i = i + 1
    ax.spines['left']._adjust_location()
    i = i + 1
    return f, ax, sc, txts

# semente para o algoritmo aleatorio
RS = 20180308
digits_proj = TSNE(random_state=RS, perplexity=100).fit_transform(X)
scatter(digits_proj, y)

plt.savefig('tsne-tweets.png', dpi=120)

