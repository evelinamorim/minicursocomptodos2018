# minicursocomptodos2018

Este foi um tutorial dado por mim no dia 09/03/2018 para o evento Computação para tod@as.

## Descrição dos Dados 


**Sentiment-Analysis-Dataset.zip**: Este dataset é constituido de tweets e seus sentimentos. O arquivo 
pode ser baixado em http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/.

**tweets_clean.txt** : Dados para o treino dos embeddings do Twitter. Ele é baseado no dataset 
Sentiment-Analysis-Dataset.zip. 

Para construir o conjunto de dados de tweets de pós-graduandos, foi feito um crawler no perfil 
do twitter do PHDComics, e então cerca de 400 usuários foram extraídos dos 10000 usuários, de acordo 
com as seguintes palavras chaves no campo *description*: 'phd','p.h.d.','ph.d.','phd.','graduate',
'postdoctoral','msc','ms.c','ms.c.'. 

**users_graduate.json**: Este arquivo armazena só os dados dos cerca de 400 usuários extraídos de 
acordo com a metodologia descrita acima.

**user_data.zip**: Este dado possui os tweets por usuário. 

**grad_students_tweets.zip**: Esse arquivo contém quatro campos, a saber: 

> numero_da_linha,
> data_tweet,
> id_tweet,
> texto_tweet,
> nome_do_usuario 

## Apresentação

O arquivo pdf contém a apresentação completa dada no tutorial.

## Prática

A prática consiste em plotar um gráfico de *embeddings* de tweets utilizando o algoritmo TSNE. Ela 
está codificada em python no arquivo pratica.py. Para utilizar, basta ter um modelo word2vec treinado 
em tweets_clean.txt e o arquivo grad_students_tweets.zip descompactado.
