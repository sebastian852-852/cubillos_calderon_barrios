# -*- coding: utf-8 -*-
"""generador

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rcl5nGGu-La41tXLTyogyMgrrly_Mww3
"""

import os
import networkx as nx
import glob
import json
import shutil, bz2, getopt, sys
from collections import defaultdict
from datetime import datetime
import graphlib
import timeit


# Genera un grafo dirigido de retweets a partir de una lista de tweets
def gen_rt_graph(tweets):
    # Crea un grafo dirigido (DiGraph) usando la biblioteca NetworkX
    G = nx.DiGraph()
    for tweet in tweets:
        if 'retweeted_status' in tweet:
            # Obtiene los usuarios involucrados en el retweet
            rt_user = tweet['retweeted_status']['user']['screen_name']
            rtg_user = tweet['user']['screen_name']
            # Agrega una arista al grafo desde el retweeted_user al retweeting_user
            G.add_edge(rt_user, rtg_user)
    return G

# Crea un archivo JSON que contiene información sobre retweets a partir de una lista de tweets
def create_rt_json(tweets):
    rts = {}
    for tweet in tweets:
        if 'retweeted_status' in tweet:
            rt_user = tweet['retweeted_status']['user']['screen_name']
            rtg_user = tweet['user']['screen_name']
            tweet_id = tweet['id']
            if rt_user not in rts:
                # Inicializa la entrada del usuario en el diccionario si no existe
                rts[rt_user] = {'receivedRetweets': 1, 'tweets': [{'id': tweet_id, 'retweeted_by': rtg_user}]}
            else:
                # Actualiza la información si el usuario ya existe en el diccionario
                rts[rt_user]['receivedRetweets'] += 1
                rts[rt_user]['tweets'].append({'id': tweet_id, 'retweeted_by': rtg_user})
    # Guarda el diccionario como un archivo JSON
    with open('rt.json', 'w') as f:
        json.dump(rts, f)

# Genera un grafo no dirigido de retweets mutuos a partir de una lista de tweets
def gen_co_rt_graph(tweets):
    G = nx.Graph()
    rts = defaultdict(list)
    for tweet in tweets:
        if 'retweeted_status' in tweet:
            rt_user = tweet['retweeted_status']['user']['screen_name']
            rtg_user = tweet['user']['screen_name']
            # Agrega el retweeting_user a la lista de retweets para el retweeted_user
            rts[rt_user].append(rtg_user)
    # Agrega aristas al grafo para usuarios que han retuiteado el mismo tweet
    for users in rts.values():
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                G.add_edge(users[i], users[j])
    return G

# Crea un archivo JSON que contiene información sobre retweets mutuos a partir de una lista de tweets
def create_core_rt_json(tweets):
    core_rts = {}
    for tweet in tweets:
        if 'retweeted_status' in tweet:
            rt_user = tweet['retweeted_status']['user']['screen_name']
            rtg_user = tweet['user']['screen_name']
            if rt_user not in core_rts:
                # Inicializa la entrada del usuario en el diccionario si no existe
                core_rts[rt_user] = {'totalCoretweets': 1, 'retweeters': [rtg_user]}
            else:
                # Actualiza la información si el usuario ya existe en el diccionario
                core_rts[rt_user]['totalCoretweets'] += 1
                core_rts[rt_user]['retweeters'].append(rtg_user)
    # Filtra usuarios con menos de 3 retweets mutuos y guarda el diccionario como un archivo JSON
    core_rts = {user: data for user, data in core_rts.items() if len(data['retweeters']) > 2}
    with open('corrtw.json', 'w') as f:
        json.dump(core_rts, f)

# Genera un grafo dirigido de menciones a partir de una lista de tweets
def gen_mention_graph(tweets):
    G = nx.DiGraph()
    for tweet in tweets:
        if 'entities' in tweet and 'user_mentions' in tweet['entities']:
            tweeting_user = tweet['user']['screen_name']
            for mention in tweet['entities']['user_mentions']:
                mentioned_user = mention['screen_name']
                # Agrega una arista al grafo desde el tweeting_user al mentioned_user
                G.add_edge(tweeting_user, mentioned_user)
    return G

# Crea un archivo JSON que contiene información sobre menciones a partir de una lista de tweets
def create_mention_json(tweets):
    mentions = {}
    for tweet in tweets:
        if 'entities' in tweet and 'user_mentions' in tweet['entities']:
            mentioning_user = tweet['user']['screen_name']
            tweet_id = tweet['id']
            for user_mention in tweet['entities']['user_mentions']:
                mentioned_user = user_mention['screen_name']
                if mentioned_user not in mentions:
                    # Inicializa la entrada del usuario en el diccionario si no existe
                    mentions[mentioned_user] = {'receivedMentions': 1, 'mentions': [{'id': tweet_id, 'mentioned_by': mentioning_user}]}
                else:
                    # Actualiza la información si el usuario ya existe en el diccionario
                    mentions[mentioned_user]['receivedMentions'] += 1
                    mentions[mentioned_user]['mentions'].append({'id': tweet_id, 'mentioned_by': mentioning_user})
    # Guarda el diccionario como un archivo JSON
    with open('mencion.json', 'w') as f:
        json.dump(mentions, f)

# Procesa los tweets de un directorio dado, generando varios grafos y archivos JSON
def process_tweets(input_directory, start_date, end_date, hashtags):
    tweets = []
    # Recorre los archivos en el directorio y agrega los tweets a la lista
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.bz2'):
                with bz2.BZ2File(os.path.join(root, file), 'rb') as f:
                    for line in f:
                        tweet = json.loads(line)
                        tweets.append(tweet)

    # Genera los grafos y los JSON
    G_rt = gen_rt_graph(tweets)
    nx.write_gexf(G_rt, 'retweet_graph.gexf')
    G_crt = gen_co_rt_graph(tweets)
    nx.write_gexf(G_crt, 'co_retweet_graph.gexf')
    G_m = gen_mention_graph(tweets)
    nx.write_gexf(G_m, 'mention_graph.gexf')

    create_rt_json(tweets)
    create_mention_json(tweets)
    create_core_rt_json(tweets)


start_time = timeit.default_timer()

# Función principal que maneja los argumentos de la línea de comandos
def main(argv):
    input = '/content/33.json.bz2'
    fecha_inicial = None
    fecha_final = None
    hashtags = None

    # Procesa los tweets con los valores predeterminados o proporcionados en la línea de comandos
    process_tweets(input, fecha_inicial, fecha_final, hashtags)
    try:
        opts, _ = getopt.getopt(argv, "d:fi:ff:h:grt:jrt:gm:jm:gcrt:jcrt:")
    except getopt.GetoptError:
        print("generador.py -d <path relativo> -fi <fecha inicial> -ff <fecha final> -h <nombre de archivo> -grt -jrt -gm -jm -gcrt -jcrt")
        sys.exit(2)

    options = {
        '-d': lambda arg: arg,
        '-fi': lambda arg: datetime.strptime(arg, '%d-%m-%y'),
        '-ff': lambda arg: datetime.strptime(arg, '%d-%m-%y'),
        '-h': lambda arg: arg,
        '-grt': lambda arg: True,
        '-jrt': lambda arg: True,
        '-gm': lambda arg: True,
        '-jm': lambda arg: True,
        '-gcrt': lambda arg: True,
        '-jcrt': lambda arg: True
    }

    for opt, arg in opts:
        if opt in options:
            result = options[opt](arg)
            if opt in ['-d', '-fi', '-ff', '-h']:
                globals()[opt[1:]] = result
            else:
                globals()[f'gen_{opt[2:]}'] = result


if __name__ == "__main__":
    main(sys.argv[1:])

end_time = timeit.default_timer()
print(f"Tiempo total de ejecución: {end_time - start_time} segundos")