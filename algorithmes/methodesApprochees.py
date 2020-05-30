import networkx as nwx
import numpy as np
import matplotlib.pyplot as plt
import time
from operator import itemgetter 
import math

import sys
sys.path.append('../')
from algorithmes import graphes as gr

# III. Méthodes approchées

# Méthode du couplage maximal

'''Renvoie les sommets d'un couplage.'''
def algo_couplage(G):
    C = []
    Cbis = [] # liste des arêtes de C
    extremites = list(G.edges())
    m = len(extremites)
    for i in range(m):
        # Si aucune des deux extrémités de e_i n'est dans C, alors :
        if extremites[i][0] not in C and extremites[i][1] not in C:
            # Ajouter les deux extrémités de e_i à C
            C.append(extremites[i][0])
            C.append(extremites[i][1])
            Cbis.append(extremites[i])
    return C

# Algorithme glouton

'''Renvoie une couverture de façon gloutonne.'''
def algo_glouton(G):
    Gbis = gr.copie(G) # copie du graphe afin de préserver l'original pour les autres commandes
    C = []
    E = list(Gbis.edges())
    while E != []: # Tant que E n'est pas vide
        v = gr.degreMax(Gbis) # Prendre un sommet v de degré maximum
        C.append(v) # Ajouter v à C
        # et supprimer de E les arêtes couvertes par v
        Gbis = gr.suppression(Gbis,v) # on supprime de G, v
        #nwx.draw(G)
        #plt.show()  # pour afficher ensuite
        E = list(Gbis.edges()) # on réinitalise E avec le nouveau graphe G avec v de supprimé
    return C


# Comparaisons en fonction de n et de p

'''Mesure le temps de calcul des algos au-dessus :'''

# en fonction de n
def mesure_temps_algoCouverture_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yC = []
    yG = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        algo_couplage(G)
        yC.append(time.time() - start_time)
        start_time = time.time()
        algo_glouton(G)
        yG.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yC, c='blue', label='algo_couplage')
    plt.scatter(x, yG, c='red', label='algo_glouton')
    plt.legend()
    plt.show()
    #return x,yC,yG

# en fonction de p
def mesure_temps_algoCouverture_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yC = []
    yG = []
    for i in range(nbInstances):
        p = (i+1)*1/(nbInstances)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        algo_couplage(G)
        yC.append(time.time() - start_time)
        start_time = time.time()
        algo_glouton(G)
        yG.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la probabilité p de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yC, c='blue', label='algo_couplage')
    plt.scatter(x, yG, c='red', label='algo_glouton')
    plt.legend()
    plt.show()
    #return x,yC,yG

'''Mesure la qualité de la solution retournée des algos au-dessus :'''

# en fonction de n
def mesure_qualite_algoCouverture_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yC = []
    yG = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        yC.append(len(algo_couplage(G)))
        yG.append(len(algo_glouton(G)))
    plt.title('Qualité de la solution retournée en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Nombre de sommets de la solution retournée')
    plt.scatter(x, yC, c='blue', label='algo_couplage')
    plt.scatter(x, yG, c='red', label='algo_glouton')
    plt.legend()
    plt.show()
    #return x,yC,yG

# en fonction de p
def mesure_qualite_algoCouverture_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yC = []
    yG = []
    for i in range(nbInstances):
        p = (i+1)*1/(nbInstances)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        yC.append(len(algo_couplage(G)))
        yG.append(len(algo_glouton(G)))
    plt.title('Qualité de la solution retournée en fonction de la probabilité p de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Nombre de sommets de la solution retournée')
    plt.scatter(x, yC, c='blue', label='algo_couplage')
    plt.scatter(x, yG, c='red', label='algo_glouton')
    plt.legend()
    plt.show()
    #return x,yC,yG

def antiGlouton():
    k = 6
    GAMDG = nwx.Graph()
    for i in range(1,21):
        GAMDG.add_node(i)
    for i in range(1,k+1):
        GAMDG.add_edge(21-i,15-i)
    GAMDG.add_edge(8,14)
    GAMDG.add_edge(8,13)
    GAMDG.add_edge(7,12)
    GAMDG.add_edge(7,11)
    GAMDG.add_edge(6,10)
    GAMDG.add_edge(6,9)
    GAMDG.add_edge(5,14)
    GAMDG.add_edge(5,13)
    GAMDG.add_edge(5,12)
    GAMDG.add_edge(4,11)
    GAMDG.add_edge(4,10)
    GAMDG.add_edge(4,9)
    GAMDG.add_edge(3,14)
    GAMDG.add_edge(3,13)
    GAMDG.add_edge(3,12)
    GAMDG.add_edge(3,11)
    GAMDG.add_edge(2,14)
    GAMDG.add_edge(2,13)
    GAMDG.add_edge(2,12)
    GAMDG.add_edge(2,11)
    GAMDG.add_edge(2,10)
    GAMDG.add_edge(1,14)
    GAMDG.add_edge(1,13)
    GAMDG.add_edge(1,12)
    GAMDG.add_edge(1,11)
    GAMDG.add_edge(1,10)
    GAMDG.add_edge(1,9)
    return GAMDG