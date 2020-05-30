import networkx as nwx
import numpy as np
import matplotlib.pyplot as plt
import time
from operator import itemgetter 
import math

# II. Graphes

'''Prend en paramètre le fichier txt du graphe, et le crée.'''
def creerGraphe(fichier):
    G = nwx.Graph()
    s = False
    a = False
    with open(fichier,"r") as f:
        for line in f:      
            if(line=="Sommets\n"):
                s = True
            if(line=="Nombre d aretes\n"):
                s = False
            if(s == True and line!="Sommets\n"):
                G.add_node(int(line))
            if(line=="Aretes\n"):
                a = True
            if(a == True and line!="Aretes\n"):
                lineSplit = line.split(' ')
                G.add_edge(int(lineSplit[0]),int(lineSplit[1]))
    f.close()
    return G

'''Prend en paramètre un graphe, et retourne une copie indépendante.'''
def copie(G):
    G1 = nwx.Graph()
    G1.add_nodes_from(list(G.nodes()))
    G1.add_edges_from(list(G.edges()))
    return G1

# 1) Opérations de base :

'''Méthode prenant en paramètres un graphe G et un sommet v de G,
et qui retourne un nouveau graphe G0 obtenu à partir de G
en supprimant le sommet v (et les arêtes incidentes). '''
def suppression(G,v):
    G.remove_node(v)
    return G

'''Généralisation de la méthode précédente lorsqu’un ensemble de sommets
est supprimé.''' 
def suppressions(G,v):
    G.remove_nodes_from(v)
    return G

'''Méthode prenant en entrée un graphe et renvoyant un tableau contenant
les degrés des sommets du graphe.'''
def degres(G):
    return list(nwx.degree(G))

'''Méthode permettant de déterminer un sommet de degré maximum.'''
def degreMax(G):
    return max(degres(G), key = itemgetter(1))[0]

'''Méthode permettant de déterminer le degré maximum du graphe.'''
def valeurDegreMax(G):
	dico = dict(degres(G))
	return dico[degreMax(G)]

# 2) Génération d'instances

'''Méthode prenant en entrée un entier n > 0 et un paramètre p ∈]0,1[,
et qui renvoie un graphe sur n sommets, et où chaque arête (i,j) 
est présente avec probabilité p.'''
def grapheAlea(n,p):
    if(n <= 0):
        raise ValueError ("n doit être strictement supérieur à 0")
    if(p<=0 or p>=1):
        raise ValueError ("p doit être compris entre 0 et 1 exclus")
    G = nwx.Graph()
    for i in range(n):
        G.add_node(i)
        for j in range(n):
            if(i<j):
                res = np.random.uniform(0,1)
                if res <= p :
                    G.add_edge(i,j)
    return G

'''Mesure le temps de calcul sur une classe d'instances.'''
def mesure_temps_grapheAlea(p,Nmax,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    y = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        start_time = time.time()
        g = grapheAlea(n,p)
        y.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, y)
    plt.show()
    #return x,y