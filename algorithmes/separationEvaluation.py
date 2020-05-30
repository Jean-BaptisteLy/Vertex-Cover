import networkx as nwx
import numpy as np
import matplotlib.pyplot as plt
import time
from operator import itemgetter 
import math
import random

import sys
sys.path.append('../')
from algorithmes import graphes as gr
from algorithmes import methodesApprochees as ma

# IV. Séparation et évaluation

'''Chaque fonction renvoie un tuple :
    la solution optimale
    et le nombre de noeuds visités.'''

# 1) Branchement

def branchement(G,C=[]):
    cpt = 0 # compteur de noeuds visités
    E = list(G.edges()) # ensemble des arêtes
    if E == []:
        return C,cpt
    couvertures = []
    e = E[0] # prend une arête e
    for i in range(2): # soit u est dans la couverture, soit v
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i]) # graphe où l’on a supprimé le sommet 
        c,cptemp = branchement(Gbis,C_temp)
        couvertures.append(c)
        cpt += cptemp + 1
    if len(couvertures[0]) <= len(couvertures[1]):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

'''Mesure le temps de calcul des algos au-dessus :'''

# en fonction de n
def mesure_temps_branchement_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    y = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchement(G)
        y.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, y)
    plt.show()
    #return x,y

# en fonction de p
def mesure_temps_branchement_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    y = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchement(G)
        y.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la probabilité p de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, y)
    plt.show()
    #return x,y

'''Mesure le nombre de noeuds visités des algos au-dessus :'''

# en fonction de n
def mesure_nbreNoeuds_branchement_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    y = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        sol,cpt = branchement(G)
        y.append(cpt)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, y)
    plt.show()
    #return x,y

# en fonction de p
def mesure_nbreNoeuds_branchement_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    y = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        sol,cpt = branchement(G)
        y.append(cpt)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, y)
    plt.show()
    #return x,y

# 2) Ajout de bornes

def bornesCouplage(G,C=[]):
    cpt = 0
    E = list(G.edges()) # ensemble des arêtes
    couvertures = []
    e = E[0] # arete e
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0: # cran d'arrêt
            '''si en enlevant le sommet, il n'y a plus d'arrêt
            alors c'est une feuille'''
            couvertures.append(C_temp)
            cpt += 1
        else:
            # calcul des bornes
            ac = ma.algo_couplage(Gbis)
            n = len(list(Gbis.nodes()))
            m = len(list(Gbis.edges()))
            b1 = m / gr.valeurDegreMax(Gbis)
            b2 = len(ac) / 2
            b3 = ( 2*n-1-math.sqrt(((2*n-1)**2)-(8*m)) ) / 2
            borneInf = max([b1,b2,b3])
            borneSup = len(ac)
            if borneInf < borneSup: # on continue
                c,cptemp = bornesCouplage(Gbis,C_temp)
                couvertures.append(c)
                cpt += cptemp + 1
            else: # sinon on élague
                C_temp = C_temp + ac
                couvertures.append(C_temp)
                cpt += 1
    # sélection de la couverture minimale à retourner
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

def bornesGlouton(G,C=[]):
    cpt = 0
    E = list(G.edges())
    couvertures = []
    e = E[0]
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0:
            couvertures.append(C_temp)
            cpt += 1
        else:
            ac = ma.algo_couplage(Gbis)
            ag = ma.algo_glouton(Gbis)
            n = len(list(Gbis.nodes()))
            m = len(list(Gbis.edges()))
            b1 = m / gr.valeurDegreMax(Gbis)
            b2 = len(ac) / 2
            b3 = ( 2*n-1-math.sqrt(((2*n-1)**2)-(8*m)) ) / 2
            borneInf = max([b1,b2,b3])
            borneSup = len(ag)
            if borneInf < borneSup:
                c,cptemp = bornesGlouton(Gbis,C_temp)
                couvertures.append(c)
                cpt += cptemp + 1
            else:
                C_temp = C_temp + ag
                couvertures.append(C_temp)
                cpt += 1
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

'''Mesure le temps de calcul des algos au-dessus :'''

# en fonction de n
def mesure_temps_bornes_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yB = []
    yC = []
    yG = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchement(G)
        yB.append(time.time() - start_time)
        start_time = time.time()
        bornesCouplage(G)
        yC.append(time.time() - start_time)
        start_time = time.time()
        bornesGlouton(G)
        yG.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,yB,yC,yG

# en fonction de p
def mesure_temps_bornes_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yB = []
    yC = []
    yG = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchement(G)
        yB.append(time.time() - start_time)
        start_time = time.time()
        bornesCouplage(G)
        yC.append(time.time() - start_time)
        start_time = time.time()
        bornesGlouton(G)
        yG.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la probabilité p de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,y

'''Mesure le nombre de noeuds visités des algos au-dessus :'''

# en fonction de n
def mesure_nbreNoeuds_bornes_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yB = []
    yC = []
    yG = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        sol,cptB = branchement(G)
        sol,cptC = bornesCouplage(G)
        sol,cptG = bornesGlouton(G)
        yB.append(cptB)
        yC.append(cptC)
        yG.append(cptG)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,y

# en fonction de p
def mesure_nbreNoeuds_bornes_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yB = []
    yC = []
    yG = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        sol,cptB = branchement(G)
        sol,cptC = bornesCouplage(G)
        sol,cptG = bornesGlouton(G)
        yB.append(cptB)
        yC.append(cptC)
        yG.append(cptG)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,y

'''Efficacité des méthodes si l’on n’utilise que les bornes inférieures,
 ou que le calcul de solution réalisable (avec une borne inférieure triviale).
'''

def bornesInf(G,C=[],borneSup=None):
    cpt = 0
    E = list(G.edges()) # ensemble des arêtes
    couvertures = []
    e = E[0] # arete e
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0: # cran d'arrêt
            '''si en enlevant le sommet, il n'y a plus d'arrêt
            alors c'est une feuille'''
            couvertures.append(C_temp)
            cpt += 1
            if borneSup != None and borneSup > len(C_temp):
                borneSup = len(C_temp)
        else:
            # calcul des bornes
            ac = ma.algo_couplage(Gbis)
            n = len(list(Gbis.nodes()))
            m = len(list(Gbis.edges()))
            b1 = m / gr.valeurDegreMax(Gbis)
            b2 = len(ac) / 2
            b3 = ( 2*n-1-math.sqrt(((2*n-1)**2)-(8*m)) ) / 2
            borneInf = max([b1,b2,b3])
            if borneSup == None or borneInf < borneSup: # on continue
                c,cptemp = bornesInf(Gbis,C_temp,borneSup)
                couvertures.append(c)
                cpt += cptemp + 1
                if borneSup != None and borneSup > len(c):
                    borneSup = len(c)
            else: # sinon on élague
                C_temp = C_temp + ac
                couvertures.append(C_temp)
                cpt += 1
                if borneSup != None and borneSup > len(C_temp):
                    borneSup = len(C_temp)
    # sélection de la couverture minimale à retourner
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

def bornesCouplageTriviale(G,C=[]):
    cpt = 0
    E = list(G.edges()) # ensemble des arêtes
    couvertures = []
    e = E[0] # arete e
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0: # cran d'arrêt
            '''si en enlevant le sommet, il n'y a plus d'arrêt
            alors c'est une feuille'''
            couvertures.append(C_temp)
            cpt += 1
        else:
            # calcul des bornes
            ac = ma.algo_couplage(Gbis)
            borneInf = len(ac) / 2
            borneSup = len(ac)
            if borneInf < borneSup: # on continue
                c,cptemp = bornesCouplage(Gbis,C_temp)
                couvertures.append(c)
                cpt += cptemp + 1
            else: # sinon on élague
                C_temp = C_temp + ac
                couvertures.append(C_temp)
                cpt += 1
    # sélection de la couverture minimale à retourner
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

def bornesGloutonTriviale(G,C=[]):
    cpt = 0
    E = list(G.edges())
    couvertures = []
    e = E[0]
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0:
            couvertures.append(C_temp)
            cpt += 1
        else:
            ac = ma.algo_couplage(Gbis)
            ag = ma.algo_glouton(Gbis)
            borneInf = len(ac) / 2
            borneSup = len(ag)
            if borneInf < borneSup:
                c,cptemp = bornesGlouton(Gbis,C_temp)
                couvertures.append(c)
                cpt += cptemp + 1
            else:
                C_temp = C_temp + ag
                couvertures.append(C_temp)
                cpt += 1
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

'''Mesure le temps de calcul des algos au-dessus :'''

# en fonction de n
def mesure_temps_bornes3_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yB = []
    yC = []
    yG = []
    yI = []
    yCT = []
    yGT = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchement(G)
        yB.append(time.time() - start_time)
        start_time = time.time()
        bornesCouplage(G)
        yC.append(time.time() - start_time)
        start_time = time.time()
        bornesGlouton(G)
        yG.append(time.time() - start_time)
        start_time = time.time()
        bornesInf(G)
        yI.append(time.time() - start_time)
        start_time = time.time()
        bornesCouplageTriviale(G)
        yCT.append(time.time() - start_time)
        start_time = time.time()
        bornesGloutonTriviale(G)
        yGT.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.scatter(x, yI, c='black', label='bornesInf')
    plt.scatter(x, yCT, c='cyan', label='couplage-trivial')
    plt.scatter(x, yGT, c='magenta', label='glouton-trivial')
    plt.legend()
    plt.show()
    #return x,yB,yC,yG,yI,yCT,yGT

# en fonction de p
def mesure_temps_bornes3_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yB = []
    yC = []
    yG = []
    yI = []
    yCT = []
    yGT = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchement(G)
        yB.append(time.time() - start_time)
        start_time = time.time()
        bornesCouplage(G)
        yC.append(time.time() - start_time)
        start_time = time.time()
        bornesGlouton(G)
        yG.append(time.time() - start_time)
        start_time = time.time()
        bornesInf(G)
        yI.append(time.time() - start_time)
        start_time = time.time()
        bornesCouplageTriviale(G)
        yCT.append(time.time() - start_time)
        start_time = time.time()
        bornesGloutonTriviale(G)
        yGT.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la probabilité p de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.scatter(x, yI, c='black', label='bornesInf')
    plt.scatter(x, yCT, c='cyan', label='couplage-trivial')
    plt.scatter(x, yGT, c='magenta', label='glouton-trivial')
    plt.legend()
    plt.show()
    #return x,yB,yC,yG,yI,yCT,yGT

'''Mesure le nombre de noeuds visités des algos au-dessus :'''

# en fonction de n
def mesure_nbreNoeuds_bornes3_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yB = []
    yC = []
    yG = []
    yI = []
    yCT = []
    yGT = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        sol,cptB = branchement(G)
        sol,cptC = bornesCouplage(G)
        sol,cptG = bornesGlouton(G)
        sol,cptI = bornesInf(G)
        sol,cptCT = bornesCouplageTriviale(G)
        sol,cptGT = bornesGloutonTriviale(G)
        yB.append(cptB)
        yC.append(cptC)
        yG.append(cptG)
        yI.append(cptI)
        yCT.append(cptCT)
        yGT.append(cptGT)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.scatter(x, yI, c='black', label='bornesInf')
    plt.scatter(x, yCT, c='cyan', label='couplage-trivial')
    plt.scatter(x, yGT, c='magenta', label='glouton-trivial')
    plt.legend()
    plt.show()
    #return x,yB,yC,yG,yI,yCT,yGT

# en fonction de p
def mesure_nbreNoeuds_bornes3_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yB = []
    yC = []
    yG = []
    yI = []
    yCT = []
    yGT = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        sol,cptB = branchement(G)
        sol,cptC = bornesCouplage(G)
        sol,cptG = bornesGlouton(G)
        sol,cptI = bornesInf(G)
        sol,cptCT = bornesCouplageTriviale(G)
        sol,cptGT = bornesGloutonTriviale(G)
        yB.append(cptB)
        yC.append(cptC)
        yG.append(cptG)
        yI.append(cptI)
        yCT.append(cptCT)
        yGT.append(cptGT)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yB, c='green', label='branchement')
    plt.scatter(x, yC, c='blue', label='couplage')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.scatter(x, yI, c='black', label='bornesInf')
    plt.scatter(x, yCT, c='cyan', label='couplage-trivial')
    plt.scatter(x, yGT, c='magenta', label='glouton-trivial')
    plt.legend()
    plt.show()
    #return x,yB,yC,yG,yI,yCT,yGT

# 3) Amélioration du branchement

# u est le sommet à ne pas prendre en compte

# 1.
def branchementAmelioreBornesGlouton(G,C=[],u=None):
    cpt = 0
    E = list(G.edges())
    G_copie = gr.copie(G)
    if u in G_copie.nodes():
        gr.suppression(G_copie,u)
    E = list(G_copie.edges())
    if E == []:
        C.append(u)
        return C,cpt
    couvertures = []
    e = E[0]
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0:
            couvertures.append(C_temp)
            cpt += 1
        else:
            ac = ma.algo_couplage(Gbis)
            ag = ma.algo_glouton(Gbis)
            n = len(list(Gbis.nodes()))
            m = len(list(Gbis.edges()))
            b1 = m / gr.valeurDegreMax(Gbis)
            b2 = len(ac) / 2
            b3 = ( 2*n-1-math.sqrt(((2*n-1)**2)-(8*m)) ) / 2
            borneInf = max([b1,b2,b3])
            borneSup = len(ag)
            if borneInf < borneSup:
                if i == 0: # première branche : pas de sommet u à ignorer
                    u = None
                    c,cptemp = branchementAmelioreBornesGlouton(Gbis,C_temp,u)
                elif i == 1: # deuxième branche : sommet u à ignorer
                    u = e[0]
                    c,cptemp = branchementAmelioreBornesGlouton(Gbis,C_temp,u)
                couvertures.append(c)
                cpt += cptemp + 1
            else:
                C_temp = C_temp + ag
                couvertures.append(C_temp)
                cpt += 1
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

'''Mesure le temps de calcul des algos au-dessus :'''

# en fonction de n
def mesure_temps_bornesAmeliore_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yG = []
    yGA = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        bornesGlouton(G)
        yG.append(time.time() - start_time)
        start_time = time.time()
        branchementAmelioreBornesGlouton(G)
        yGA.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yGA, c='blue', label='gloutonAmeliore')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,yG,yGA

# en fonction de p
def mesure_temps_bornesAmeliore_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yG = []
    yGA = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        bornesGlouton(G)
        yG.append(time.time() - start_time)
        start_time = time.time()
        branchementAmelioreBornesGlouton(G)
        yGA.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yGA, c='blue', label='gloutonAmeliore')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,yG,yGA

'''Mesure le nombre de noeuds visités des algos au-dessus :'''

# en fonction de n
def mesure_nbreNoeuds_bornesAmeliore_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yG = []
    yGA = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        sol,cptG = bornesGlouton(G)
        sol,cptGA = branchementAmelioreBornesGlouton(G)
        yG.append(cptG)
        yGA.append(cptGA)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yGA, c='blue', label='gloutonAmeliore')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,yG,yGA

# en fonction de p
def mesure_nbreNoeuds_bornesAmeliore_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yG = []
    yGA = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        sol,cptG = bornesGlouton(G)
        sol,cptGA = branchementAmelioreBornesGlouton(G)
        yG.append(cptG)
        yGA.append(cptGA)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yGA, c='blue', label='gloutonAmeliore')
    plt.scatter(x, yG, c='red', label='glouton')
    plt.legend()
    plt.show()
    #return x,yG,yGA

# 2.
def branchementAmeliorePlusBornesGlouton(G,C=[],u=None):
    cpt = 0
    E = list(G.edges())
    G_copie = gr.copie(G)
    if u in G_copie.nodes():
        gr.suppression(G_copie,u)
    E = list(G_copie.edges())
    if E == []:
        C.append(u)
        return C,cpt
    couvertures = []
    e = E[0]
    dico = dict(gr.degres(G))
    e_liste = list(e)
    if dico[e[0]] < dico[e[1]]:
        e_liste[0] = e[1]
        e_liste[1] = e[0]
    e = tuple(e_liste)
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0:
            couvertures.append(C_temp)
            cpt += 1
        else:
            ac = ma.algo_couplage(Gbis)
            ag = ma.algo_glouton(Gbis)
            n = len(list(Gbis.nodes()))
            m = len(list(Gbis.edges()))
            b1 = m / gr.valeurDegreMax(Gbis)
            b2 = len(ac) / 2
            b3 = ( 2*n-1-math.sqrt(((2*n-1)**2)-(8*m)) ) / 2
            borneInf = max([b1,b2,b3])
            borneSup = len(ag)
            if borneInf < borneSup:
                if i == 0:
                    u = None
                    c,cptemp = branchementAmeliorePlusBornesGlouton(Gbis,C_temp,u)
                elif i == 1:
                    u = e[0]
                    c,cptemp = branchementAmeliorePlusBornesGlouton(Gbis,C_temp,u)
                couvertures.append(c)
                cpt += cptemp + 1
            else:
                C_temp = C_temp + ag
                couvertures.append(C_temp)
                cpt += 1
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

# 3.

def branchementAmeliorePlusPlusBornesGlouton(G,C=[],u=None):
    cpt = 0
    E = list(G.edges())
    G_copie = gr.copie(G)
    if u in G_copie.nodes():
        gr.suppression(G_copie,u)
    E = list(G_copie.edges())
    if E == []:
        C.append(u)
        return C,cpt
    couvertures = []
    e = E[0]
    dico = dict(gr.degres(G))
    e_liste = list(e)
    if dico[e[0]] < dico[e[1]]:
        e_liste[0] = e[1]
        e_liste[1] = e[0]
    e = tuple(e_liste)
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0:
            couvertures.append(C_temp)
            cpt += 1
        elif dico[e[i]] == 1:
            cpt += 1
        else:
            ac = ma.algo_couplage(Gbis)
            ag = ma.algo_glouton(Gbis)
            n = len(list(Gbis.nodes()))
            m = len(list(Gbis.edges()))
            b1 = m / gr.valeurDegreMax(Gbis)
            b2 = len(ac) / 2
            b3 = ( 2*n-1-math.sqrt(((2*n-1)**2)-(8*m)) ) / 2
            borneInf = max([b1,b2,b3])
            borneSup = len(ag)
            if borneInf < borneSup:
                if i == 0:
                    u = None
                    c,cptemp = branchementAmeliorePlusPlusBornesGlouton(Gbis,C_temp,u)
                elif i == 1:
                    u = e[0]
                    c,cptemp = branchementAmeliorePlusPlusBornesGlouton(Gbis,C_temp,u)
                couvertures.append(c)
                cpt += cptemp + 1
            else:
                C_temp = C_temp + ag
                couvertures.append(C_temp)
                cpt += 1
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

'''Mesure le temps de calcul des algos au-dessus :'''

# en fonction de n
def mesure_temps_ameliorationBranchement_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yBA = []
    yBAP = []
    yBAPP = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchementAmelioreBornesGlouton(G)
        yBA.append(time.time() - start_time)
        start_time = time.time()
        branchementAmeliorePlusBornesGlouton(G)
        yBAP.append(time.time() - start_time)
        start_time = time.time()
        branchementAmeliorePlusPlusBornesGlouton(G)
        yBAPP.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yBA, c='green', label='branchement amélioré')
    plt.scatter(x, yBAP, c='blue', label='branchement amélioré plus')
    plt.scatter(x, yBAPP, c='red', label='branchement amélioré plus plus')
    plt.legend()
    plt.show()
    #return x,yBa,yBAP,yBAPP

# en fonction de p
def mesure_temps_ameliorationBranchement_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yBA = []
    yBAP = []
    yBAPP = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        branchementAmelioreBornesGlouton(G)
        yBA.append(time.time() - start_time)
        start_time = time.time()
        branchementAmeliorePlusBornesGlouton(G)
        yBAP.append(time.time() - start_time)
        start_time = time.time()
        branchementAmeliorePlusPlusBornesGlouton(G)
        yBAPP.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yBA, c='green', label='branchement amélioré')
    plt.scatter(x, yBAP, c='blue', label='branchement amélioré plus')
    plt.scatter(x, yBAPP, c='red', label='branchement amélioré plus plus')
    plt.legend()
    plt.show()
    #return x,yBa,yBAP,yBAPP

'''Mesure le nombre de noeuds visités des algos au-dessus :'''

# en fonction de n
def mesure_nbreNoeuds_ameliorationBranchement_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yBA = []
    yBAP = []
    yBAPP = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        sol,cptBA = branchementAmelioreBornesGlouton(G)
        sol,cptBAP = branchementAmeliorePlusBornesGlouton(G)
        sol,cptBAPP = branchementAmeliorePlusPlusBornesGlouton(G)
        yBA.append(cptBA)
        yBAP.append(cptBAP)
        yBAPP.append(cptBAPP)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yBA, c='green', label='branchement amélioré')
    plt.scatter(x, yBAP, c='blue', label='branchement amélioré plus')
    plt.scatter(x, yBAPP, c='red', label='branchement amélioré plus plus')
    plt.legend()
    plt.show()
    #return x,yBa,yBAP,yBAPP

# en fonction de p
def mesure_nbreNoeuds_ameliorationBranchement_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yBA = []
    yBAP = []
    yBAPP = []
    for i in range(nbInstances):
        #p = (i+1)*1/(nbInstances)
        p = 1 / math.sqrt(n)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        sol,cptBA = branchementAmelioreBornesGlouton(G)
        sol,cptBAP = branchementAmeliorePlusBornesGlouton(G)
        sol,cptBAPP = branchementAmeliorePlusPlusBornesGlouton(G)
        yBA.append(cptBA)
        yBAP.append(cptBAP)
        yBAPP.append(cptBAPP)
    plt.title('Nombre de noeuds visités en fonction de la taille n de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Nombre de noeuds visités')
    plt.scatter(x, yBA, c='green', label='branchement amélioré')
    plt.scatter(x, yBAP, c='blue', label='branchement amélioré plus')
    plt.scatter(x, yBAPP, c='red', label='branchement amélioré plus plus')
    plt.legend()
    plt.show()
    #return x,yBa,yBAP,yBAPP

# 4) Qualité des algorithmes approchés

# 1. Evaluation expérimentale du rapport d'approximation des algorithmes
# de couplage et glouton

def rapportApproximation(p,Nmax,nbInstances):
    x = np.linspace(1,Nmax,nbInstances)
    yC = []
    yG = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        g = gr.grapheAlea(n,p)
        sol,cpt = branchementAmeliorePlusPlusBornesGlouton(g)
        solC = ma.algo_couplage(g)
        solG = ma.algo_glouton(g)
        yC.append(len(solC)/len(sol))
        yG.append(len(solG)/len(sol))
    plt.title("Rapport d'approximation")
    plt.xlabel("Taille n")
    plt.ylabel("r-approché")
    plt.plot(x, yC, c='blue', label='algo_couplage')
    plt.plot(x, yG, c='red', label='algo_glouton')
    plt.legend()
    plt.show()
    #return x,yC,yG

# 2. Autres heuristiques

def lasVegas(G,C=[],u=None):
    cpt = 0
    E = list(G.edges())
    G_copie = gr.copie(G)
    if u in G_copie.nodes():
        gr.suppression(G_copie,u)
    E = list(G_copie.edges())
    if E == []:
        C.append(u)
        return C,cpt
    couvertures = []
    e = random.choice(E) # modification en random
    dico = dict(gr.degres(G))
    e_liste = list(e)
    if dico[e[0]] < dico[e[1]]:
        e_liste[0] = e[1]
        e_liste[1] = e[0]
    e = tuple(e_liste)
    for i in range(2):
        C_temp = C.copy()
        C_temp.append(e[i])
        Gbis = gr.suppression(gr.copie(G),e[i])
        if gr.valeurDegreMax(Gbis) == 0:
            couvertures.append(C_temp)
            cpt += 1
        elif dico[e[i]] == 1:
            cpt += 1
        else:
            ac = ma.algo_couplage(Gbis)
            ag = ma.algo_glouton(Gbis)
            n = len(list(Gbis.nodes()))
            m = len(list(Gbis.edges()))
            b1 = m / gr.valeurDegreMax(Gbis)
            b2 = len(ac) / 2
            b3 = ( 2*n-1-math.sqrt(((2*n-1)**2)-(8*m)) ) / 2
            borneInf = max([b1,b2,b3])
            borneSup = len(ag)
            if borneInf < borneSup:
                if i == 0:
                    u = None
                    c,cptemp = branchementAmeliorePlusPlusBornesGlouton(Gbis,C_temp,u)
                elif i == 1:
                    u = e[0]
                    c,cptemp = branchementAmeliorePlusPlusBornesGlouton(Gbis,C_temp,u)
                couvertures.append(c)
                cpt += cptemp + 1
            else:
                C_temp = C_temp + ag
                couvertures.append(C_temp)
                cpt += 1
    if len(couvertures)==0:
        return C,cpt
    elif len(couvertures)==1 or ( len(couvertures[0]) <= len(couvertures[1]) ):
        C = couvertures[0]
    else:
        C = couvertures[1]
    return C,cpt

# Comparaisons en fonction de n et de p

'''Mesure le temps de calcul des algos au-dessus :'''

# en fonction de n
def mesure_temps_lasVegas_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yB = []
    yLV = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        lasVegas(G)
        yLV.append(time.time() - start_time)
        start_time = time.time()
        branchementAmeliorePlusPlusBornesGlouton(G)
        yB.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yB, c='blue', label='avec Las Vegas')
    plt.scatter(x, yLV, c='red', label='sans Las Vegas')
    plt.legend()
    plt.show()

# en fonction de p
def mesure_temps_lasVegas_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yB = []
    yLV = []
    for i in range(nbInstances):
        p = (i+1)*1/(nbInstances)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        start_time = time.time()
        lasVegas(G)
        yLV.append(time.time() - start_time)
        start_time = time.time()
        branchementAmeliorePlusPlusBornesGlouton(G)
        yB.append(time.time() - start_time)
    plt.title('Temps de calcul moyen tn en fonction de la probabilité p de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Temps de calcul moyen tn en secondes')
    plt.scatter(x, yB, c='blue', label='avec Las Vegas')
    plt.scatter(x, yLV, c='red', label='sans Las Vegas')
    plt.legend()
    plt.show()

'''Mesure la qualité de la solution retournée des algos au-dessus :'''

# en fonction de n
def mesure_qualite_lasVegas_n(Nmax,p,nbInstances):    
    x = np.linspace(1,Nmax,nbInstances)
    yB = []
    yLV = []
    for i in range(nbInstances):
        n = int((i+1)*Nmax/nbInstances)
        G = gr.grapheAlea(n,p)
        yB.append(len(lasVegas(G)))
        yLV.append(len(branchementAmeliorePlusPlusBornesGlouton(G)))
    plt.title('Qualité de la solution retournée en fonction de la taille n de chaque instance')
    plt.xlabel('Taille n')
    plt.ylabel('Nombre de sommets de la solution retournée')
    plt.scatter(x, yB, c='blue', label='avec Las Vegas')
    plt.scatter(x, yLV, c='red', label='sans Las Vegas')
    plt.legend()
    plt.show()

# en fonction de p
def mesure_qualite_lasVegas_p(n,nbInstances):    
    x = np.linspace(0.1,1,nbInstances)
    yB = []
    yLV = []
    for i in range(nbInstances):
        p = (i+1)*1/(nbInstances)
        if p == 1:
            p = 0.99
        G = gr.grapheAlea(n,p)
        yB.append(len(lasVegas(G)))
        yLV.append(len(branchementAmeliorePlusPlusBornesGlouton(G)))
    plt.title('Qualité de la solution retournée en fonction de la probabilité p de chaque instance')
    plt.xlabel('Probabilité p')
    plt.ylabel('Nombre de sommets de la solution retournée')
    plt.scatter(x, yB, c='blue', label='avec Las Vegas')
    plt.scatter(x, yLV, c='red', label='sans Las Vegas')
    plt.legend()
    plt.show()