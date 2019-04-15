import matplotlib.pyplot as plt #module d'affichage graphique
import math                     #module de calcul

"""Initialisation des variables & constantes"""
#----------------------------
#ajouter input equation de u2 en symbolique ?
c=1                             #vitesse de la lumière dans le vide
G=1                             #Constante de Newton
M=1                             #Masse du trou noir
distance_max=1                  #Multiple de D permettant d'arrêter la simulation avant la divergence
dphi = 10**(-4)                 #Intervalle de phi<<1 (10^-4) pour éviter les divergences
ITERATION=int(3*math.pi/dphi)   #Nombre de points à calculer (supérieur à 2pi car certains orbites peuvent aller au delà d'un tour complet)
Rs = 2*G*M/c**2                 #Rayon de Schwarzschild

phi_initial=0

"""Integration d'Euler"""
#----------------------------
def euler(dist,angle):
    angle=angle*math.pi/180     #passage en radian pour la fonction tan
    if angle==0:                #permet d'eviter l'erreur division par zero
        angle=0.01

    u=[1/dist]*ITERATION        #Variable associée à la position initial (u est multiplier par iteration pour remplire toutes les cases de la liste et permettre un calcul plus rapide par la suite)
    u1=1/(dist*math.tan(angle)) #Variable associée à la direction initiale
    
    ITERATION_REEL=0
    for i in range(ITERATION-1):#on réalise une itération de moins car le calcul nous donne la valeur i+1
        ITERATION_REEL+=1
        u2=3/2*Rs*u[i]**2-u[i]  #calcul de d²u/dɸ²  approximée à partir des équations des géodésiques et des conditions initiales (u=1/r)
        u1=u1+u2*dphi           #calcul de du/dɸ approximée
        u[i+1]=u[i]+u1*dphi     #calcul de u(ɸ) approximée
        if 1/u[i+1]<=Rs or 1/u[i+1]>distance_max*dist:  #condition qui stop le calcul si la particule rentre dans le trou noir
            break                                       #ou si la distance au trou noir est trop grande(divergence)

    #Création de phi et r à partir des valeurs de u renseignées (il faut réduire la taille des listes car la liste u n'est jamais remplit à cause des conditions d'arrêt)
    phi=[phi_initial]*ITERATION_REEL       
    r=[dist]*ITERATION_REEL
    for i in range(ITERATION_REEL-1):
        phi[i+1]=phi[i]+dphi
        r[i+1]=1/u[i]

    return phi,r #permet de récupérer les listes de r et phi pour analyse tout en oubliant les autres variables utilisées dans la fonction euler

#permet de créer une liste comprennant la taille du trou noir en fonction de phi (constant pour Schwarzschild)
iter=101 #comme le trou noir est indépendant des variables des trajectoires des particules, on peut choisir un interval différent et réduire le temps de calcul
black_hole=[Rs]*iter 
orbite_stable=[3*Rs]*iter
phi_entier=[0]*iter
for i in range(iter-1):
    phi_entier[i+1]=phi_entier[i]+2*math.pi/(iter-1)

"""Affichage"""
#------------------------
#afficher les trajectoires parallèles entre elles et perpendiculaire au trou noir
plt.figure(num='light deviation') #plot differents figure according to a specific name
ax = plt.subplot(111, projection='polar') #subplot permet de mettre plusieurs graph en 1
plt.xlabel('phi(°)')                      #le processus d'affichage est détaillé dans la fonction affichage
ax.set_ylabel('R(UA)\n\n\n', rotation=0)
ax.set_title("Deviation of light close to a black hole\n", va='bottom')
ax.set_rlabel_position(-90)
ax.plot(phi_entier,black_hole, label='black hole')
ax.plot(phi_entier,orbite_stable, label='dernier orbite stable')
D=500
for dD in range(0,200,1):
    dD=dD/10
    phi_initial=math.atan(dD/D) #décalage de phi pour être sur une ligne perpendiculaire au trou noir
    phi,r=euler(math.sqrt(D**2+dD**2),math.atan(dD/D)*180/math.pi) #même décalage pour l'angle d'incidence
    ax.plot(phi,r)
ax.legend(loc='best')    
plt.show()    
phi_initial=0

#afficher les trajectoires parallèles entre elles
plt.figure(num='light deviation 2') #plot differents figure according to a specific name
ax = plt.subplot(111, projection='polar') #subplot permet de mettre plusieurs graph en 1
plt.xlabel('phi(°)')                      #le processus d'affichage est détaillé dans la fonction affichage
ax.set_ylabel('R(UA)\n\n\n', rotation=0)
ax.set_title("Deviation of light close to a black hole\n", va='bottom')

ax.set_rlabel_position(-90)
ax.plot(phi_entier,black_hole, label='black hole')
ax.plot(phi_entier,orbite_stable, label='dernier orbite stable')
for D in range(10,50,5):
    phi,r=euler(D,20)
    ax.plot(phi,r)
ax.legend(loc='best')

#afficher les trajectoires pour une position donnée      
plt.figure(num='light deviation 3') #plot differents figure according to a specific name
ax = plt.subplot(111, projection='polar')
ax.set_rlabel_position(-90)
ax.plot(phi_entier,black_hole, label='black hole')
ax.plot(phi_entier,orbite_stable, label='dernier orbite stable')
for alpha in range(0,80,10):
    phi,r=euler(10,alpha)
    ax.plot(phi,r)
plt.show()

#afficher qu'une seule trajectoire
#----------------------------
def affichage(func,x,y):    #permet de prendre comme argument n'importe quelle fonction ayant elle même deux arguments
 #plot differents figure according to a specific name    
    phi,r=func(x,y) #appel la fonction au choix (ici euler) et associe les variables phi et r aux valeurs de sorties
    ax.plot(phi,r,label='trajectoire de la particule') #label donne un nom à la courbe
    ax.legend(loc='best') #permet de positionner la légende au mieux

#D=float(input('distance au trou noir :')) #on pourrait insérer D et alpha dans affichage et mettre explicitement euler à la place de func mais,
#alpha=float(input('angle de départ :'))     #en les sortant, on peut réaliser plus facilement des boucles ou autre opérations utilisant la fonction affichage
#affichage(euler,D,alpha)            #permet d'appeler la fonction euler avec les valeurs choisis et d'afficher la trajectoire associée
plt.figure(num='light deviation 4')
ax = plt.subplot(111, projection='polar') #permet de projeter les coordonnées en polaire
ax.set_title("Deviation of light close to a black hole\n", va='bottom')  #donne un titre au graphique
ax.set_rlabel_position(-90)             # modifie l'angle d'affichage de l'axe radial
plt.xlabel('phi(°)')                    #donne un nom aux axes
ax.set_ylabel('R(UA)\n\n\n', rotation=0)
ax.plot(phi_entier,black_hole, label='black hole')  #permet de visualiser l'horizon du trou noir
ax.plot(phi_entier,orbite_stable, label='dernier orbite stable')  #permet de visualiser la zone dans laquelle les orbites ne sont plus stables

affichage(euler,20,15)
affichage(euler,3,90)

plt.show() #affiche le ou les subplot en même temps
