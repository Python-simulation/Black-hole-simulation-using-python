import matplotlib.pyplot as plt #graphical modules
import math                     #import known fonction and constants
import numpy as np              #use for matrices and list
from PIL import Image           #use to open, modify and save images
from scipy.interpolate import interp1d  #use for interpolation
import time                         #use to check computation time
from scipy.integrate import solve_ivp #integrate ordinary differential equations
import os.path                       #use to search files on computer
vrai_debut=time.process_time()
# =============================================================================                 
#Variables & constants initialisation (don't change)
t_max=10*math.pi
c=299792458            #light speed in vaccum m/s
G=6.67408 *10**-11     #Newton constant m3/Kg/s2
Ms=1.9891 *10**30      #solar mass in Kg
Ds=149597870700        #sun-earth distance in m
#------------------------------------------------------------------------------
"""Only values that can be changed (without knowing the program)"""
#M=1.7342*10**22/Ms            #black hole mass in solar mass  (alternative, del M below)
#Rs = 2*G*Ms*M/c**2/Ds          #Schwarzschild radius in Astronomical unit (ua)
Rs=8    #Schwarzschild radius in ua
M=Rs*c**2/2/G*Ds/Ms  #Black hole mass in solar masses  (del if use solar mass)
D=50    #distance from the black hole in ua
final_size_img=1000 # in axe_X
use_matrix=True #use matrices if exists
save_matrix=True #False: if don't want to save countless matrices, True: will save or overwrite matrices if exists
kind='linear'  #interpolation: linear for speed(less accurate), cubic for precision (slow)
FOV_img=360    #the image FOV (it doesn't change the current image FOV !)
FOV=FOV_img  #can be <= FOV_img to reduce compute area,
#FOV=100 #must be >FOV_img if FOV_img small(otherwise creat a non-compute cercle)
dossier='images\\' #not necessary but allow to separate initial images from computed ones
nom_court='milkyway'
#nom_court='téléchargement'
#nom_court='bg-color-grid' #extract from https://www.esa.int/gsp/ACT/phy/Projects/Blackholes/WebGL.html
#extension='.png'      #advice: use png if initial image is png (more precise)
extension='.jpg'    #advice: use jpg on jpg image (smaller than png in size)
offset_X_tot=0 #if first offset !=0 (allow to start at middle and keep going with a diferent offset_X)
offset_X=0 # initialize offset and can be use to choose offset instead of dependence on nbr_offset
nbr_offset=1    #number of image needed with a constant offset between (must be changed to specify a precise angle)
fixed_background=False
nom_image=dossier+nom_court+extension
#------------------------------------------------------------------------------
print ("M ","%.1e"%M,"M☉","\t%.2e"%(M*Ms),"Kg")
print("Rs",Rs,"ua","\t%.2e"%(Rs*Ds),"m") 
print("D ",D,"ua","\t%.2e"%(D*Ds),"m")  
# =============================================================================
# =============================================================================
def fun(phi,u):
    """Represent the differential equation : d²u(ɸ)/dɸ²=3/2*Rs*u²(ɸ)-u(ɸ)"""
    v0=u[1]  #correspond to u'
    v1=3/2*Rs*u[0]**2-u[0] #correspond to u"
    return ([v0,v1])
# =============================================================================
#def eventR(phi,u): #not needed and don't work with ivp (without it we get an error message but irrelevant)
#    """stop integration if radius > sphere limit"""
#    R=1000
#    return (1/u[0]-math.sqrt(R**2-D**2*math.sin(phi)**2)+D*math.cos(phi))
#eventR.terminal = True
def eventRs(phi,u):
    """stop integration if radius < black hole radius"""
    return (1/u[0]-Rs)
eventRs.terminal = True
# =============================================================================
def integration(D,alpha):
    """Integrate the function "fun" using solve_ivp.
    Compute photon trajectories giving his distance from black hole and his initial speed"""
    if alpha==0:     #skip divided by 0 error
        return [0],[0]          #r and phi=0
    if alpha==180:
        return [D],[0]         # if angle= pi then, tan(pi)=0 so 1/tan=1/0
    y0=[1/D, 1/(D*math.tan(alpha*math.pi/180))] #initial value for position and angular speed
    sol=solve_ivp(fun=fun, t_span=[0,t_max], y0=y0, method='Radau',dense_output=False, events=[eventRs])#,eventR])#,t_eval=np.linspace(0,t_max,10000))
    if sol.t[-1]==t_max:
        raise StopIteration ("integration error, alpha reached computing limit (loop number)")
    phi=sol.t
    r=abs(1/sol.y[0,:]) #must use this because integration can't be stop before infinity because negative
    return r,phi
# =============================================================================
def trajectories(D,alpha_finder,img_res,Rs):
    """Compute several photons trajectories to interpolate all possibles trajectories in the equatorial plan """
    #--------------------------------------------------------------------------
    debut=time.process_time()
    def search_alpha_min(D,img_res,Rs):
        """Return last angle at which the photon is kept by the black hole"""
#        debut=time.process_time()
        alpha_min=0
        for alpha in range(0,180,4):
            r,phi=integration(D,alpha)
            if r[-1]>1.1*Rs:
                break  
        if alpha-4 > 0:
            alpha_min=alpha-4
#        print("alpha_min :",alpha_min,"(-4)")                 
        i=1
        while alpha_min==0 or round(alpha_min*img_res) != round((alpha_min+i*10)*img_res):  #increase precision
            for alpha in range(int(alpha_min/i),int(180/i),1):
                alpha=alpha*i
                r,phi=integration(D,alpha)
                if r[-1] > 1.1*Rs:
                    break
            if alpha-i > 0:          
                alpha_min = alpha-i
#            print("alpha_min : ",alpha_min," (-",i,")",sep="")             
            i=i/10                
        i=10*i
        alpha_min+=i
        print("alpha_min: ",alpha_min," [",alpha_min-i,";",alpha_min,"]",sep="") 
#        fin=time.process_time()
#        print("min angle time",fin-debut)
        return alpha_min
    #--------------------------------------------------------------------------
    alpha_min=search_alpha_min(D,img_res,Rs)   
    ax = plt.subplot(111, projection='polar') #warning if use python in ligne (!= graphical) graphs got superposed
    plt.ylabel('phi(°)\n\n\n\n', rotation=0)
    ax.set_xlabel('R(UA)')
    ax.set_title("light trajectories close to a black hole\n", va='bottom')
    ax.set_rlim((0,4*D))
    ax.set_rlabel_position(-90)
    seen_angle=[]
    deviated_angle=[]
#    debut=time.process_time()
    booli=False  #avoid points from the first loop to exceed points from the second loop
    points=40
    for i in range(6):
#        print(alpha_finder)
        for alpha in np.linspace(alpha_finder, alpha_min, num=points, endpoint=booli):
            r,phi=integration(D,alpha) 
            if r[-1]>Rs*1.1: #if not capture by black hole
                seen_angle.append(180-alpha) #put 180 in the center
                deviated_angle.append(180/math.pi*(phi[-1]+math.asin(D/r[-1]*math.sin(phi[-1]))))    
                Ci='C'+str(i)
                ax.plot(phi,r,Ci)   #plot one trajectory   
        if kind=='linear':
            alpha_finder=alpha_min+(alpha_finder-alpha_min)/(points/3+1) # start a more precise cycle from last point    
        else:
            alpha_finder=alpha_min+(alpha_finder-alpha_min)/(points+1) # start a more precise cycle from last point                

        points=10 
        if i==4:
            booli=True #allow to display the last point
#        fin=time.process_time()
#    print("angles time",fin-debut)
    print("")
    fin=time.process_time()
    print("Trajectories time:",round(fin-debut,1)) 
#    plt.savefig('trajectories.png', format='png', dpi=1000, bbox_inches='tight')
#    plt.savefig('trajectories.eps', format='eps', dpi=1000, bbox_inches='tight')
    plt.show()   
    plt.close() #must be fixed if use spyder graph
    return seen_angle,deviated_angle
# =============================================================================
npts = 20001 
def splin(x_pivot,f_pivot):
    """From : source unknown 
    Display the interpolation and reduce compute time when used"""
    interpolation = interp1d(x_pivot, f_pivot, kind=kind,bounds_error=True)   # interpol function 
    xmin = min(x_pivot)
    xmax = max(x_pivot)

    dx = (xmax-xmin)/(npts-1)
    seen_angle_splin=[0]*(npts-1)
    deviated_angle_splin=[0]*(npts-1)
    for ix in range (0,npts-1):
       xx = xmin + dx*ix
       polc = interpolation(xx)
       seen_angle_splin[ix]=xx
       deviated_angle_splin[ix]=polc
    return seen_angle_splin, deviated_angle_splin
# =============================================================================
def img_offset_X(img_debut,offset_X):
    """Return the image with an offset in the X-axis. Allow to creat illusion of black hole movement"""
    if FOV != 360 and nbr_offset != 1 :
        raise StopIteration ("Can't compute offset for FOV != 360°")
    axe_X=img_debut.size[0]
    axe_Y=img_debut.size[1]
    while offset_X >= axe_X:
        offset_X-=axe_X
    if offset_X==0:
        return img_debut    
    if offset_X<0:
        offset_X=-offset_X
        img_right=img_debut.crop((0,0,axe_X-offset_X,axe_Y))
        img_left=img_debut.crop((axe_X-offset_X,0,axe_X,axe_Y))
        img_debut.paste(img_right,(offset_X,0))
    else:
        img_right=img_debut.crop((0,0,offset_X,axe_Y))
        img_left=img_debut.crop((offset_X,0,axe_X,axe_Y))
        img_debut.paste(img_right,(axe_X-offset_X,0))
    img_debut.paste(img_left,(0,0))
    return img_debut
# ============================================================================= 
def listdirectory2(path,matrix_file): 
    """from https://python.developpez.com/faq/?page=Fichier#isFile.
    Allow to search if files exist in folders"""
    for root, dirs, files in os.walk(path): 
        for i in files: 
            fichier=os.path.join(root, i)
            if fichier==matrix_file:
                return True
    return False
# =============================================================================
def spheric2cart(theta,phi):
    theta=theta/180*math.pi  
    phi=phi/180*math.pi
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)    
    return x,y,z
# =============================================================================
def cart2spheric(x,y,z):
    #doesn't compute r because chosen egal to 1
    theta=math.acos(z)*180/math.pi
    phi=math.atan2(y,x)*180/math.pi
    while phi<0: #define phi [0,360]
        phi+=360
    while theta<0: # define theta [0,180]
        theta+=180 
    if phi==360:
        phi=0
    return theta,phi
# =============================================================================
def rotation_matrix(beta):
#    beta*=math.pi/180
    """from https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    Return the rotation matrix associated with counterclockwise rotation about
    the x axis by beta degree."""
    a = math.cos(beta / 2.0)
    b= -math.sin(beta / 2.0)
    aa, bb= a*a, b*b
    ab= a*b
    return np.array([[aa + bb, 0, 0],
                     [0, aa - bb, 2*ab],
                     [0, -2*ab, aa - bb]])
# =============================================================================
def find_position(x,y,interpolation):
    """take seen pixel position and search deviated pixel position."""
    if y==0: #skip theta=0
        return x,y
    phi,theta=x*FOV_img/360/img_res,y*FOV_img_Y/180/img_res_Y #convert position in spheric coord
    phi,theta=phi+(360-FOV_img)/2,theta+(180-FOV_img_Y)/2
#    print(phi,theta)
    u,v,w=spheric2cart(theta,phi) #give cartesian coord of pixel
    if theta==90: #needed to avoid error with atan()
        beta=0
    elif phi==180 or phi==0:
        beta=math.pi/2        
    else:
        beta=-math.atan(w/v) #see complex number argument to find angle between plan 0xv and equator (same value for all x -> projection on y,z) 
    v2= np.dot(rotation_matrix(beta), [u,v,w]) #take 3*3 created matrix and aplly to vector
    _,seen_angle=cart2spheric(v2[0],v2[1],v2[2])    #return phi in equator "projection"

    if seen_angle>360: #assure angle is in [0,360]
        seen_angle-=360
    if seen_angle>180: #only right because spherical problem (not right with Kerr)
        seen_angle=360-seen_angle
        try:
#            deviated_angle=360-interpolation(seen_angle) #old version slower
            deviated_angle=360-deviated_angle_splin[int(seen_angle*(npts-1)/last_angle)] #search deviated angle base on seen angle           
        except:
            return -1,-1 #inside photosphere (black hole)
    else:
        try:
#            deviated_angle=interpolation(seen_angle) #old version slower
            deviated_angle=deviated_angle_splin[int(seen_angle*(npts-1)/last_angle)]#search deviated angle base on seen angle           
        except:
            return -1,-1  #inside photosphere (black hole)       
    u,v,w=spheric2cart(90,deviated_angle) #get cart coord of deviated pixel
    v2= np.dot(rotation_matrix(-beta), [u,v,w]) #rotate back to the original plan
    theta,phi=cart2spheric(v2[0],v2[1],v2[2])   #give spheric coord of deviated pixel
    phi,theta=phi-(360-FOV_img)/2,theta-(180-FOV_img_Y)/2
    x2,y2=phi*360/FOV_img*img_res,theta*180/FOV_img_Y*img_res_Y #give deviated angle pixel position
    return x2,y2   # return float but matrices will implicitly floor them
# =============================================================================
def matrices_creation(interpolation):
    """Call find_position function and creat matrices with pixels positions informations."""
    img_matrice_x=np.array([[-1]*axe_X]*axe_Y)
    img_matrice_y=np.array([[-1]*axe_X]*axe_Y)
    debut=time.process_time()
    if FOV<FOV_img:
        print("\nmatrix estimation time:",round(3.73*10**(-5)*(FOV/FOV_img)*(axe_X*axe_Y-(2*(180-last_angle)/360*axe_X)**2),1)) #modif les FOV en x et y car diff si on a fov_x=360 fov_y=180
    else:
        print("\nmatrix estimation time:",round(3.73*10**(-5)*(axe_X*axe_Y-(2*(180-last_angle)/360*axe_X)**2),1)) #modif les FOV en x et y car diff si on a fov_x=360 fov_y=180
#    for x in range(3000,axe_X-3000):        
    for x in range(0,axe_X): #colomns scan   (phi)   mettre autre axe_x pour commencer que debut vrai image (enleve partie noir)
        if x==round(axe_X/4):
            print("25%")
        elif x==round(axe_X/2):
            print("50%")
        elif x==round(axe_X*3/4):
            print("75%")
#        for y in range(1500,axe_Y-1500): take back old program to resize image        
        for y in range(0,axe_Y): #lines scan (theta)
            x2,y2=find_position(x,y,interpolation) #search deviated angle pixel position
            img_matrice_x[y,x]=x2 #create matrices to use data at any time
            img_matrice_y[y,x]=y2
    fin=time.process_time()
    print("matrix time:",round(fin-debut,1))
    return img_matrice_x,img_matrice_y
# =============================================================================
def img_pixels(img_debut,img2):
    """Use matrices, take deviated pixels color and assign them to seen pixels."""
    debut=time.process_time()
    pixels = img_debut.load()
    pixels2 = img2.load() #could have put direcly image creation here, but wanted to recover data if crash or computation stop
    for x in range(0,axe_X):   #colomns scan   (phi)
        for y in range(0,axe_Y):  #lines scan (theta)
            x2=int(img_matrice_x[y,x]) #extract data from matrix
            y2=int(img_matrice_y[y,x])
            if x2 != -1:
                try:
                    R1,G1,B1=pixels[x2,y2] #get deviated pixel color
                    pixels2[x,y] = (R1, G1, B1) # colorize seen pixel with deviated pixel color
                except:
                    pixels2[x,y] = (255, 0, 0) #optional (colorize pixels out of picture)
                    ""
            else:       
#                pixels2[x,y] = (0, 0, 0) #optional (colorize the black hole instead of alpha)
                ""
    if fixed_background==True:
        img2=img_offset_X(img2,-offset_X_tot)  # if want a fixed background and moving black hole
    fin=time.process_time()
    print("pixels time :",round(fin-debut,1))         
#    img2.save(nom_court+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+".png") #use it instead if use high contrasted image
    img2=img2.convert("RGB") 
    if nbr_offset==1:
        img2.show()   
        img2.save(nom_court+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+extension)  #use png instead if accuracy needed  
    else:
        img2.save(nom_court+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+" offset_X="+str(offset_X_tot)+extension)
    return img2
# =============================================================================
img_debut = Image.open(nom_image, mode='r') #use equirectangular images
img_debut=img_debut.convert("RGB")  #if needed RGBA, change all "pixels" ligne by RGBA instead of RGB
axe_X=img_debut.size[0]
axe_Y=img_debut.size[1]
size_factor=final_size_img/axe_X
axe_X=int(size_factor*axe_X)
axe_Y=int(size_factor*axe_Y)
if axe_X % 2 != 0:   #even dimensions needed for image
    axe_X-=1
if axe_Y % 2 != 0:
    axe_Y-=1 
final_size_img=axe_X
img_debut = img_debut.resize((axe_X,axe_Y),Image.ANTIALIAS) #can be avoid if put axe_X before resize with conditions
FOV_img_Y=FOV_img*axe_Y/axe_X
print("\nFOV ",FOV_img,"*",FOV_img_Y,"°",sep='')
if FOV_img_Y>180:
    raise StopIteration ("Can't have a FOV>360 in the Y-axis")
print('size ',axe_X,"*",axe_Y," pixels\n",sep='')
img_res=axe_X/360  #=Pixels per degree along axis
img_res_Y=axe_Y/180  #=Pixels per degree along axis
#------------------------------------------------------------------------------
seen_angle,deviated_angle=trajectories(D,FOV/2,img_res,Rs)  #call funtion trajectories to compute angles

#ax2 = plt.subplot(111) 
#plt.ylabel('deviated angle(°)')
#plt.xlabel('seen angle(°)')
#plt.title("Check if good interpolation (-final pts)", va='bottom')
#seen_angle_test=list(seen_angle)
#deviated_angle_test=list(deviated_angle)
#seen_angle_test.remove(seen_angle_test[-1])
#deviated_angle_test.remove(deviated_angle_test[-1])
#ax2.plot(seen_angle_test,deviated_angle_test,'o')
#plt.show()

seen_angle_splin,deviated_angle_splin = splin(seen_angle,deviated_angle)  #to display the interpolation( not needed for commpute)
ax3 = plt.subplot(111) #not needed for computation
plt.ylabel('deviated angle(°)')
plt.xlabel('seen angle(°)')
plt.title("Light deviation interpolation", va='bottom')
ax3.plot(seen_angle,deviated_angle,'o')
ax3.plot(seen_angle_splin,deviated_angle_splin)
#plt.xlim((100,160))
#plt.ylim((120,350))
#plt.savefig('interpolation.png', format='png', dpi=1000, bbox_inches='tight')
#plt.savefig('interpolation.eps', format='eps', dpi=1000, bbox_inches='tight')
plt.show()
plt.close()
last_angle=seen_angle[-1]
print("last angle",last_angle)
#------------------------------------------------------------------------------
if use_matrix==True:    #check if matrix exist
    dir_path = os.path.dirname(os.path.realpath(__file__))+'\\matrix\\'
    matrix_file=dir_path+str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_x.txt'
    x_file=listdirectory2(dir_path,matrix_file)
    matrix_file=dir_path+str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_y.txt'
    y_file=listdirectory2(dir_path,matrix_file)
else:
    x_file=False
    y_file=False
if x_file == True and y_file == True:
    print("\nmatrices already exist, skipping steps")
    if FOV<FOV_img:
        print("saving this time:",round(3.73*10**(-5)*(FOV/FOV_img)*(axe_X*axe_Y-(2*(180-last_angle)/360*axe_X)**2),1))
    else:
        print("saving this time:",round(3.73*10**(-5)*(axe_X*axe_Y-(2*(180-last_angle)/360*axe_X)**2),1))
    print("\nmatrix opening estimation:",round(1.65*10**(-6)*axe_X*axe_Y,1)) 
    matrix_opening_debut=time.process_time()
    img_matrice_x=np.array([-1]*axe_X)
    img_matrice_y=np.array([-1]*axe_Y)
    img_matrice_x=np.loadtxt('matrix\\'+str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_x.txt') #to be rigorous, add final_size_img_Y, FOV_img, FOV_img_Y, kind..
    img_matrice_y=np.loadtxt('matrix\\'+str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_y.txt') 
    matrix_opening_fin=time.process_time()
    print("matrix opening time:",round(matrix_opening_fin-matrix_opening_debut,1))
else:
    interpolation = interp1d(seen_angle, deviated_angle, kind=kind,bounds_error=True)
    img_matrice_x,img_matrice_y=matrices_creation(interpolation) #create a matrix with corresponding (x,y) -> (x2,y2)
    if save_matrix==True:
        np.savetxt('matrix\\'+str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_x.txt', img_matrice_x, fmt='%i')
        np.savetxt('matrix\\'+str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_y.txt', img_matrice_y, fmt='%i') 
#------------------------------------------------------------------------------
img_debut=img_offset_X(img_debut,offset_X_tot)  #allow to begin with an offset then carry on with diff offsets (useful if stop prematuraly and want to start back in the middle)
print("\ntotal offsets estimation time:",round(1.6*10**-6*axe_X*axe_Y*nbr_offset,1))
for a in range(nbr_offset):
    print("\n",a+1,"/",nbr_offset,sep='')
    print("offset_X:",offset_X_tot)
    img_debut=img_offset_X(img_debut,offset_X)  
    img2=Image.new( 'RGBA', (axe_X,axe_Y)) #creat a transparent image (outside of the function to recover info if must stop loop)
    img2=img_pixels(img_debut,img2)
    offset_X=int(axe_X/nbr_offset) #at the end to have offset=0 for the first iteration
    offset_X_tot+=offset_X
vrai_fin=time.process_time()
print("\nglobal computing time:",round(vrai_fin-vrai_debut,1))
