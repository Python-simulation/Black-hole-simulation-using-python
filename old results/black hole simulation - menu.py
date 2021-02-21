import sys
import math  #import known fonction and constants
import time  # Used to check computation time
import os.path # Used to search files on computer

import matplotlib.pyplot as plt  # Graphical module
import numpy as np # Use for matrices and list
from scipy.interpolate import interp1d  # Use for interpolation
from scipy.integrate import solve_ivp  # Integrate ordinary differential equations
from scipy.constants import pi
from scipy.constants import c  # Speed light vaccum (m/s)
from scipy.constants import G  # Newton constant m3/Kg/s2
from PIL import Image  # Use to open, modify and save images
from tkinter import Button, Tk, Frame, Entry, Label, Checkbutton, BooleanVar, StringVar, Spinbox
from tkinter.filedialog import askopenfilename
from numba import jit, prange  # njit
# could be nice to use #@njit(parallel=True)

"""
Programs simulating photons trajectory closed to a static black hole.
Render the perceive image deformation by the black hole.
The code is ugly because it has been done when I first started python.
Must change it completly to use classes and suppress the awful global variables
"""
# =============================================================================
"""Variables & constants initialisation (don't change)"""
t_max = 10*pi
Ms = 1.9891 * 10**30  # solar mass in Kg
Ds = 149597870700  # sun-earth distance in m
#------------------------------------------------------------------------------
"""Only values that can be changed (without knowing the program)"""
#M = 1.7342*10**22/Ms  # Black hole mass in solar mass (alternative, del M below)
#Rs = 2*G*Ms*M/c**2/Ds  # Schwarzschild radius in Astronomical unit (ua)
Rs = 8  # Schwarzschild radius in ua
M = Rs * c**2 / 2 / G * Ds / Ms  # Black hole mass in solar masses  (del if use solar mass)
D = 50  # Distance from the black hole in ua
final_size_img = 360  # Over axe_X
use_matrix = True  # Use matrices if exists
save_matrix = True  # False: if don't want to save countless matrices, True: will save or overwrite matrices if exists
kind = 'linear'  # Interpolation: linear for speed(less accurate), cubic for precision (slow)
FOV_img = 360  # The image FOV (it doesn't change the current image FOV !)
FOV = FOV_img  # Can be <= FOV_img to reduce compute area,
#FOV = 100  # Must be >FOV_img if FOV_img small(otherwise creat a non-compute cercle)
offset_X_tot = 0  # If first offset !=0 (allow to start at middle and keep going with a diferent offset_X)
offset_X = 0 #  Initialize offset and can be use to choose offset instead of dependence on nbr_offset
nbr_offset = 1  # Number of image needed with a constant offset between (must be changed to specify a precise angle)
fixed_background = True
display_trajectories = True
display_interpolation = True
#------------------------------------------------------------------------------
print ("M ", "%.1e"%M, "M☉", "\t%.2e"%(M*Ms), "Kg")
print("Rs", Rs, "ua", "\t%.2e"%(Rs*Ds), "m")
print("D ", D, "ua", "\t%.2e"%(D*Ds), "m")


def return_folder_file_extension(image_name):
    *folder, file = image_name.replace("\\", "/").split("/")

    if len(folder) != 0:
        folder = folder[0]

    file, extension = file.split(".")

    return folder, file, "."+extension

def fun(phi, u):
    """Represent the differential equation : d²u(ɸ)/dɸ²=3/2*Rs*u²(ɸ)-u(ɸ)"""
    v0 = u[1]  #correspond to u'
    v1 = 3/2*Rs*u[0]**2 - u[0] #correspond to u"
    return v0, v1


#def eventR(phi, u): #not needed and don't work with ivp (without it we get an error message but irrelevant)
#    """stop solver if radius > sphere limit"""
#    R = 1000
#    return (1/u[0]-math.sqrt(R**2-D**2*math.sin(phi)**2)+D*math.cos(phi))
#eventR.terminal = True


def eventRs(phi, u):
    """stop solver if radius < black hole radius"""
    return 1/u[0] - Rs
eventRs.terminal = True


def solver(D, alpha):
    """Solve the differetial equation, in spherical coordinate, for a static
    black hole using solve_ivp.
    Allows to compute the photon trajectory giving its distance from the
    black hole and its initial angular speed"""
    if alpha == 0:     #skip divided by 0 error
        return [0], [0]          #r and phi=0

    if alpha == 180:
        return [D], [0]         # if angle= pi then, tan(pi)=0 so 1/tan=1/0

    y0 = [1/D, 1/(D*math.tan(math.radians(alpha)))] #initial value for position and angular speed
    sol = solve_ivp(fun=fun, t_span=[0, t_max], y0=y0, method='Radau', events=[eventRs])#,eventR])#,t_eval=np.linspace(0, t_max, 10000)) #dense_output=False

    if sol.t[-1] == t_max:
        raise StopIteration ("solver error, alpha reached computing limit (loop number)")

    phi = sol.t
    r = np.abs(1/sol.y[0,:]) #must use this because solver can't be stop before infinity because negative

    return r, phi


def search_alpha_min(D, img_res, Rs):
    """Return last angle at which the photon is kept by the black hole"""
#        debut = time.process_time()
    alpha_min = 0

    for alpha in range(0, 180, 4):
        r, phi = solver(D, alpha)
        if r[-1] > 1.1*Rs:
            break

    if alpha-4 > 0:
        alpha_min = alpha-4
#        print("alpha_min :",alpha_min,"(-4)")
    i = 1

    while alpha_min == 0 or round(alpha_min*img_res) != round((alpha_min+i*10)*img_res):  #increase precision

        for alpha in range(int(alpha_min/i), int(180/i), 1):
            alpha = alpha*i
            r, phi = solver(D, alpha)

            if r[-1] > 1.1*Rs:
                break

        if alpha-i > 0:
            alpha_min = alpha-i
#            print("alpha_min : ",alpha_min," (-",i,")",sep="")

        i = i/10
    i = 10*i
    alpha_min += i
    print("alpha_min: ",alpha_min," [",alpha_min-i,";",alpha_min,"]",sep="")
#        fin = time.process_time()
#        print("min angle time",fin-debut)

    return alpha_min


def trajectories(D, alpha_finder, img_res, Rs):
    """Compute several photons trajectories in order to interpolate the
    possibles trajectories and gain in execution time"""
#    debut = time.process_time()

    alpha_min = search_alpha_min(D, img_res, Rs)

    if display_trajectories is True:
        plt.figure(num='trajectories')
        plt.clf() #clear the graph to avoir superposing data from the same set (can be deactivated if need to superpose)
        ax = plt.subplot(111, projection='polar') #warning if use python in ligne (!= graphical) graphs got superposed
        plt.ylabel('phi(°)\n\n\n\n', rotation=0)
        ax.set_xlabel('R(UA)')
        ax.set_title("light trajectories close to a black hole\n", va='bottom')
        ax.set_rlim((0, 4*D))
        ax.set_rlabel_position(-90)

    seen_angle = []
    deviated_angle = []
#    debut = time.process_time()
    booli = False  #avoid points from the first loop to exceed points from the second loop
    points = 40

    for i in range(6):
#        print(alpha_finder)

        for alpha in np.linspace(alpha_finder, alpha_min, num=points, endpoint=booli):
            r, phi = solver(D, alpha)

            if r[-1]>Rs*1.1: #if not capture by black hole
                seen_angle.append(180-alpha) #put 180 in the center
                deviated_angle.append(math.degrees((phi[-1]+math.asin(D/r[-1]*math.sin(phi[-1])))))
                Ci = 'C'+str(i)

                if display_trajectories is True:
                    ax.plot(phi, r, Ci)   #plot one trajectory

        if kind == 'linear':
            alpha_finder = alpha_min + (alpha_finder - alpha_min)/(points/3 + 1) # start a more precise cycle from last point

        else:
            alpha_finder = alpha_min + (alpha_finder - alpha_min)/(points + 1) # start a more precise cycle from last point

        points = 10

        if i == 4:
            booli = True #allow to display the last point
#        fin = time.process_time()
    #--------------------------------------------------------------------------
#    print("angles time",fin-debut)
    print("")
#    fin = time.process_time()
#    print("Trajectories time:",round(fin-debut, 1))

    if display_trajectories is True:
#        plt.savefig('trajectories.png', format='png', dpi=1000, bbox_inches='tight')
#        plt.savefig('trajectories.eps', format='eps', dpi=1000, bbox_inches='tight')
        plt.draw()
#        plt.close() #must be fixed if use spyder graph
    return seen_angle, deviated_angle


npts=20001  # Should not be here -> see how to remove it
def interpolate(x_pivot, f_pivot, npts=20001):
    """Display the interpolation (allows to reduce computation time when used)
    """
    interpolation = interp1d(x_pivot, f_pivot, kind=kind, bounds_error=True)
    xmin = min(x_pivot)
    xmax = max(x_pivot)

    seen_angle_splin = np.linspace(xmin, xmax, npts)
    deviated_angle_splin = interpolation(seen_angle_splin)

    return seen_angle_splin, deviated_angle_splin


def img_offset_X(img_debut, offset_X):
    """Return the image with an offset in the X-axis. Allow to creat illusion of black hole movement"""
    if FOV != 360 and nbr_offset != 1 :
        raise StopIteration ("Can't compute offset for FOV != 360°")

    axe_X = img_debut.size[0]
    axe_Y = img_debut.size[1]

    while offset_X >= axe_X:
        offset_X -= axe_X

    if offset_X == 0:
        return img_debut

    if offset_X < 0:
        offset_X = -offset_X
        img_right = img_debut.crop((0, 0, axe_X-offset_X, axe_Y))
        img_left = img_debut.crop((axe_X-offset_X, 0, axe_X, axe_Y))
        img_debut.paste(img_right, (offset_X, 0))

    else:
        img_right = img_debut.crop((0, 0, offset_X, axe_Y))
        img_left = img_debut.crop((offset_X, 0, axe_X, axe_Y))
        img_debut.paste(img_right, (axe_X-offset_X, 0))

    img_debut.paste(img_left, (0, 0))

    return img_debut


def listdirectory2(path, matrix_file):
    """from https://python.developpez.com/faq/?page=Fichier#isFile.
    Allow to search if files exist in folders"""
    for root, dirs, files in os.walk(path):
        for i in files:
            fichier = os.path.join(root, i)
            if fichier == matrix_file:
                return True
    return False


@jit(nopython=True)
def spheric2cart(theta, phi):
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return x, y, z


@jit(nopython=True)
def cart2spheric(x, y, z):
    #doesn't compute r because chosen egal to 1
    theta = math.acos(z)
    phi = math.atan2(y, x)
    while phi < 0: #define phi [0, 360]
        phi += pi + pi
    while theta < 0: # define theta [0, 180]
        theta += pi
    if phi == (pi + pi):
        phi = 0
    return theta, phi


def rotation_matrix(beta):
    """from https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    Return the rotation matrix associated with counterclockwise rotation about
    the x axis by beta degree."""
    a = math.cos(beta / 2.0)
    b = -math.sin(beta / 2.0)
    aa, bb = a*a, b*b
    ab = a*b
    return np.array([[aa + bb, 0, 0],
                     [0, aa - bb, 2*ab],
                     [0, -2*ab, aa - bb]])


def find_position(x, y):
    """take seen pixel position and search deviated pixel position."""
    global axe_X, axe_Y, img_res, img_res_Y, deviated_angle_splin, last_angle

    if y == 0: # skip theta=0
        return x, y

    phi, theta = x*FOV_img/360/img_res, y*FOV_img_Y/180/img_res_Y #convert position in spheric coord
    phi, theta = phi+(360-FOV_img)/2, theta+(180-FOV_img_Y)/2
#    print(phi, theta)
    u, v, w = spheric2cart(math.radians(theta), math.radians(phi)) #give cartesian coord of pixel

    if theta == 90:  # Needed to avoid error with atan()
        beta = 0

    elif phi == 180 or phi == 0:
        beta = pi/2

    else:
        beta = -math.atan(w/v) #see complex number argument to find angle between plan 0xv and equator (same value for all x -> projection on y, z)

    v2 = np.dot(rotation_matrix(beta), [u, v, w]) #take 3*3 created matrix and aplly to vector
    _, seen_angle = cart2spheric(v2[0], v2[1], v2[2])    #return phi in equator "projection"
    seen_angle = math.degrees(seen_angle)

    if seen_angle > 360: #assure angle is in [0, 360]
        seen_angle -= 360

    if seen_angle > 180: #only right because spherical problem (not right with Kerr)
        seen_angle = 360-seen_angle

        try:
            deviated_angle = 360-deviated_angle_splin[int(seen_angle*(npts-1)/last_angle)] #search deviated angle base on seen angle
        except:
            return -1, -1 #inside photosphere (black hole)
    else:

        try:
            deviated_angle = deviated_angle_splin[int(seen_angle*(npts-1)/last_angle)]#search deviated angle base on seen angle
        except:
            return -1,-1  #inside photosphere (black hole)

    u, v, w = spheric2cart(pi/2, math.radians(deviated_angle)) #get cart coord of deviated pixel
    v2 = np.dot(rotation_matrix(-beta), [u, v, w]) #rotate back to the original plan
    theta, phi = cart2spheric(v2[0], v2[1], v2[2])   #give spheric coord of deviated pixel
    theta, phi = math.degrees(theta), math.degrees(phi)
    phi, theta = phi-(360-FOV_img)/2, theta-(180-FOV_img_Y)/2
    x2, y2 = phi*360/FOV_img*img_res, theta*180/FOV_img_Y*img_res_Y #give deviated angle pixel position

    return x2, y2   # return float but matrices will implicitly floor them


def matrices_creation():
    """Call find_position function and creat matrices with pixels positions informations."""
    global axe_X, axe_Y, left_side, right_side, up_side, down_side, img_matrice_x, img_matrice_y, FOV, FOV_img, last_angle

    img_matrice_x = np.array([[-1]*axe_X]*axe_Y)
    img_matrice_y = np.array([[-1]*axe_X]*axe_Y)
    debut = time.process_time()

    if FOV < FOV_img:
        print("\nmatrix creation estimation time:",round(2.98*10**(-5)*(FOV/FOV_img)*(axe_X*axe_Y-(2*(180-last_angle)/360*axe_X)**2),1)) #modif les FOV en x et y car diff si on a fov_x=360 fov_y=180

    else:
        print("\nmatrix creation estimation time:",round(2.98*10**(-5)*(axe_X*axe_Y-(2*(180-last_angle)/360*axe_X)**2),1)) #modif les FOV en x et y car diff si on a fov_x=360 fov_y=180

    for x in prange(left_side, right_side):
#    for x in range(0, axe_X): #colomns scan   (phi)   mettre autre axe_x pour commencer que debut vrai image (enleve partie noir)

        if x == round(axe_X/4):
            print("25%")
        elif x == round(axe_X/2):
            print("50%")
        elif  x== round(axe_X*3/4):
            print("75%")
#        for y in range(1500, axe_Y-1500): take back old program to resize image
#        y=np.linspace(0, axe_Y-1, axe_Y)  # doesn't work but need to be test if reduce time

        for y in prange(up_side, down_side):
#        for y in range(0, axe_Y): #lines scan (theta)
            x2, y2 = find_position(x, y) #search deviated angle pixel position
            img_matrice_x[y, x] = x2 #create matrices to use data at any time
            img_matrice_y[y, x] = y2

    fin = time.process_time()
    print("matrix created in time:",round(fin-debut, 1), "s")

    return img_matrice_x, img_matrice_y


def img_pixels(img_debut, img2):
    """Use matrices, take deviated pixels color and assign them to seen pixels."""
    global left_side, right_side, up_side, down_side, img_matrice_x, img_matrice_y #, image_name

    pixels = img_debut.load()
    pixels2 = img2.load() #could have put direcly image creation here, but wanted to recover data if crash or computation stop

    # TODO: I'm sure it can be done with numpy matrix in one line
    for x in prange(left_side, right_side):   #faster than 0 to axe_X if zoom on high quality image but not compute if unzoom

        for y in prange(up_side, down_side):  #lines scan (theta)
#    for x in range(0, axe_X):   #colomns scan   (phi)
#        for y in range(0, axe_Y):  #lines scan (theta)
            x2 = int(img_matrice_x[y, x]) #extract data from matrix
            y2 = int(img_matrice_y[y, x])

            if x2 != -1:
                try:
                    R1, G1, B1 = pixels[x2, y2]  # Get deviated pixel color
                    pixels2[x, y] = (R1, G1, B1) # Colorize seen pixel with deviated pixel color
                except:
                    pixels2[x, y] = (255, 0, 0)  # Optional (colorize pixels out of picture)
                    ""  # if don't want to colorize, comment the line above
            else:
#                pixels2[x, y] = (0, 0, 0)  # Optional (colorize the black hole instead of alpha)
                ""  # if don't want to colorize, comment the line above

#    folder, file_name, extension = return_folder_file_extension(image_name)
#    img2.save(file_name+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+".png") #use it instead if use high contrasted image
    img2 = img2.convert("RGB") # Warning, if disable -> now with manipulable graph, must add a del command to del previous image inside black hole
    return img2


def resize_img(img_original, final_size_img):
    global left_side, right_side, up_side, down_side, img_res, img_res_Y, axe_X, axe_Y, FOV_img_Y, FOV_img, offset_X

    img_original = img_original.convert("RGB")  #if needed RGBA, change all "pixels" ligne by RGBA instead of RGB
    axe_X = img_original.size[0]
    axe_Y = img_original.size[1]
    size_factor = final_size_img/axe_X
    axe_X = int(size_factor*axe_X)
    axe_Y = int(size_factor*axe_Y)

    if axe_X % 2 != 0:   #even dimensions needed for image (don't rember why but error if not)
        axe_X -= 1

    if axe_Y % 2 != 0:
        axe_Y -= 1

    final_size_img = axe_X
    img_original = img_original.resize((axe_X, axe_Y),Image.ANTIALIAS) #can be avoid if put axe_X before resize with conditions
    FOV_img_Y = FOV_img*axe_Y/axe_X

    img_original = img_offset_X(img_original, int(offset_X))

    if FOV_img_Y>180:
        raise StopIteration ("Can't have a FOV>360 in the Y-axis")

    print('\nsize ',axe_X,"*",axe_Y," pixels",sep='')
    img_res = axe_X/360  #=Pixels per degree along axis
    img_res_Y = axe_Y/180  #=Pixels per degree along axis
    left_side = 0
    right_side = axe_X #must be integer
    up_side = 0
    down_side = axe_Y
#    left_side = int(axe_X/2.5)
#    right_side = axe_X-int(axe_X/2.5) #must be integer
#    up_side = int(axe_Y/2.5)
#    down_side = axe_Y-int(axe_Y/2.5)
    return img_original


def check_matrices():
    global img_matrice_x, img_matrice_y, axe_X, axe_Y

    abs_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    dir_path = os.path.join(abs_path, 'matrix')

    matrix_name_x = str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_x.txt'
    matrix_file_x = os.path.join(dir_path, matrix_name_x)
    x_file = listdirectory2(dir_path, matrix_file_x)
    matrix_name_y = str(D)+'_'+str(Rs)+'_'+str(final_size_img)+'_'+str(FOV_img)+'_y.txt'
    matrix_file_y = os.path.join(dir_path, matrix_name_y)
    y_file = listdirectory2(dir_path, matrix_file_y)

    if use_matrix is True and x_file is True and y_file is True:
        print("\nmatrix opening estimation:", round(1.65*10**(-6)*axe_X*axe_Y, 1))
        matrix_opening_debut = time.process_time()
        img_matrice_x = np.array([-1]*axe_X)
        img_matrice_y = np.array([-1]*axe_Y)
        img_matrice_x = np.loadtxt(matrix_file_x) #to be rigorous, add final_size_img_Y, FOV_img, FOV_img_Y, kind..
        img_matrice_y = np.loadtxt(matrix_file_y)
        matrix_opening_fin = time.process_time()
        print("matrix opening time:",round(matrix_opening_fin-matrix_opening_debut, 1))

    else:
        img_matrice_x, img_matrice_y = matrices_creation() #create a matrix with corresponding (x, y) -> (x2, y2)

        if save_matrix is True:
            np.savetxt(matrix_file_x, img_matrice_x, fmt='%i')
            np.savetxt(matrix_file_y, img_matrice_y, fmt='%i')


def gif(nbr_offset):
    global img_debut, img2, offset_X_tot, offset_X, offset_X2, image_name
    folder, file_name, extension = return_folder_file_extension(image_name)

    offset_X_temp = 0 #locals, relative to img2 given, not absolute
    offset_X_tot = 0
    print("\ntotal offsets estimation time:",round(1.6*10**-6*axe_X*axe_Y*(nbr_offset+1),1))

    for a in prange(nbr_offset+1):

        if a < nbr_offset:
            print("\n",a+1,"/",nbr_offset,"\toffset_X: ",offset_X_tot, sep='')

        img_debut = img_offset_X(img_debut, offset_X_temp)
        img2 = Image.new('RGBA', (axe_X, axe_Y)) #creat a transparent image (outside of the function to recover info if must stop loop)
        debut = time.process_time()
        img2 = img_pixels(img_debut, img2)

        if fixed_background != True and fixed_background != False:

            if fixed_background.get() is True:
                img2 = img_offset_X(img2,-offset_X_tot)  # if want a fixed background and moving black hole

        if fixed_background is True:
            img2 = img_offset_X(img2,-offset_X_tot)  # if want a fixed background and moving black hole

        if nbr_offset != 1 and a<nbr_offset: #if need to save real offset, put offset_x in global and offset_x+offset_x2+offset_x_tot in save name
            img2.save(file_name+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+" offset_X="+str(int(offset_X_tot+offset_X+offset_X2))+extension)

        fin = time.process_time()

        if a < nbr_offset:
            print("pixels time :", round(fin-debut, 1))
        else:
            print("correction time :", round(fin-debut, 1))

        offset_X_temp = int(axe_X/nbr_offset) #at the end to have offset=0 for the first iteration
        offset_X_tot += offset_X_temp


def black_hole():
    global left_side, right_side, up_side, down_side, img_debut, D, Rs, deviated_angle_splin, seen_angle_splin, img_res, seen_angle, deviated_angle, offset_X_tot, FOV, FOV_img, last_angle

    vrai_debut = time.process_time()
    left_side_temp = int(left_side)
    right_side_temp = int(right_side)  # Must be integer
    up_side_temp = int(up_side)
    down_side_temp = int(down_side)
    left_side = 0
    right_side = int(axe_X)  # Must be integer
    up_side = 0
    down_side = int(axe_Y)
    seen_angle, deviated_angle = trajectories(D, FOV/2, img_res, Rs)  #call funtion trajectories to compute angles
    seen_angle_splin, deviated_angle_splin = interpolate(seen_angle, deviated_angle)  #to display the interpolation( not needed for commpute)

    if display_interpolation is True:
        plt.figure(num='Trajectories')
        plt.clf() #clear the graph to avoir superposing data from the same set (can be deactivated if need to superpose)
#        ax3 = plt.subplot(111) #not needed for computation
        plt.ylabel('deviated angle(°)')
        plt.xlabel('seen angle(°)')
        plt.title("Light deviation interpolation", va='bottom')
        plt.plot(seen_angle, deviated_angle,'o')
        plt.plot(seen_angle_splin, deviated_angle_splin)
        #plt.xlim((100, 160))
        #plt.ylim((120, 350))
        #plt.savefig('interpolation.png', format='png', dpi=1000, bbox_inches='tight')
        #plt.savefig('interpolation.eps', format='eps', dpi=1000, bbox_inches='tight')
        plt.draw()
#        plt.close()

    last_angle=seen_angle[-1]
    print("last angle",last_angle)

    check_matrices()

#    img_debut=img_offset_X(img_debut, offset_X_tot)  #allow to begin with an offset then carry on with diff offsets (useful if stop prematuraly and want to start back in the middle)

    gif(nbr_offset)

    left_side = left_side_temp
    right_side = right_side_temp
    up_side = up_side_temp
    down_side = down_side_temp
    vrai_fin = time.process_time()
    print("\nglobal computing time:",round(vrai_fin-vrai_debut, 1))


def open_image(image_name):
    img_original = Image.open(image_name, mode='r')  # Use equirectangular image
    img_debut = resize_img(img_original, final_size_img)
    print("FOV ",FOV_img,"*",FOV_img_Y,"°\n",sep='')
    return [img_original, img_debut]


def plot_black_hole():
    global img, img2, cid, ax, fig
    fig, ax = plt.subplots()  # Alternative to img.show
    img = ax.imshow(img2)
    fig.canvas.set_window_title('Black hole')
    ax.set_title("scrool to zoom in or out \nright click to add an offset in the background \nleft click to refresh image \n close the option windows to stop the program")
    fig.canvas.mpl_connect('scroll_event', onscroll)  #listen to events
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('axes_leave_event', disconnect)
    fig.canvas.mpl_connect('axes_enter_event', connect)


zoom = 0
offset_X2 = 0
def onscroll(event): #function that listens to scroll event
    global offset_X, offset_X2, img, left_side, right_side, up_side, down_side, fig, ax, zoom, final_size_img, img2, img_matrice_x, img_matrice_y, img_debut
    if out_graph is False:
        zoom += 10*event.step
        if zoom >= axe_X/2/FOV_img*FOV_img_Y:
            zoom = axe_X/2/FOV_img*FOV_img_Y
        if zoom <= 0:
            zoom = 0
        left_side = int(zoom*FOV_img/FOV_img_Y)
        right_side = int(axe_X-zoom*FOV_img/FOV_img_Y) #must be integer
        up_side = int(zoom)
        down_side = int(axe_Y-zoom)
        ax.set_xlim((left_side, right_side-1))
        ax.set_ylim((down_side-1, up_side))
        fig.canvas.draw()


def onclick(event):  # function that listens to click event
    global fig, ax, img, img2, offset_X, out_graph, offset_X2, axe_X, img_debut, left_side, right_side, down_side, right_side
    if out_graph is False:
        if event.button == 1:
            ax.set_xlim((left_side, right_side-1))
            ax.set_ylim((down_side-1, up_side))
        if event.button == 3 and event.xdata >= 0 and event.xdata <= axe_X:
            offset_X += offset_X2
            offset_X2 = axe_X/2-event.xdata-offset_X
            img_debut = img_offset_X(img_debut, int(offset_X2))
        img2 = Image.new( 'RGBA', (axe_X, axe_Y))
        img2 = img_pixels(img_debut, img2)
        img = ax.imshow(img2)
        fig.canvas.draw()
        if event.button == 2:
            print("do you think this should do someting ?")


def disconnect(event):  # function that listens to click event
    global cid, out_graph
    out_graph = True


def connect(event):  # function that listens to click event
    global cid, out_graph
    out_graph = False


def compute():
    global D, Rs, img, img2, final_size_img
    message5["text"] = ""
    message4["text"] = ""
    message2["text"] = ""
    message["text"] = ""
    # print("temp", final_size_img, size.get())
    # print(Rs, radius.get())
    # print(D, distance.get())
    # final_size_img = int(size.get())
    # Rs = float(radius.get())
    # D = float(distance.get())
    try:
        if float(distance.get()) <= 0 or float(radius.get()) <= 0:
            message["text"] = "Can't be 0 or negative"
        elif float(distance.get()) == D and float(radius.get()) == Rs and int(size.get()) == final_size_img:
            message["text"] = "same values as before"
        elif float(distance.get()) < float(radius.get()):
            message["text"] = "Inside black hole !"
        else:
            D = float(distance.get())
            Rs = float(radius.get())
            try:
                if final_size_img != int(size.get()):
                    increase_resolution()
            except ValueError as ex:
                print(ex)
                return

            message["text"] = "Computing"
            # final_size_img = int(size.get())
            M = Rs*c**2/2/G*Ds/Ms  #Black hole mass in solar masses  (del if use solar mass)
            print ("M ","%.1e"%M,"M☉","\t%.2e"%(M*Ms),"Kg")
            print("Rs",Rs,"ua","\t%.2e"%(Rs*Ds),"m")
            print("D ",D,"ua","\t%.2e"%(D*Ds),"m")
            black_hole()
            img = ax.imshow(img2)
            ax.set_xlim((left_side, right_side-1))
            ax.set_ylim((down_side-1, up_side))
            fig.canvas.draw()
            message["text"] = "Done !"
    except ValueError:
        message["text"] = "Radius, distance"
        message5["text"] = "& image size are floats"


def increase_resolution():
    global offset_X, offset_X2, img, left_side, right_side, up_side, down_side, fig, ax, zoom, final_size_img, img2, img_debut
#    if 'img' not in globals():    #don't work, want to check if graph is close to creat a new one
#        print("graphique fermé")
    # try:
    if float(size.get()) <= 0:
        message2["text"] = "Can't be 0 or negative"
        raise ValueError ("Can't be 0 or negative")
    elif float(size.get()) == final_size_img:
        message2["text"] = "same size as before"
    else:
        # message2["text"] = "Computing"
        new_size_image = int(size.get())
        offset_X += offset_X2
        offset_X2 = 0
        res_fact = new_size_image/final_size_img

        final_size_img = new_size_image
        if int(left_side*res_fact) != int(right_side*res_fact-1):
            ax.set_xlim((int(left_side*res_fact), int(right_side*res_fact-1)))
        if int(down_side*res_fact-1) != int(up_side*res_fact):
            ax.set_ylim((int(down_side*res_fact-1), int(up_side*res_fact)))
        offset_X *= res_fact
        zoom *= res_fact
        try:
            img_debut = resize_img(img_original, final_size_img)
        except ValueError as ex:
            print("error when resizing image")
            raise ValueError (ex)


def save_file():  # function that listens to click event
    global img2, offset_X, offset_X2, image_name
    folder, file_name, extension = return_folder_file_extension(image_name)
    image_name_save = file_name+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+" offset_X="+str(int(offset_X+offset_X2))+extension
    img2.save(image_name_save)
    print("Save: "+image_name_save)
    message4["text"] = "Save: "+image_name_save


def save_gif():  # function that listens to click event
    global img2, offset_X, offset_X2, image_name
    folder, file_name, extension = return_folder_file_extension(image_name)
    print("Computing")
    message3["text"] = "Computing"

    try:
        if int(number.get()) <= 0:
            print("Can't be 0 or negative")
            message3["text"] = "Can't be 0 or negative"
        else:
            gif(int(number.get()))
            message3["text"] = "Save: "+file_name+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+" offset_X=*"+extension
    except:
        print("need integer")
        message3["text"] = "need integer"


def open_file_name():
    """Source: https://gist.github.com/Yagisanatode/0d1baad4e3a871587ab1
    Not adapted for functions but classes"""
    global image_name, extension, img_original, img_debut, img, img2, ax, fig

    nom_image_temp = askopenfilename(
#        initialdir="",
        filetypes =[("Image File", ".png .jpg")],
        title = "Image file")

    if nom_image_temp:
        image_name = nom_image_temp
        print("Openning:", image_name)
        [img_original, img_debut] = open_image(image_name)
#        [img_original, img_debut] = open_image(image_name)
        img2=Image.new( 'RGBA', (axe_X, axe_Y)) #creat a transparent image (outside of the function to recover info if must stop loop)
        img2=img_pixels(img_debut, img2)
        img=ax.imshow(img2)
        fig.canvas.draw()
        print("Done !")
        message2["text"] = "Done !"
#    else:
#        print("cancel")


root = Tk()
frame = Frame(root)
root.title("Black hole options")
frame.pack()
#bottomframe = Frame(root)  #si bas diff du haut (inbrique plusieurs widgets)
#bottomframe.pack( side = "bottom" )

open_file_button = Button(frame, text="Open image", width=14, command=open_file_name)
open_file_button.grid(row=0, column=0)

L1 = Label(frame, text="Radius")
L1.grid(row=1, column=0)
var = StringVar(root)
var.set(Rs)
radius = Spinbox(frame, from_=1e-100, to=1e100, textvariable=var, bd=2, width=7)
radius.grid(row=1, column=1)

L2 = Label(frame, text="Distance")
L2.grid(row=2, column=0)
var = StringVar(root)
var.set(D)
distance = Spinbox(frame, from_=1e-100, to=1e100, textvariable=var, bd=2, width=7)
distance.grid(row=2, column=1)

compute_button = Button(frame, text="Compute", width=14, command=compute)
compute_button.grid(row=1, column=2)

message = Label(frame, text="", width=20)   #allow to display message when activate [text]
message.grid(row=1, column=3)
message5 = Label(frame, text="", width=20)   #allow to display message when activate [text]
message5.grid(row=2, column=3)

L3 = Label(frame, text="Image size")
L3.grid(row=3, column=0)
var = StringVar(root)
var.set(final_size_img)
size = Spinbox(frame, from_=1, to=1e100, textvariable=var, bd=2, width=7)
size.grid(row=3, column=1)

message2 = Label(frame, text="", width=20)   #allow to display message when activate [text]
message2.grid(row=3, column=3)

save_button = Button(frame, text="Save image", width=14, command=save_file)
save_button.grid(row=4, column=2)

message4 = Label(frame, text="")   #allow to display message when activate [text]
message4.grid(row=4, column=3)

message6 = Label(frame, text="Fix background")   #allow to display message when activate [text]
message6.grid(row=5, column=0)

fixed_background = BooleanVar()
C1 = Checkbutton(frame, text = "", variable = fixed_background, \
                 onvalue = True, offvalue = False)
C1.grid(row=5, column=1)

L4 = Label(frame, text="images")
L4.grid(row=6, column=0)

var = StringVar(root)
var.set(10)
number = Spinbox(frame, from_=1, to=1e100, textvariable=var, bd=2, width=7)
number.grid(row=6, column=1)

save_gif_button = Button(frame, text="Save animation", width=14, command=save_gif)
save_gif_button.grid(row=6, column=2)

message3 = Label(frame, text="")   #allow to display message when activate [text]
message3.grid(row=6, column=3)


# =============================================================================
# Beginning of the programs
# =============================================================================
image_name = os.path.join('images', 'default.png')
[img_original, img_debut] = open_image(image_name)

black_hole()

plot_black_hole()

root.mainloop()
# raise SystemExit  #not needed and too brutal

# """ temp"""

# # increase_resolution()

# M = Rs*c**2/2/G*Ds/Ms  #Black hole mass in solar masses  (del if use solar mass)

# Rs=8.0
# D_list = np.round(10**np.linspace(np.log10(50), np.log10(100000), 30))

# for D in D_list:
#     black_hole()

#     image_name = "/home/jonathan/temp_perso/milkyway.jpg"
#     folder, file_name, extension = return_folder_file_extension(image_name)
#     image_name_save = file_name+" D="+str(D)+" Rs="+str(Rs)+" size="+str(final_size_img)+" offset_X="+str(int(offset_X+offset_X2))+extension
#     img2.save(image_name_save)
#     print("Save: "+image_name_save)

# # """mesure speed"""
# # speed = [50.0, 51.5, 53.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, 125.0, 150.0,
# #          175.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0,
# #          1000.0, 1500.0, 2000.0, 3000.0, 3500.0, 5000.0, 7000.0, 10000.0,
# #          20000.0, 50000.0, 100000.0
# #          ]
# # plt.plot(speed)



# # plt.plot(speed)