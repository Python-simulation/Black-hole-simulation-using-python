# -*- coding: utf-8 -*-
"""
Black hole simulation

@author: Jonathan Peltier

GitHub repository:
https://github.com/Python-simulation/Black-hole-simulation-using-python/

BlackHole class solving photons trajectories closed to a static black hole.
Render the perceived image deformation by a black hole.
Object-oriented programming version.
Numpy optimized version 30x faster.
"""

import sys
import math  # import known fonctions and constants
import time  # Used to check computation time
import os.path  # Used to search files on computer
from tkinter import Tk, Frame
from tkinter import Button, Label, Checkbutton, BooleanVar, StringVar, Spinbox
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt  # Graphical module
import matplotlib
if matplotlib.get_backend() not in ("TKAgg", "Qt5Agg"):
    matplotlib.use("TKAgg", force=True)
#from matplotlib.widgets import Slider  # TODO: use it for offset GUI
import numpy as np  # Use for matrices and list
from scipy.interpolate import interp1d  # Use for interpolation
from scipy.integrate import solve_ivp  # Integrate ord diff eqs
from scipy.constants import pi  # 3.141592653589793
from scipy.constants import c  # Speed light vaccum = 299792458 m/s
from scipy.constants import G  # Newton constant = 6.67408e-11 m3/Kg/s2
from scipy.constants import au  # sun-earth distance m = 149597870691 m
from PIL import Image  # Use to open, modify and save images
from PIL import ImageDraw

M_sun = 1.98840987e+30  # solar mass in Kg taken from AstroPy


class BlackHole:
    """Main class"""
    def __init__(self):
        """Main class"""
        self.init_var()
        plt.ion()

        try:
            abs_path = os.path.abspath(os.path.dirname(sys.argv[0]))
            folder = os.path.join(abs_path, 'images')
            img_name = os.path.join(folder, 'milkyway.jpg')
            self.open(img_name, size=self.axe_X)

        except FileNotFoundError:
            print("milkyway image not found, creating a default image")
            self.create_default_image(size=self.axe_X)

    def init_var(self):
        """Initialize most variables."""
        # TODO: allow both M and Rs definition
        #M = 1.7342*10**22/M_sun  # Black hole mass in solar mass (alternative, del M below)
        #Rs = 2*G*M_sun*M/c**2/Ds  # Schwarzschild radius in Astronomical unit (ua)
        self.Rs = 8  # Schwarzschild radius in ua
        self.M = self.Rs * c**2 / 2 / G * au / M_sun  # Black hole mass in solar masses  (del if use solar mass)
        self.D = 50  # Distance from the black hole in ua
        self.axe_X = 360  # Image size over x
        self.FOV_img = 360  # The image FOV (it doesn't change the current image FOV !)

        self.kind = 'cubic'  # Interpolation: linear for speed(less accurate), cubic for precision (slow)
        self.fixed_background = True
        self.display_trajectories = True
        self.display_interpolation = True
        self.display_blackhole = True
        # Note that openning matrices is slower than computing the blackhole,
        # but skip trajectories calculation than takes 1.5s -> better to open
        # matricies at low resolution but better to compute at high resolution
        self.use_matrix = True  # Use matrices if exists  # TODO: GUI option
        self.save_matrix = False  # will save or overwrite matrices if exists

        self.zoom = 0
        self.offset_X = 0
        self.offset_X2 = 0
        self.out_graph = False
        #----------------------------------------------------------------------
        self.img_matrix_x = None
        self.img_matrix_y = None
        self.img2 = None
        self.ax = None
        #----------------------------------------------------------------------
        # if want matricies without images loaded
#        self.axe_Y = self.axe_X//2
#        self.img_res = self.axe_X/360  # =Pixels per degree along axis
#        self.img_res_Y = self.axe_Y/180  # =Pixels per degree along axis
#        self.FOV_img_Y = self.FOV_img//2
#        self.img_debut = None

    def create_default_image(self, size="default", pattern="grid"):
        """Create a default image if doesn't want to import one."""
        if size == "default":
            size = self.axe_X

        abs_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        folder = os.path.join(abs_path, 'images')
        self.img_name = os.path.join(folder, 'default.png')

        axe_X = 1000  # int(self.axe_X)
        axe_Y = 500  # self.axe_X//2

        if pattern == "noise":
            pixels = np.random.randint(0, 255, (axe_Y, axe_X, 3))
            self.img_original = Image.fromarray(pixels.astype('uint8'), 'RGB')

        elif pattern in ("grid", "cercle", "rectangle"):
            self.img_original = Image.new('RGB', (axe_X, axe_Y), color=255)
            Drawer = ImageDraw.Draw(self.img_original)
            Drawer.rectangle((0, 0, axe_X/2, axe_Y/2), fill="yellow")
            Drawer.rectangle((0, axe_Y/2, axe_X/2, axe_Y), fill="green")
            Drawer.rectangle((axe_Y, 0, axe_X, axe_Y/2), fill="blue")
            Drawer.rectangle((axe_Y, axe_Y/2, axe_X, axe_Y), fill="red")

            nbr_rect = 40
            if pattern == "grid":
                for i in range(0, axe_X, axe_X//nbr_rect):
                    Drawer.line((i, 0, i, axe_Y), fill="black", width=2)
                for i in range(0, axe_Y, axe_Y//(nbr_rect//2)):
                    Drawer.line((0, i, axe_X, i), fill="black", width=2)
            else:
                for i in range(0, axe_X, axe_X//nbr_rect):

                    if pattern == "cercle":
                        Drawer.ellipse((i, i/2, axe_X-i, axe_Y-i/2), outline="black")
                    elif pattern == "rectangle":
                        Drawer.rectangle((i, i/2, axe_X-i, axe_Y-i/2), outline="black")

        else:
            raise ValueError("pattern parameter must be: grid, noise, cercle or rectangle")

        self.img_debut = self.img_resize(size)
        return self.img_debut

    def open(self, img_name, size="default"):
        """Open an equirectangular image.
        Can resize it with the size option.
        """
        print("Openning %s" % img_name)
        self.img_original = Image.open(img_name, mode='r')
        self.img_name = img_name

        if size == "default":
            size = self.img_original.size[0]

        self.img_debut = self.img_resize(size)
        return self.img_debut

    def img_resize(self, axe_X):
        """Create img_debut at the desired size from the img_original."""
        self.img_debut = self.img_original.convert("RGB")
        size_X, size_Y = self.img_debut.size
        size_factor = axe_X/size_X
        axe_X = int(axe_X)
        axe_Y = int(size_factor*size_Y)

        # even dimensions needed for image (error if not)
        if axe_X % 2 != 0:
            axe_X -= 1

        if axe_Y % 2 != 0:
            axe_Y -= 1

        self.img_debut = self.img_debut.resize((axe_X, axe_Y), Image.Resampling.LANCZOS)
        self.FOV_img_Y = self.FOV_img * axe_Y / axe_X

        if self.FOV_img_Y > 180:
            raise StopIteration("Can't have a FOV>180 in the Y-axis")

        print("size %sx%s pixels\n" % (axe_X, axe_Y))
        self.img_res = axe_X/360  # =Pixels per degree along axis
        self.img_res_Y = axe_Y/180  # =Pixels per degree along axis

        self.axe_X, self.axe_Y = axe_X, axe_Y

        return self.img_debut

    def compute(self, Rs, D):
        """main method used to compute the black hole deformation and apply it
        on a image."""
        self.Rs = Rs
        self.D = D
        self.M = (self.Rs * c**2 * au) / (2 * G * M_sun)
        print("M = %.1e M☉\t%.2e Kg" % (self.M, self.M*M_sun))
        print("Rs = %s ua\t%.2e m" % (self.Rs, self.Rs*au))
        print("D = %s ua\t%.2e m\n" % (self.D, self.D*au))

        vrai_debut = time.process_time()

        if self.use_matrix and self.check_matrices():
            img_matrix_x, img_matrix_y = self.open_matrices()
            plt.close('Trajectories interpolation')
            plt.close('Trajectories plan')

        else:
            seen_angle, deviated_angle = self.trajectories()

            self.interpolation = self.interpolate(seen_angle, deviated_angle)

            if self.display_interpolation is True:
                xmin = np.min(seen_angle)
                xmax = np.max(seen_angle)
                seen_angle_splin = np.linspace(xmin, xmax, 20001)
                deviated_angle_splin = self.interpolation(seen_angle_splin)
                plt.figure('Trajectories interpolation')
                plt.clf()
                plt.title("Light deviation interpolation", va='bottom')
                plt.xlabel('seen angle(°)')
                plt.ylabel('deviated angle(°)')
                plt.plot(seen_angle, deviated_angle, 'o')
                plt.plot(seen_angle_splin, deviated_angle_splin)
                plt.grid()
                #plt.savefig('interpolation.png', dpi=250, bbox_inches='tight')
                plt.draw()
    #
            print("last angle", seen_angle[-1])
            print("trajectories time: %.1f" % (time.process_time()-vrai_debut))

            img_matrix_x, img_matrix_y = self.create_matrices()

            if self.save_matrix is True:
                matrix_file_x, matrix_file_y = self.matrices_names()
                np.savetxt(matrix_file_x, img_matrix_x, fmt='%i')
                np.savetxt(matrix_file_y, img_matrix_y, fmt='%i')
                print("Saved matrices: \n\t%s\n\t%s" % (matrix_file_x, matrix_file_y))

        self.img_matrix_x = img_matrix_x
        self.img_matrix_y = img_matrix_y

        self.img2 = self.img_pixels(self.img_debut)

        vrai_fin = time.process_time()
        print("\nglobal computing time: %.1f\n" % (vrai_fin-vrai_debut))

        if self.display_blackhole:
            self.plot()

    def trajectories(self):
        """Compute several photons trajectories in order to interpolate the
        possibles trajectories and gain in execution time."""
        # OPTIMIZE: take too much time due to too much solver call
        alpha_min = self.search_alpha_min()
        alpha_finder = self.FOV_img/2

        if self.display_trajectories is True:
            plt.figure('Trajectories plan')
            plt.clf() #clear the graph to avoir superposing data from the same set (can be deactivated if need to superpose)
            ax = plt.subplot(111, projection='polar') #warning if use python in ligne (!= graphical) graphs got superposed
            ax.set_title("light trajectories close to a black hole\n", va='bottom')
            ax.set_xlabel('R(UA)')
            plt.ylabel('phi(°)\n\n\n\n', rotation=0)
            ax.set_rlim((0, 4*self.D))
            ax.set_rlabel_position(-90)

        seen_angle = np.array([])
        deviated_angle = np.array([])

        booli = False  # avoid points from the first loop to exceed points from the second loop
        points = 40  # careful with this if using kind=linear

        for i in range(6):
    #        print(alpha_finder)

            for alpha in np.linspace(alpha_finder, alpha_min,
                                     num=points, endpoint=booli):
                r, phi = self.solver(alpha)

                if r[-1] > 1.1*self.Rs:  # if not capture by black hole
                    seen_angle = np.append(seen_angle, 180-alpha)
                    dev_angle = phi[-1] + math.asin(self.D/r[-1]*math.sin(phi[-1]))
                    dev_angle = math.degrees(dev_angle)
                    deviated_angle = np.append(deviated_angle, dev_angle)
                    Ci = 'C'+str(i)

                    if self.display_trajectories is True:
                        ax.plot(phi, r, Ci)  # plot one trajectory

            if self.kind == 'linear':
                alpha_finder = alpha_min + (alpha_finder - alpha_min)/(points/3 + 1) # start a more precise cycle from last point

            else:
                alpha_finder = alpha_min + (alpha_finder - alpha_min)/(points + 1) # start a more precise cycle from last point

            points = 10  # careful with this if using kind=linear

            if i == 4:
                booli = True  # allow to display the last point

        if self.display_trajectories is True:
    #        plt.savefig('trajectories.png', format='png', dpi=1000, bbox_inches='tight')
            plt.draw()

        return seen_angle, deviated_angle

    def search_alpha_min(self):
        """Return last angle at which the photon is kept by the black hole."""
        alpha_min = 0

        for alpha in range(0, 180, 4):
            r = self.solver(alpha)[0]
            if r[-1] > 1.1*self.Rs:
                break

        if (alpha-4) > 0:
            alpha_min = alpha - 4
    #        print("alpha_min :",alpha_min,"(-4)")
        i = 1

        while alpha_min == 0 or round(alpha_min*self.img_res) != round((alpha_min+i*10)*self.img_res):  #increase precision

            for alpha in range(int(alpha_min/i), int(180/i), 1):
                alpha = alpha*i
                r = self.solver(alpha)[0]

                if r[-1] > 1.1*self.Rs:
                    break

            if (alpha-i) > 0:
                alpha_min = alpha - i
    #            print("alpha_min : ",alpha_min," (-",i,")",sep="")

            i = i/10
        i = 10*i
        alpha_min += i
        print("alpha_min: %s [%s, %s]" % (alpha_min, alpha_min-i, alpha_min))

        return alpha_min

    def solver(self, alpha):
        """Solve the differential equation, in spherical coordinate, for a
        static black hole using solve_ivp.
        Allows to compute the photon trajectory giving its distance from the
        black hole and its initial angular speed."""
        if alpha == 0:  # skip divided by 0 error
            return [0], [0]  # r and phi=0

        if alpha == 180:
            return [self.D], [0]  # if angle= pi then, tan(pi)=0 so 1/tan=1/0

        # initial value for position and angular speed
        y0 = [1/self.D, 1/(self.D*math.tan(math.radians(alpha)))]
        sol = solve_ivp(fun=self._diff_eq, t_span=[0, 10*pi], y0=y0, method='Radau', events=[self._eventRs]) #, self._eventR])#,t_eval=np.linspace(0, t_max, 10000)) #dense_output=False

        if sol.t[-1] == 10*pi:
            raise StopIteration("solver error, alpha reached computation limit (loop number)")

        phi = np.array(sol.t)
        r = np.abs(1/sol.y[0, :])  # must use this because solver can't be stop before infinity because negative

        return r, phi

    def _diff_eq(self, phi, u):
        """Represent the differential equation : d²u(ɸ)/dɸ²=3/2*Rs*u²(ɸ)-u(ɸ)
        """
        v0 = u[1]  #correspond to u'
        v1 = 3/2*self.Rs*u[0]**2 - u[0] #correspond to u"
        return v0, v1

    def _eventRs(self, phi, u):
        """stop solver if radius < black hole radius"""
        with np.errstate(all='ignore'):
            return 1/u[0] - self.Rs
    _eventRs.terminal = True

#    def _eventR(self, phi, u): #not needed and don't work with ivp (without it we get an error message but irrelevant)
#        """stop solver if radius > sphere limit"""
#        R = 1e15
#        return (1/u[0]-math.sqrt(R**2-self.D**2*math.sin(phi)**2)+self.D*math.cos(phi))
#    _eventR.terminal = True

    def interpolate(self, x_pivot, f_pivot):
        """Create interpolation data to reduce computation time."""
        interpolation = interp1d(x_pivot, f_pivot,
                                 kind=self.kind, bounds_error=False)
        return interpolation

    def matrices_names(self, folder=None):
        """Return the matrices names."""
        if folder is None:
            abs_path = os.path.abspath(os.path.dirname(sys.argv[0]))
            folder = os.path.join(abs_path, 'matrix')

        matrix_name_x = "%s_%s_%s_%s_x.txt" % (
            self.D, self.Rs, self.axe_X, self.FOV_img)
        matrix_file_x = os.path.join(folder, matrix_name_x)

        matrix_name_y = "%s_%s_%s_%s_y.txt" % (
            self.D, self.Rs, self.axe_X, self.FOV_img)
        matrix_file_y = os.path.join(folder, matrix_name_y)

        return matrix_file_x, matrix_file_y

    def check_matrices(self, folder=None):
        """Check if matricess exists."""
        if folder is None:
            abs_path = os.path.abspath(os.path.dirname(sys.argv[0]))
            folder = os.path.join(abs_path, 'matrix')

        matrix_file_x, matrix_file_y = self.matrices_names(folder=folder)

        x_file = listdirectory(folder, matrix_file_x)
        y_file = listdirectory(folder, matrix_file_y)

        matrices_exist = x_file is True and y_file is True
        return matrices_exist

    def create_matrices(self):
        """Call find_position function and create matrices with pixels
        positions informations.
        Create two matrices with corresponding (x, y) -> (x2, y2).
        """
        debut = time.process_time()

        x = np.arange(0, self.axe_X)
        y = np.arange(0, self.axe_Y)
        xv, yv = np.meshgrid(x, y)

        print("\nmatrix creation estimation time: %.1fs" % (1e-6*self.axe_X*self.axe_Y))

        img_matrix_x, img_matrix_y = self.find_position(xv, yv)

        fin = time.process_time()
        print("matrix created in time: %.1f s" % (fin-debut))
        return img_matrix_x, img_matrix_y

    def open_matrices(self):
        """Open the matricies corresponding to the chosen Rs, D, axe_X and FOV
        parameters."""
        print("\nmatrix opening estimation: %.1f" % (
            1.65e-6*self.axe_X*self.axe_Y))
        matrix_opening_debut = time.process_time()

        matrix_file_x, matrix_file_y = self.matrices_names()
        img_matrix_x = np.loadtxt(matrix_file_x, dtype=int)
        img_matrix_y = np.loadtxt(matrix_file_y, dtype=int)
        matrix_opening_fin = time.process_time()
        print("matrix opening time:", round(matrix_opening_fin-matrix_opening_debut, 1))
        return img_matrix_x, img_matrix_y

    def find_position(self, xv, yv):
        """Takes seen pixel position and search deviated pixel position."""
        # Convert position in spheric coord
        phi = xv*self.FOV_img/360/self.img_res
        theta = yv*self.FOV_img_Y/180/self.img_res_Y
        phi2 = phi+(360-self.FOV_img)/2
        theta2 = theta+(180-self.FOV_img_Y)/2

        u, v, w = spheric2cart(np.radians(theta2), np.radians(phi2))  # give cartesian coord of pixel

        # ignore errors due to /0 -> inf, -inf
        # divide (w/v) and invalid arctan2()
        with np.errstate(all='ignore'):  # OPTIMIZE: see comment about pi = -pi and don't matter if -0 or 0 -> just replace by pi
            beta = -np.arctan(w/v)
#        beta2 = -np.arctan2(w, v)

#        v2 = np.dot(rotation_matrix(beta), [u, v, w]) # take 3*3 created matrix and aplly to vector
        matrix = rotation_matrix(beta)
        u2 = matrix[0, 0]*u
        v2 = matrix[1, 1]*v+matrix[1, 2]*w
        w2 = matrix[2, 1]*v+matrix[2, 2]*w
        _, seen_angle = cart2spheric(u2, v2, w2)    # return phi in equator "projection"

        seen_angle = np.degrees(seen_angle)
        seen_angle = np.mod(seen_angle, 360)  # define phi [0, 360]

#        seen_angle[seen_angle > 360] -= 360
        deviated_angle = np.zeros(seen_angle.shape)
        deviated_angle[seen_angle < 180] = self.interpolation(seen_angle[seen_angle < 180])
        deviated_angle[seen_angle >= 180] = 360 - self.interpolation(360-seen_angle[seen_angle >= 180])
#        np.flip(deviated_angle, 1)  " mais probleme overlap entre left et right

        theta = pi/2# *np.ones(deviated_angle.shape)
        phi = np.radians(deviated_angle)
        u3, v3, w3 = spheric2cart(theta, phi) #get cart coord of deviated pixel

        matrix = rotation_matrix(-beta)
        u4 = matrix[0, 0]*u3
        v4 = matrix[1, 1]*v3+matrix[1, 2]*w3
        w4 = matrix[2, 1]*v3+matrix[2, 2]*w3

        theta, phi = cart2spheric(u4, v4, w4)  #give spheric coord of deviated pixel

        theta, phi = np.degrees(theta), np.degrees(phi)

        phi -= (360-self.FOV_img)/2
        theta -= (180-self.FOV_img_Y)/2

        with np.errstate(all='ignore'):  # OPTIMIZE
            phi = np.mod(phi, 360)  # define phi [0, 360]
            theta = np.mod(theta, 180)  # define phi [0, 360]

        phi[phi == 360] = 0
        xv2 = phi*360/self.FOV_img*self.img_res
        yv2 = theta*180/self.FOV_img_Y*self.img_res_Y #give deviated angle pixel position

        xv2[np.isnan(xv2)] = -1
        yv2[np.isnan(yv2)] = -1

        xv2 = np.array(xv2, dtype=int)
        yv2 = np.array(yv2, dtype=int)

        return xv2, yv2

    def gif(self, nbr_offset=1):
        """Apply seveal offset and save each images to be reconstructed
        externaly to make agif animation of a moving black hole."""
        file_name, extension = return_folder_file_extension(self.img_name)[1:]

        offset_X_temp = 0  # locals, relative to img2 given, not absolute
        offset_X_tot = 0
        time_estimate = 2.2e-8*self.axe_X*self.axe_Y*(nbr_offset+1)
        print("\ntotal offsets estimation time: %.1f" % (time_estimate))

        if nbr_offset == 1:  # avoid two offsets for a single image
            nbr_offset = 0

        # +1 for final offset to set back image to initial offset
        for a in range(nbr_offset+1):
            if a < nbr_offset:
                print("\n%s/%s\toffset: %s" % (a+1, nbr_offset, offset_X_tot))

            self.img_debut = img_offset_X(self.img_debut, offset_X_temp)

            img2 = self.img_pixels(self.img_debut)

            if self.fixed_background != True and self.fixed_background != False:

                if self.fixed_background.get() is True:  # needed for GUI
                    img2 = img_offset_X(img2, -offset_X_tot)  # if want a fixed background and moving black hole

            if self.fixed_background is True:
                img2 = img_offset_X(img2, -offset_X_tot)  # if want a fixed background and moving black hole

            if nbr_offset != 1 and a < nbr_offset: #if need to save real offset, put offset_x in global and offset_x+offset_x2+offset_x_tot in save name
                image_name_save = "%s_D=%s_Rs=%s_size=%s_offset=%i%s" % (file_name, self.D, self.Rs, self.axe_X, offset_X_tot+self.offset_X+self.offset_X2, extension)
                img2.save(image_name_save)
                print("Save: "+image_name_save)

            if a < nbr_offset:
                offset_X_temp = int(self.axe_X/nbr_offset) #at the end to have offset=0 for the first iteration
                offset_X_tot += offset_X_temp

            elif nbr_offset > 1:
                print("\nOffset reset to %i" % (offset_X_tot % self.axe_X))

        self.img2 = img2

    def plot(self):
        """Plot the black hole and connect functions to the canvas."""
        self.fig = plt.figure('Black hole')
        self.fig.clf() #clear the graph to avoir superposing data from the same set (can be deactivated if need to superpose)
        self.ax = plt.subplot()

        if self.img2 is not None:
            self.ax.imshow(self.img2)
        else:
            print("No black hole deformation in the memory, displayed the original image instead.")
            self.ax.imshow(self.img_debut)

        self.ax.set_title("scrool to zoom in or out \nright click to add an offset in the background \nleft click to refresh image \n close the option windows to stop the program")
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('axes_leave_event', self.disconnect)
        self.fig.canvas.mpl_connect('axes_enter_event', self.connect)

        self.draw()

    def img_pixels(self, img_debut):
        """Takes deviated pixels color and assign them to seen pixels by using
        the matrices img_matrix_x and img_matrix_y."""
        pixels = np.array(img_debut)
        pixels2 = np.array(img_debut)

        xv, yv = self.img_matrix_x, self.img_matrix_y

        yv[yv >= self.axe_Y] = -2  # locate pixels outside of the image
        xv[xv >= self.axe_X] = -2

        pixels2 = pixels[yv, xv]  # apply the black hole deformation

        pixels2[xv == -1] = [0, 0, 0]  # color the black hole in black
        pixels2[yv == -2] = [255, 192, 203]  # color pixels outside
        pixels2[xv == -2] = [255, 192, 203]

        img2 = Image.fromarray(pixels2.astype('uint8'), 'RGB')
        return img2

    def img_save(self):
        """Save the image img2 with the parameters values."""
        file_name, extension = return_folder_file_extension(self.img_name)[1:]
        image_name_save = "%s_D=%s_Rs=%s_size=%s_offset=%i%s" % (file_name, self.D, self.Rs, self.axe_X, self.offset_X+self.offset_X2, extension)

        if self.img2 is not None:
            self.img2.save(image_name_save)
            print("Saved "+image_name_save)
        else:
            print("No image to save")

    def onscroll(self, event):
        """Zoom in or out the canvas when using the scrool wheel."""
        if self.out_graph is False:
            self.zoom += 10*event.step

            if self.zoom >= self.axe_X/2/self.FOV_img*self.FOV_img_Y:
                self.zoom = self.axe_X/2/self.FOV_img*self.FOV_img_Y

            if self.zoom <= 0:
                self.zoom = 0

            self.draw()

    def draw(self):
        """Draw the black hole on the canvas by setting the axes need to match
        the zoom setting."""
        # TODO: take graph axe value but careful with changing size when computing
        # need to change the equation because based on full scale and not arbitrary position
        left_side = self.zoom*self.FOV_img/self.FOV_img_Y
        right_side = self.axe_X - self.zoom*self.FOV_img/self.FOV_img_Y
        up_side = self.zoom
        down_side = self.axe_Y - self.zoom

        if right_side == self.axe_X:
            right_side -= 1
        if down_side == self.axe_Y:
            down_side -= 1

        self.ax.set_xlim((left_side, right_side))
        self.ax.set_ylim((down_side, up_side))
#        print((self.left_side, self.right_side), (self.down_side, self.up_side))
        self.fig.canvas.draw()
        plt.draw()
        plt.pause(0.001)

    def onclick(self, event):
        """Use to apply an offset when right clicking. Will be replace by a
        Slider from matplotlib."""
        # OPTIMIZE: create bar to offset and del this function
        if self.out_graph is False:

            if (event.button == 3 and event.xdata >= 0
                    and event.xdata <= self.axe_X):
                self.offset_X += self.offset_X2
                self.offset_X2 = int(self.axe_X/2 - event.xdata - self.offset_X)
                self.img_debut = img_offset_X(self.img_debut, self.offset_X2)
                self.img2 = self.img_pixels(self.img_debut)
                self.ax.imshow(self.img2)
                self.fig.canvas.draw()

    def disconnect(self, event):
        """Used to known when the mouse is outside the black hole canvas."""
        self.out_graph = True


    def connect(self, event):
        """Used to known when the mouse is inside the black hole canvas."""
        self.out_graph = False


class BlackHoleGUI:
    """GUI controling a blackhole instance."""
    def __init__(self, blackhole=None):
        """GUI controling a blackhole instance."""
        if blackhole is None:
            self.blackhole = BlackHole()
        else:
            self.blackhole = blackhole

        self.create_interface()

    def create_interface(self):
        """Create the interface."""
        root = Tk()
        frame = Frame(root)
        root.title("Black hole options")
        frame.pack()

        open_file_button = Button(frame, text="Open image", width=14, command=self.open_file_name)
        open_file_button.grid(row=0, column=0)

        L1 = Label(frame, text="radius")
        L1.grid(row=1, column=0)
        var = StringVar(root)
        var.set(self.blackhole.Rs)
        self.radius = Spinbox(frame, from_=1e-100, to=1e100, textvariable=var, bd=2, width=7)
        self.radius.grid(row=1, column=1)

        L2 = Label(frame, text="distance")
        L2.grid(row=2, column=0)
        var = StringVar(root)
        var.set(self.blackhole.D)
        self.distance = Spinbox(frame, from_=1e-100, to=1e100, textvariable=var, bd=2, width=7)
        self.distance.grid(row=2, column=1)

        compute_button = Button(frame, text="Compute", width=14, command=self.compute)
        compute_button.grid(row=1, column=2)

        self.message = Label(frame, text="", width=20)
        self.message.grid(row=1, column=3)
        self.message5 = Label(frame, text="", width=20)
        self.message5.grid(row=2, column=3)

        L3 = Label(frame, text="Image size")
        L3.grid(row=3, column=0)
        var = StringVar(root)
        var.set(self.blackhole.axe_X)
        self.size = Spinbox(frame, from_=1, to=1e100, textvariable=var, bd=2, width=7)
        self.size.grid(row=3, column=1)

        self.message2 = Label(frame, text="", width=20)
        self.message2.grid(row=3, column=3)

        save_button = Button(frame, text="Save image", width=14, command=self.img_save)
        save_button.grid(row=4, column=2)

        self.message4 = Label(frame, text="")
        self.message4.grid(row=4, column=3)

        message6 = Label(frame, text="Fix background")
        message6.grid(row=5, column=0)

        fixed_background = BooleanVar()
        C1 = Checkbutton(frame, text="", variable=fixed_background,
                         onvalue=True, offvalue=False)
        C1.grid(row=5, column=1)

        L4 = Label(frame, text="images")
        L4.grid(row=6, column=0)

        var = StringVar(root)
        var.set(10)
        self.number = Spinbox(frame, from_=1, to=1e100, textvariable=var, bd=2, width=7)
        self.number.grid(row=6, column=1)

        save_gif_button = Button(frame, text="Save animation", width=14, command=self.save_gif)
        save_gif_button.grid(row=6, column=2)

        self.message3 = Label(frame, text="")
        self.message3.grid(row=6, column=3)

        root.mainloop()

    def compute(self):
        """Call the compute method of the blackhole instance."""
        if not self.test_GUI():
            return None
        self.reset_msg()

        try:
            if float(self.distance.get()) <= 0 or float(self.radius.get()) <= 0:
                self.message["text"] = "Can't be 0 or negative"
                return None
            elif (float(self.distance.get()) == self.blackhole.D
                      and float(self.radius.get()) == self.blackhole.Rs
                      and int(self.size.get()) == self.blackhole.axe_X
                      and self.blackhole.img_matrix_x is not None):
                self.message["text"] = "same values as before"
                return None
            elif float(self.distance.get()) < float(self.radius.get()):
                self.message["text"] = "Inside black hole !"
                return None
            else:
                self.blackhole.D = float(self.distance.get())
                self.blackhole.Rs = float(self.radius.get())

        except ValueError:
            self.message["text"] = "radius, distance"
            self.message5["text"] = "& image size are floats"
            return None

        if self.blackhole.axe_X != int(self.size.get()):
            try:
                self.increase_resolution()
            except ValueError as ex:
                print(ex)
                return None

        self.message["text"] = "Computing"

        self.blackhole.compute(self.blackhole.Rs, self.blackhole.D)
        self.message["text"] = "Done !"

    def increase_resolution(self):
        """Increase the image size and correct offset and zoom to match the new
        size."""
        if not self.test_GUI():
            return None
        self.reset_msg()

        if int(self.size.get()) <= 0:
            self.message2["text"] = "Can't be 0 or negative"
            raise ValueError("Can't be 0 or negative")
        elif int(self.size.get()) == self.blackhole.axe_X:
            self.message2["text"] = "same size as before"
        else:
            # self.message2["text"] = "Computing"
            new_size_image = int(self.size.get())
            self.blackhole.offset_X += self.blackhole.offset_X2
            self.blackhole.offset_X2 = 0
            res_fact = new_size_image/self.blackhole.axe_X

            self.blackhole.axe_X = new_size_image

            self.blackhole.offset_X *= res_fact
            self.blackhole.zoom *= res_fact

            try:
                self.blackhole.img_resize(self.blackhole.axe_X)
                self.blackhole.img_debut = img_offset_X(
                    self.blackhole.img_debut, self.blackhole.offset_X)
            except ValueError as ex:
                print("error when resizing image")
                raise ValueError(ex)

    def img_save(self):
        """Save the image img2 with the parameters values. GUI version."""
        if not self.test_GUI():
            return None
        self.reset_msg()

        file_name, extension = return_folder_file_extension(self.blackhole.img_name)[1:]
        image_name_save = "%s_D=%s_Rs=%s_size=%s_offset=%i%s" % (file_name, self.blackhole.D, self.blackhole.Rs, self.blackhole.axe_X, self.blackhole.offset_X+self.blackhole.offset_X2, extension)

        if self.blackhole.img2 is not None:
            self.blackhole.img2.save(image_name_save)
            print("Saved "+image_name_save)
            self.message4["text"] = "Saved "+image_name_save
        else:
            self.message4["text"] = "No image to save"

    def save_gif(self):
        """Call the gif method if the blackhole instance."""
        if not self.test_GUI():
            return None
        self.reset_msg()

        file_name, extension = return_folder_file_extension(self.blackhole.img_name)[1:]

        if self.blackhole.img2 is not None:
            self.message3["text"] = "Computing"
        else:
            self.message3["text"] = "No image to save"
            return None
        try:
            if int(self.number.get()) <= 0:
                print("Can't be 0 or negative")
                self.message3["text"] = "Can't be 0 or negative"
            else:
                self.blackhole.gif(int(self.number.get()))
                image_name_save = "%s_D=%s_Rs=%s_size=%s_offset=%s%s" % (file_name, self.blackhole.D, self.blackhole.Rs, self.blackhole.axe_X, "*", extension)
                self.message3["text"] = "Saved "+image_name_save
        except Exception:
            print("need integer")
            self.message3["text"] = "need integer"

    def open_file_name(self):
        """Open a new image and aply the previous parameters.
        Compute a new black hole if parameters were changed.
        Source: https://gist.github.com/Yagisanatode/0d1baad4e3a871587ab1"""
        if not self.test_GUI():
            return None
        self.reset_msg()

        img_name = askopenfilename(
            # initialdir="",
            filetypes=[("Image File", ".png .jpg")],
            title="Image file")

        if img_name:
            size = int(self.size.get())
            self.blackhole.open(img_name, size=size)
            if self.blackhole.img_matrix_x is not None:
                if self.blackhole.axe_X != self.blackhole.img_matrix_x.shape[1]:
                    self.blackhole.compute(self.blackhole.Rs, self.blackhole.D)

                self.blackhole.offset_X += self.blackhole.offset_X2
                self.blackhole.offset_X2 = 0
                self.blackhole.img_debut = img_offset_X(
                    self.blackhole.img_debut, self.blackhole.offset_X)
                self.blackhole.img2 = self.blackhole.img_pixels(self.blackhole.img_debut)

                self.blackhole.plot()

    def reset_msg(self):
        """Reset all GUI messages."""
        if not self.test_GUI():
            return None
        self.message5["text"] = ""
        self.message4["text"] = ""
        self.message3["text"] = ""
        self.message2["text"] = ""
        self.message["text"] = ""

    def test_GUI(self):
        """Return True if the GUI is active and False otherwise."""
        try:
            temp = self.message["text"]
            self.message["text"] = ""
            self.message["text"] = temp
            return True
        except Exception:
            print("The GUI is not openned, function ignored")
            return False


def listdirectory(path, matrix_file):
    """Allow to search if files exist in folders.
    Source: https://python.developpez.com/faq/?page=Fichier#isFile"""
    for root, dirs, files in os.walk(path):
        for i in files:
            fichier = os.path.join(root, i)
            if fichier == matrix_file:
                return True
    return False


def spheric2cart(theta, phi):
    """Convert spherical coordinates to cartesian."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cart2spheric(x, y, z):
    """Convert cartesian coordinates to spherical."""
    # doesn't compute r because chosen egal to 1
    with np.errstate(all='ignore'):
        theta = np.arccos(z)
    phi = np.arctan2(y, x)

    return theta, phi


def rotation_matrix(beta):
    """Return the rotation matrix associated with counterclockwise rotation
    about the x axis by beta degree.
    Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    beta = np.array(beta)
    aa_bb, ab2neg = np.cos(beta), np.sin(beta)
    zero, one = np.zeros(beta.shape), np.ones(beta.shape)

    return np.array([[one, zero, zero],
                     [zero, aa_bb, -ab2neg],
                     [zero, ab2neg, aa_bb]])


def img_offset_X(img, offset_X):
    """Return the image with an offset in the X-axis.
    Allow to rotate around the blackhole."""
    offset_X = int(offset_X)
    (axe_X, axe_Y) = img.size

    while offset_X >= axe_X:
        offset_X -= axe_X

    if offset_X == 0:
        return img

    if offset_X < 0:
        offset_X = -offset_X
        img_right = img.crop((0, 0, axe_X-offset_X, axe_Y))
        img_left = img.crop((axe_X-offset_X, 0, axe_X, axe_Y))
        img.paste(img_right, (offset_X, 0))

    else:
        img_right = img.crop((0, 0, offset_X, axe_Y))
        img_left = img.crop((offset_X, 0, axe_X, axe_Y))
        img.paste(img_right, (axe_X-offset_X, 0))

    img.paste(img_left, (0, 0))

    return img


def return_folder_file_extension(img_name):
    """Return the foler, file and extension of a file."""
    *folder, file = img_name.replace("\\", "/").split("/")

    if len(folder) != 0:
        folder = folder[0]

    file, extension = file.split(".")

    return folder, file, "."+extension


def example():
    """Compute and save example."""
    blackhole = BlackHole()
    img_name = os.path.join('images', 'milkyway.jpg')
    blackhole.open(img_name, size=1000)
    blackhole.compute(Rs=8, D=50)
    blackhole.img2.save('example.jpg')


def approching_blackhole():
    """Approching a black hole example."""
    blackhole = BlackHole()
    Rs = 8.0
    D_list = np.round(10**np.linspace(np.log10(50), np.log10(100000), 30))
    blackhole.open(blackhole.img_name, size=2000)

    for D in D_list:
        blackhole.compute(Rs, D)
        blackhole.img_save()


if __name__ == "__main__":
#    blackholeGUI = BlackHoleGUI()

    blackhole = BlackHole()
    blackhole.compute(8, 50)
    blackholeGUI = BlackHoleGUI(blackhole)

#    img_name = os.path.join('images', 'milkyway.jpg')
#    blackhole.open(img_name, size=360)
#    blackhole.img_resize(360)
#    blackhole.compute(Rs=8, D=50)
