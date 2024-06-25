# This program is a free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public Licence as published by
# the Free Software Foundation, either version 3 of the Licence, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public Licence for details.
#
# You should have received a copy of the GNU Public Licence
# with this program. If not, see <htpps://www.gnu.org/licences/>.
#
#
##########################################################################
#                                                                                                                                                                                           #  
#  GoL-template-v1-2.py is the Python 3.8 program, which runs the 'Game of Life' simulation                    #
#  in the simplest possible way, that is suitable for educational purposes.                                                      #
#  (C) Jiri Kroc under the licence GPLv3                                                                                                                     #
#  Date of creation November 10, 2021                                                                                                                      #         
#                                                                                                                                                                                           #
# Cite as:                                                                                                                                                                            #
#  Jiri Kroc: "Robust massive parallel information processing environments in biology                                #
#  and medicine: case study", Problems of Information Society 13:2 (2022) 12-22,                                       #
#  DOI�: 10.25045/jpis.v13.i2.02                                                                                                                                  #  
#                                                                                                                                                                                           #  
##########################################################################


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as mpl
import random as rnd
import matplotlib.ticker as tck


"""
This Python program represents the simplest possible 
cellular automaton simulating a complex system:
the 'Game of Life' designed by John Conway. It has just 
around 100 lines of the code without initial configuration 
definition.
"""

# Initialize the size of the simulated lattice
x_size = 60
y_size = 60
lattice_size = [x_size, y_size]
# When true, this flag inserts random swapping of values
# into the simulation to test its robustness.
rnd_swap_on = False #False #True
# Probability of swapping of each cell's value
probab_swap = 0.001
# random initial condition Tdue/False
rnd_initial_value  = False 
grid = False


# Initialization of both lattices 
lattice_old = np.zeros((x_size,y_size), dtype =  int)
lattice_new = np.zeros_like(lattice_old)
if rnd_initial_value == True:
  # Random initial values (fixed amount of randomly chosen cells)
  lattice_old = np.random.randint(low=0, high=2, size=lattice_size, dtype=np.int)
else: 
  # Glider gun initial values (comment whe random initial cond. used):
  lattice_old[7,5] = 1; lattice_old[7,6] = 1; 
  lattice_old[8,5] = 1; lattice_old[8,6] = 1; 
  lattice_old[7,15] = 1; lattice_old[8,15] = 1; lattice_old[9,15] = 1; 
  lattice_old[6,16] = 1; lattice_old[10,16] = 1;  
  lattice_old[11,17] = 1; lattice_old[11,18] = 1;  
  lattice_old[5,17] = 1; lattice_old[5,18] = 1;  
  lattice_old[8,19] = 1; 
  lattice_old[6,20] = 1; lattice_old[10,20] = 1;  
  lattice_old[7,21] = 1; lattice_old[8,21] = 1;  lattice_old[9,21] = 1; 
  lattice_old[8,22] = 1; 
  lattice_old[6,25] = 1; lattice_old[5,25] = 1;  lattice_old[7,25] = 1; 
  lattice_old[6,26] = 1; lattice_old[5,26] = 1;  lattice_old[7,26] = 1; 
  lattice_old[4,27] = 1; lattice_old[8,27] = 1; 
  lattice_old[4,29] = 1; lattice_old[8,29] = 1;  
  lattice_old[3,29] = 1; lattice_old[9,29] = 1; 
  lattice_old[5,39] = 1; lattice_old[6,39] = 1; 
  lattice_old[5,40] = 1; lattice_old[6,40] = 1; 
  # Copperhead spaceship 
  lattice_old[22,53] = 1; lattice_old[22,54] = 1; 
  lattice_old[22,57] = 1; lattice_old[22,58] = 1; 
  lattice_old[23,55] = 1; lattice_old[23,56] = 1; 
  lattice_old[24,55] = 1; lattice_old[24,56] = 1; 
  lattice_old[25,52] = 1; lattice_old[25,54] = 1; 
  lattice_old[25,57] = 1; lattice_old[25,59] = 1; 
  lattice_old[26,52] = 1; lattice_old[26,59] = 1; 
  lattice_old[28,52] = 1; lattice_old[28,59] = 1; 
  lattice_old[29,53] = 1; lattice_old[29,54] = 1; 
  lattice_old[29,57] = 1; lattice_old[29,58] = 1;
  lattice_old[30,54] = 1; lattice_old[30,55] = 1; 
  lattice_old[30,56] = 1; lattice_old[30,57] = 1; 
  lattice_old[32,55] = 1; lattice_old[32,56] = 1; 
  lattice_old[33,55] = 1; lattice_old[33,56] = 1; 
  # Blinker
  lattice_old[25,15] = 1; lattice_old[25,16] = 1; 
  lattice_old[25,17] = 1; 
  # Toad
  lattice_old[35,25] = 1; lattice_old[35,26] = 1; 
  lattice_old[35,27] = 1; 
  lattice_old[36,26] = 1; lattice_old[36,27] = 1; 
  lattice_old[36,28] = 1;   
  # Pentadecathlon 
  lattice_old[40,40] = 1; lattice_old[40,41] = 1; lattice_old[40,42] = 1;  
  lattice_old[41,40] = 1; lattice_old[41,42] = 1; 
  lattice_old[42,40] = 1; lattice_old[42,41] = 1; lattice_old[42,42] = 1;
  lattice_old[43,40] = 1; lattice_old[43,41] = 1; lattice_old[43,42] = 1;
  lattice_old[44,40] = 1; lattice_old[44,41] = 1; lattice_old[44,42] = 1;
  lattice_old[45,40] = 1; lattice_old[45,41] = 1; lattice_old[45,42] = 1;
  lattice_old[46,40] = 1; lattice_old[46,42] = 1;
  lattice_old[47,40] = 1; lattice_old[47,41] = 1; lattice_old[47,42] = 1; 
  # Beacon
  lattice_old[40,5] = 1; lattice_old[40,6] = 1; 
  lattice_old[41,5] = 1; lattice_old[41,6] = 1;       
  lattice_old[42,7] = 1; lattice_old[42,8] = 1; 
  lattice_old[43,7] = 1; lattice_old[43,8] = 1;  
# Beacon Mirrored 
  lattice_old[54,22] = 1; lattice_old[54,23] = 1; 
  lattice_old[55,22] = 1; lattice_old[55,23] = 1;       
  lattice_old[56,20] = 1; lattice_old[56,21] = 1; 
  lattice_old[57,20] = 1; lattice_old[57,21] = 1;            

def Sum(lattice, x, y):
	"""
    Evaluation of the Sum over neighbors except the central
    cell with periodic boundary conditions.	
	"""
	add = 0
	for i in range (-1, 2):
	  for j in range (-1, 2):
	    # modulo % operation enables periodic boundary conditions
	    add += lattice_old[(x_size + x + i)%(x_size)][(y_size + y + j)%(y_size)] 
        
	add -= lattice_old[x][y]
	return add


 

def Update():

    """
    The lattice Updating function: Evaluates the new layer using 
    the old one, and finally swap their pointers.
    """

    global lattice_new
    global lattice_old

    for x in range(0, x_size):
        for y in range(0, y_size):
            sum = Sum(lattice_old[x][y], x, y)
            if ((sum==2) & (lattice_old[x][y] == 1)):
                lattice_new[x][y] = 1
            elif ((sum == 3)):
                lattice_new[x][y] = 1
            else:
                lattice_new[x][y] = 0
            
            if rnd_swap_on == True and rnd.random() < probab_swap:
                lattice_new[x][y] = (lattice_new[x][y] +1) %2
        
def Swap():
  """
     Swaps old and new lattices by copying new one into the old one, 
     and erasing content of the new one to prepare it for a new 
     simulation step.
  """
  global lattice_new
  global lattice_old  
  # Erasing all values in the old lattice: cleaning old data.
  lattice_old = np.zeros_like(lattice_old)
  # Copying values of new to old lattice => this defines evolution!
  lattice_old = np.copy(lattice_new)
  # Preparing new lattice by erasing all values.
  lattice_new = np.zeros_like(lattice_old)


def figure_init():
  """ Figure  initialization: set colors, set major and minor ticks,
       and can be adjusted to have a grid.
  """
  fig, ax = plt.subplots()
  bwmap=(mpl.colors.ListedColormap(['white','black']))
  # Adjusting ticks to grid in imshow accordingly
  extent = (0, lattice_old.shape[1], lattice_old.shape[0], 0)
#  _, ax = plt.subplots()
  imag = plt.imshow(lattice_old, cmap=bwmap,extent=extent)
  title_text = 'Game of Life by John Conway, step = ' \
      + '{0:5d}'.format(0)
  plt.title(title_text)
#  # Set major ticks & labels
  ax.tick_params(left = True, right = True, bottom = True, top = True)
  ax.tick_params(labelleft = True, labelright = True, \
      labelbottom = True, labeltop = True)
  ax.xaxis.set_major_locator(plt.MultipleLocator(10))
  ax.yaxis.set_major_locator(plt.MultipleLocator(10))  
#  # Set minor ticks at the left and bottom
  ax.xaxis.set_minor_locator(tck.MultipleLocator(1))  
  ax.yaxis.set_minor_locator(tck.MultipleLocator(1))    
  ax.xaxis.set_tick_params(which='minor', color='green', 
      left=True, right=True)   
  ax.yaxis.set_tick_params(which='minor', color='green', 
      left=True, right=True)  
#  # Set the grid of green lines within the lattice
  if grid == True:
    ax.grid(which = 'minor', axis = 'x', color = "g", linewidth = 0.25)
    ax.xaxis.remove_overlapping_locs = False
    ax.grid(which = 'minor', axis = 'y', color = "g", linewidth = 0.25)
    ax.yaxis.remove_overlapping_locs = False  

  return fig, imag


def init_config():
  """
  Draw the initial configuration as the zero time step. 
  """
  step = 0
  title_text = 'Game of Life by John Conway, step = ' \
          + '{0:5d}'.format(step) 
  plt.title(title_text)
  image.set_array(lattice_old)      
  figure.canvas.draw()  
        

def animate(step):
    """
    The Function administering the animation: calls Update() 
    function and draw the updated lattice on canvas.
    """
    global title_text
    if step == frames_end:
        print(f'{step} == {frames_end}; closing!')
        plt.close(figure)
    else:
        title_text = 'Game of Life by John Conway, step = ' \
            + '{0:5d}'.format(step) 
        plt.title(title_text)
        Update()
        image.set_array(lattice_old)      
        figure.canvas.draw()
        Swap()
        return image


# Main program

# Defines number of frames per second shown in animations
frame_pause = 20 
# Defines the range of shown frames in animations
frames_ini = 0
frames_end = 300
frames_end += 1

# Call the function providing  the animation from
# the module Animation
figure, image = figure_init()
anim_video = anim.FuncAnimation(figure, animate, init_func = init_config, \
  interval=frame_pause, frames=range(frames_ini, \
  frames_end), repeat=False)

# Creates animated gif figure:
#anim_video.save('GoL-anim-up-gr.gif', dpi=1200, writer=anim.PillowWriter(fps=10))

# Creates mpeg movie (import necessary module ffmpeg):
# anim_video.save("GoL-movie.mpg",writer=anim.writers['avconv'])

plt.show()