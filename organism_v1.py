#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Evolving Simple Organisms
#   2017-Nov.
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division, print_function
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

from plotting import plot_food
from plotting import plot_organism

import numpy as np
import operator

from math import atan2
from math import cos
from math import degrees
from math import floor
from math import radians
from math import sin
from math import sqrt
from random import randint
from random import random
from random import sample
from random import uniform
import pandas as pd
import os
import glob
import re
from PIL import Image

import streamlit as st
import altair as alt

stats_filename = 'simulation_stats.txt'
plots_path = './plots'
gif_path = 'animation.gif'

#--- FUNCTIONS ----------------------------------------------------------------+

def make_animated_gif(folder_path):
    # Function to extract the numerical part of the filename for correct sorting
    def numerical_sort(value):
        numbers = re.findall(r'\d+', value)
        return list(map(int, numbers))

    # Use glob to get all the PNG files in the folder, sorted by filename numerically
    image_files = sorted(glob.glob(f'{folder_path}/*.png'), key=numerical_sort)

    # Open the images and save them in a list
    images = [Image.open(image) for image in image_files]

    # Save the images as an animated GIF
    images[0].save('animation.gif',
                save_all=True,
                append_images=images[1:],
                duration=100,  # Duration between frames in milliseconds
                loop=0)        # 0 means loop forever


def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def plot_and_save(gen, time_step, fitness_scores):
    plt.figure()
    plt.plot(fitness_scores, 'o', color='blue')
    plt.title(f"GENERATION: {gen} TIME_STEP: {time_step}")
    plt.xlabel("Organism")
    plt.ylabel("Fitness Score")
    
    # Ensure the directory exists
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # Save the plot to the specified directory with a unique filename
    plt.savefig(f'./plots/{gen}-{time_step}.png', dpi=100)
    plt.close()  # Close the plot to avoid display during batch runs


def dist(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)


def calc_heading(org, food):
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180: theta_d += 360
    return theta_d / 180


def plot_frame(settings, organisms, foods, gen, time):
    fig, ax = plt.subplots()
    fig.set_size_inches(9.6, 5.4)

    plt.xlim([settings['x_min'] + settings['x_min'] * 0.25, settings['x_max'] + settings['x_max'] * 0.25])
    plt.ylim([settings['y_min'] + settings['y_min'] * 0.25, settings['y_max'] + settings['y_max'] * 0.25])

    # PLOT ORGANISMS
    for organism in organisms:
        plot_organism(organism.x, organism.y, organism.r, ax)

    # PLOT FOOD PARTICLES
    for food in foods:
        plot_food(food.x, food.y, ax)

    # MISC PLOT SETTINGS
    ax.set_aspect('equal')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    plt.figtext(0.025, 0.95,r'GENERATION: '+str(gen))
    plt.figtext(0.025, 0.90,r'T_STEP: '+str(time))

    plt.savefig(f'./plots/{gen}-{time}.png', dpi=100)
    plt.close()
##    plt.show()


def evolve(settings, organisms_old, gen):

    elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_orgs = settings['pop_size'] - elitism_num

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness

        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']


    #--- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = []
    for i in range(0, elitism_num):
        organisms_new.append(organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name))


    #--- GENERATE NEW ORGANISMS ---------------------------+
    for w in range(0, new_orgs):

        # SELECTION (TRUNCATION SELECTION)
        canidates = range(0, elitism_num)
        random_index = sample(canidates, 2)
        org_1 = orgs_sorted[random_index[0]]
        org_2 = orgs_sorted[random_index[1]]

        # CROSSOVER
        crossover_weight = random()
        wih_new = (crossover_weight * org_1.wih) + ((1 - crossover_weight) * org_2.wih)
        who_new = (crossover_weight * org_1.who) + ((1 - crossover_weight) * org_2.who)

        # MUTATION
        mutate = random()
        if mutate <= settings['mutate']:

            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0,1)

            # MUTATE: WIH WEIGHTS
            if mat_pick == 0:
                index_row = randint(0,settings['hnodes']-1)
                wih_new[index_row] = wih_new[index_row] * uniform(0.9, 1.1)
                if wih_new[index_row] >  1: wih_new[index_row] = 1
                if wih_new[index_row] < -1: wih_new[index_row] = -1

            # MUTATE: WHO WEIGHTS
            if mat_pick == 1:
                index_row = randint(0,settings['onodes']-1)
                index_col = randint(0,settings['hnodes']-1)
                who_new[index_row][index_col] = who_new[index_row][index_col] * uniform(0.9, 1.1)
                if who_new[index_row][index_col] >  1: who_new[index_row][index_col] = 1
                if who_new[index_row][index_col] < -1: who_new[index_row][index_col] = -1

        organisms_new.append(organism(settings, wih=wih_new, who=who_new, name='gen['+str(gen)+']-org['+str(w)+']'))

    return organisms_new, stats


def simulate(settings, organisms, foods, gen):

    total_time_steps = int(settings['gen_time'] / settings['dt'])

    #--- CYCLE THROUGH EACH TIME STEP ---------------------+
    for t_step in range(0, total_time_steps, 1):

        # PLOT SIMULATION FRAME
        if settings['plot']==True and gen==settings['gens']-1:
            plot_frame(settings, organisms, foods, gen, t_step)
            

        # UPDATE FITNESS FUNCTION
        for food in foods:
            for org in organisms:
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # UPDATE FITNESS FUNCTION
                if food_org_dist <= 0.075:
                    org.fitness += food.energy
                    food.respawn(settings)

                # RESET DISTANCE AND HEADING TO NEAREST FOOD SOURCE
                org.d_food = 100
                org.r_food = 0

        # CALCULATE HEADING TO NEAREST FOOD SOURCE
        for food in foods:
            for org in organisms:

                # CALCULATE DISTANCE TO SELECTED FOOD PARTICLE
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE
                if food_org_dist < org.d_food:
                    org.d_food = food_org_dist
                    org.r_food = calc_heading(org, food)

        # GET ORGANISM RESPONSE
        for org in organisms:
            org.think()

        # UPDATE ORGANISMS POSITION AND VELOCITY
        for org in organisms:
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)

    return organisms


#--- CLASSES ------------------------------------------------------------------+


class food():
    def __init__(self, settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1


    def respawn(self,settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1


class organism():
    def __init__(self, settings, wih=None, who=None, name=None):

        self.x = uniform(settings['x_min'], settings['x_max'])  # position (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # position (y)

        self.r = uniform(0,360)                 # orientation   [0, 360]
        self.v = uniform(0,settings['v_max'])   # velocity      [0, v_max]
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])   # dv

        self.d_food = 100   # distance to nearest food
        self.r_food = 0     # orientation to nearest food
        self.fitness = 0    # fitness (food count)

        self.wih = wih
        self.who = who

        self.name = name


    # NEURAL NETWORK
    def think(self):

        # SIMPLE MLP
        af = lambda x: np.tanh(x)               # activation function
        h1 = af(np.dot(self.wih, self.r_food))  # hidden layer
        out = af(np.dot(self.who, h1))          # output layer

        # UPDATE dv AND dr WITH MLP RESPONSE
        self.nn_dv = float(out[0])   # [-1, 1]  (accelerate=1, deaccelerate=-1)
        self.nn_dr = float(out[1])   # [-1, 1]  (left=1, right=-1)


    # UPDATE HEADING
    def update_r(self, settings):
        self.r += self.nn_dr * settings['dr_max'] * settings['dt']
        self.r = self.r % 360


    # UPDATE VELOCITY
    def update_vel(self, settings):
        self.v += self.nn_dv * settings['dv_max'] * settings['dt']
        if self.v < 0: self.v = 0
        if self.v > settings['v_max']: self.v = settings['v_max']


    # UPDATE POSITION
    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']
        self.x += dx
        self.y += dy


#--- MAIN ---------------------------------------------------------------------+

def run(settings):
    # Initialize the DataFrame to store statistics
    empty_folder(plots_path)
    stats_df = pd.DataFrame({
        'GEN': pd.Series(dtype='int'),
        'PARAMETER': pd.Series(dtype='str'),  # 'category' is often used for string labels
        'VALUE': pd.Series(dtype='float')
    })
    cols = st.columns(4)
    with cols[0]:
        ph_gen = st.empty()
    with cols[1]:
        ph_worst = st.empty()
    with cols[2]:
        ph_avg = st.empty()
    with cols[3]:
        ph_best = st.empty()
    
    plot_placeholder = st.empty()


    #--- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for i in range(0,settings['food_num']):
        foods.append(food(settings))

    #--- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    organisms = []
    for i in range(0,settings['pop_size']):
        wih_init = np.random.uniform(-1, 1, (settings['hnodes'], settings['inodes']))     # mlp weights (input -> hidden)
        who_init = np.random.uniform(-1, 1, (settings['onodes'], settings['hnodes']))     # mlp weights (hidden -> output)

        organisms.append(organism(settings, wih_init, who_init, name='gen[x]-org['+str(i)+']'))

    #--- CYCLE THROUGH EACH GENERATION --------------------+
    
    for gen in range(0, settings['gens']):

        # SIMULATE
        organisms = simulate(settings, organisms, foods, gen)

        # EVOLVE
        organisms, stats = evolve(settings, organisms, gen)
        ph_gen.write(f"Gen: {gen}")
        ph_worst.write(f"Worst: {stats['WORST']}")
        ph_avg.write(f"Avg: {stats['AVG']}")
        ph_best.write(f"Best: {stats['BEST']}")
        
        new_row = pd.DataFrame([{
            'GEN': gen,
            'PARAMETER': 'BEST',
            'VALUE': stats['BEST'],
        },
        {
            'GEN': gen,
            'PARAMETER': 'AVG',
            'VALUE': stats['AVG'],
        },
        {
            'GEN': gen,
            'PARAMETER': 'WORST',
            'VALUE': stats['WORST'],
        }])
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)
        chart = alt.Chart(stats_df).mark_line().encode(
            x='GEN:Q',
            y='VALUE:Q',
            color='PARAMETER:N'
        ).properties(
            width=600,
            height=400
        )
        plot_placeholder.altair_chart(chart)
    
    pd.DataFrame(stats_df).to_csv(stats_filename, index=False)
    if settings['plot']:
        make_animated_gif(plots_path)
    
    st.image(gif_path)
    with open(gif_path, "rb") as file:
        btn = st.download_button(
            label="Download GIF",
            data=file,
            file_name="animated.gif",
            mime="image/gif"
        )
    st.success(f"Simulation completed. Stats saved to {stats_filename}")
    return stats_df