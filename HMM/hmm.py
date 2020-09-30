'''Implementation of Hidden Markov Model'''

import math
import numpy as np
import matplotlib.pyplot as plt


def read_data():
    with open('hmm-data.txt', 'r') as f:
        content = f.readlines()
        for l in range(len(content)):
            content[l] = content[l].strip('\n')
        # generate Grid-World
        Grid_World = content[2:12]
        for i in range(len(Grid_World)):
            Grid_World[i] = Grid_World[i].split()
            Grid_World[i] = list(map(int, Grid_World[i]))

        # generate Tower Locations
        Tower_Locations = content[16:20]
        for i in range(len(Tower_Locations)):
            Tower_Locations[i] = Tower_Locations[i][-3:].split()
            Tower_Locations[i] = list(map(int, Tower_Locations[i]))

        # generate Noisy Distances
        Noisy_Distances = content[24:35]
        for i in range(len(Noisy_Distances)):
            Noisy_Distances[i] = Noisy_Distances[i].split()
            Noisy_Distances[i] = list(map(float, Noisy_Distances[i]))
        Noisy_Distances = [tuple(r) for r in Noisy_Distances]  # convert to tuples
        return Grid_World, Tower_Locations, Noisy_Distances


def coord_list():  # list of all location coordinates
    coords = []
    for i in range(10):
        for j in range(10):
            coords.append((i, j))
    return coords


def free_list(coord_list, Grid_World):  # list of free coordinates
    free_list = []
    for c in coord_list:
        if Grid_World[c[0]][c[1]] == 1:
            free_list.append(c)
    return free_list


def obstacle_list(coord_list, Grid_World):  # list of obstacle coordinates
    obstacle_list = []
    for c in coord_list:
        if Grid_World[c[0]][c[1]] == 0:
            obstacle_list.append(c)
    return obstacle_list


def generateCPT_x_initial(coord_list, Grid_World):  # the initial CPT in first time step
    # compute the probability of a free cell
    freeCellNum = 0
    for i in Grid_World:
        for j in range(len(i)):
            if i[j] == 1:
                freeCellNum += 1
    # generate the initial CPT
    CPT_ini = []
    for c in coord_list:
        if Grid_World[c[0]][c[1]] == 1:  # only record free cell coordinates
            CPT_ini.append(1/freeCellNum)
    return np.array(CPT_ini)  # 87*1 array


def generateCPT_x(free_list, obstacle_list):
    CPT_dict = dict()
    for c in free_list:  # coordinate in xi-1
        valid_step = []
        for d in range(len(c)):  # determine feature, horizontal or vertical
            for m in [-1, 1]:  # determine movement direction
                coord_next = list(c)
                coord_next[d] += m  # the possible location coordinate to move to from c
                if all(0 <= i <= 9 for i in coord_next):  # check all coordinates of coord_next between 0 and 9
                    if tuple(coord_next) not in obstacle_list:
                        valid_step.append(tuple(coord_next))
        prob = 1/len(valid_step)  # compute the probability of moving
        CPT_dict[c] = {k:prob for k in valid_step}
    CPT_x = []
    for i in range(len(free_list)):  # ci location
        CPT_ci = []
        for j in range(len(free_list)):  # ci-1 location
            if free_list[i] in CPT_dict[free_list[j]].keys():
                CPT_ci.append(CPT_dict[free_list[j]][free_list[i]])
            else:
                CPT_ci.append(0)
        CPT_x.append(CPT_ci)
    return CPT_x  # size 87*87, row for ci, column for ci-1


def L2distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)


def measure_loc_probability(Noisy_measurement, c, Tower_Locations):
    measure_intervals = []
    for i in range(len(Tower_Locations)):
        true_distance = L2distance(c, Tower_Locations[i])
        measure_intervals.append([0.7*true_distance, 1.3*true_distance])
    prob = 1
    for i in range(len(Noisy_measurement)):
        if measure_intervals[i][0] < Noisy_measurement[i] < measure_intervals[i][1]:
            prob = prob * (1/(10*(measure_intervals[i][1] - measure_intervals[i][0])))
        else:
            prob = 0
    return prob


def generateCPT_E(free_list, Tower_Locations, Noisy_Distances):
    CPT_E = []
    for i in range(len(Noisy_Distances)):
        ND_prob = []  # to store all probabilities of each location to get this Noisy Distance
        for c in free_list:
            ND_prob.append(measure_loc_probability(Noisy_Distances[i], c, Tower_Locations))
        CPT_E.append(ND_prob)
    return CPT_E  # size 11*87


def viterbi_agl():
    Grid_World, Tower_Locations, Noisy_Distances = read_data()
    coords = coord_list()
    free_cells = free_list(coords, Grid_World)
    obstacle_cells = obstacle_list(coords, Grid_World)
    # get CPTs
    CPT_x_initial = generateCPT_x_initial(coords, Grid_World)
    CPT_x = generateCPT_x(free_cells, obstacle_cells)
    CPT_E = generateCPT_E(free_cells, Tower_Locations, Noisy_Distances)

    new_path = dict()  # temp variable
    for c in range(len(free_cells)):  # travel cell index
        new_path[c] = [c]
    path = dict(new_path)  # to store path locations
    prob_collect = []
    # get first most possible location
    ini_prob = list(np.multiply(np.array(CPT_E[0]), CPT_x_initial))  # probability of each location at the beginning
    prob_collect.append(ini_prob)
    optima = ini_prob.index(max(ini_prob))  # store the index of the cell at the end of optimal path

    # predict remaining steps
    for t in range(1, len(CPT_E)):
        c_prob = []  # store all max prob of each cell in this state
        for i in range(len(free_cells)):  # index of cell in current state
            ci_from_ci_1_prob = list(np.multiply(np.array(prob_collect[-1]), np.array(CPT_x[i])))  # prob of each ci-1 to ci
            ci_max_prob = max(ci_from_ci_1_prob)
            c_prob.append(ci_max_prob)
            if ci_max_prob != 0:
                ci_max_from = ci_from_ci_1_prob.index(ci_max_prob)  # index, from which ci-1 come to ci is most possible
                new_path[i] = list(path[ci_max_from])
                new_path[i].append(i)
        path = dict(new_path)  # update path
        prob_state = list(np.multiply(np.array(CPT_E[t]), np.array(c_prob)))  # add CPT_E into prob
        prob_collect.append(prob_state)
        optima = prob_state.index(max(prob_state))
    return [free_cells[l] for l in path[optima]]


def plot_path():  # plot the path
    Grid_World, _, _ = read_data()
    coords = coord_list()
    obstacle_cells = obstacle_list(coords, Grid_World)
    obstacle_y = [i[0] for i in obstacle_cells]
    obstacle_x = [i[1] for i in obstacle_cells]
    path_loc = viterbi_agl()
    path_y = [i[0] for i in path_loc]
    path_x = [i[1] for i in path_loc]
    fig, ax = plt.subplots()
    plt.xlim(0, 9)
    plt.ylim(9, 0)
    ax.xaxis.tick_top()
    plt.scatter(path_x, path_y, color='r', zorder=5)
    plt.plot(path_x, path_y, color='r')  # plot route
    plt.scatter(path_x[0], path_y[0], color='g', zorder=10)  # mark start
    plt.scatter(obstacle_x, obstacle_y, color='k', zorder=5)
    plt.grid()
    plt.show()


print('The most possible path of the robot is:', viterbi_agl())
plot_path()
