#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Turtle facial recognition using map graphs.
"""

# =============================================================================
# Created By  : Karun Kumar Rao, Ricardo Azevedo
# Last Updated: Fri Jun 21 17:43:41 2019
# =============================================================================


import cv2
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from PyQt4.QtGui import QFileDialog
import os


def select_k(spectrum, minimum_energy = 0.9):
    '''
    Include eigenvalues only up to a target proportion of the sum of
    the eigenvalues, starting from the smallest eigenvalue.

    Parameters
    ----------
    spectrum : numpy array
        Graph spectrum.
    minimum_energy : float
        Target proportion of the sum of eigenvalues.

    Returns
    -------
    int
        Number of eigenvalues to include.
    '''
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
    if running_total / total >= minimum_energy:
        return i + 1
    return len(spectrum)


def get_spectrum(graph, method='laplacian'):
    '''
    Calculate spectrum of a graph.

    Parameters
    ----------
    graph : networkx graph
        Graph.
    method : str (default: 'laplacian')
        Type of matrix on which to calculate the spectrum.
        (other options: 'adjacency', 'normalized_laplacian')

    Returns
    -------
    numpy array
        Graph spectrum.
    '''
    if method == 'laplacian':
        return nx.spectrum.laplacian_spectrum(graph)
    elif method == 'adjacency':
        return nx.spectrum.adjacency_spectrum(graph)
    elif method == 'normalized_laplacian':
        return nx.spectrum.normalized_laplacian_spectrum(graph)


def graph_dissimilarity(graph1, graph2, method='laplacian'):
    '''
    Calculate dissimilarity between two graphs based on their spectra.
    Dissimilarity is 0 if the graphs are identical.

    Parameters
    ----------
    graph1 : networkx graph
        Graph.
    graph2 : networkx graph
        Graph.
    method : str (default: 'laplacian')
        Type of spectrum to use
        (other options: 'adjacency', 'normalized_laplacian')

    Returns
    -------
    float
        Dissimilarity.
    '''
    spectrum1 = get_spectrum(graph1, method)
    spectrum2 = get_spectrum(graph2, method)
    k1 = select_k(spectrum1)
    k2 = select_k(spectrum2)
    k = min(k1, k2)
    dissimilarity = np.power(spectrum1[:k] - spectrum2[:k], 2).sum()
    return dissimilarity


def nearest_nonzero_idx(a,x,y):
    '''
    Returns index of nearest non-zero element to a[x,y]

    Parameters
    ----------
    a : matrix
        numpy array
    x : position in array
        int
    y : position in array
        int

    Returns
    -------
    float
        Index of nearest non zero value in a to position x,y
    '''
    tmp = a[x,y]
    a[x,y] = 0
    r,c = np.nonzero(a)
    a[x,y] = tmp
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]


def new_sighting(turtleID, species, date, location, side, fingerprint, img):
    #Takes data, returns a dictionary named turtlesid with one sighting
    turtleID = {}
    turtleID['Species'] = species
    turtleID[str(date)] = {'Location':location, side:{'Fingerprint':fingerprint, 'Image':img}}
    return turtleID


def draw_graph(G):
    nx.draw_networkx(G, nx.get_node_attributes(G, 'pos'))


def get_graph():
    #asks for image, you annotate it, and it returns a (cropped image, annotated image) combined, and graph object
    global ix, iy, drawing, mask, n, mode, img, cropped, circle_size, G
    img_location = QFileDialog.getOpenFileName()
    drawing = False
    ix,iy = -1,-1
    mode = False #change to 1 if cropping
    def draw_polygon(event, x, y, flags, param):
        global ix, iy, drawing, mask, n, mode, img, cropped, circle_size, G
        if event == cv2.EVENT_RBUTTONDOWN:
            if n == 1: #fix cropping/circlesize after first node identified
                circle_size = int(mask.shape[0] / 200)
                cropped = np.copy(img)
            cv2.circle(img, (x,y), 2*circle_size, (255,0,0), -1)
            mask[y-circle_size:y+circle_size, x-circle_size:x+circle_size] = n
            G.add_node(n, pos=np.array((x,y)))
            n+=1

        elif event == cv2.EVENT_LBUTTONDOWN:
            if not mode:
                drawing=True
            ix,iy = x,y

        elif event == cv2.EVENT_LBUTTONUP:
            if mode:
                img = img[iy:y, ix:x]
                mask = mask[iy:y, ix:x]
            else:
                drawing = False
                iy_real, ix_real = nearest_nonzero_idx(mask, iy,ix)
                y_real, x_real = nearest_nonzero_idx(mask, y,x)
                cv2.line(img,(ix_real,iy_real),(x_real,y_real),(0,255,0),2)
                e1 = mask[iy_real,ix_real]
                e2 = mask[y_real,x_real]
                if e1 == 0 or e2 == 0 or e1 == e2:
                    pass
                else:
                    edge.append((e1,e2))

    img = cv2.imread(img_location)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image', draw_polygon)

    n = 1
    edge = []
    mask = np.zeros(img.shape[:2])
    G=nx.Graph()

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(5) & 0xFF
        if k == ord('m'):
            mode = not mode
        if k == 27:
            image = np.hstack((cropped, img))
            break
    cv2.destroyAllWindows()
    G.add_nodes_from(np.arange(1,n))
    # print(edge)
    G.add_edges_from(edge)
    print(G.nodes())
    return G, image, img_location

G, image, img_location = get_graph()
date = '0'
location = '0'
species = '0'
side = 'Right'

turtleID = img_location.split('/')[-2][:-7]+input('0, 1, or 2? ')
ids = [i.split('.')[0] for i in os.listdir('./Database/')]
a = new_sighting(turtleID, species, date, location, side, G, image)
#Copy original image to image database with name ID_Date_Side
#shutil.copy2(img_location, './Raw_Images/'+turtleID+'_'+date+'_'+side+'.'+ext)
with open('Database/'+turtleID+'.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

G, img = get_graph()
