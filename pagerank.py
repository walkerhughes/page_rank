
import numpy as np
import networkx as nx
from itertools import combinations
from scipy import linalg as la


class DiGraph:
    """
    A class for representing directed graphs via their adjacency matrices.

    PageRank vector can be computed using linearsolve, eigensolve, or iterrative solving methods. 
    """
    def __init__(self, A, labels = None): 
        """
        Modifies A so that there are no sinks in the corresponding graph,
        then calculates Ahat. Saves Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        self.n = len(A[0]) # will use more than once
        self.zeros, self.ones = np.zeros(self.n), np.ones(self.n)
        
        if not labels: # if no labels given make a range
            labels = list(range(self.n))
                
        if len(labels) != self.n: # if there are too many/too few
            raise ValueError("Labels not of correct dimension.")
        
        # eliminate sinks in A
        for i in range(self.n):
            if np.allclose(A[:, i], self.zeros):
                A[:, i] = self.ones
        
        # calculate A_hat with broadcasting along columns
        # store attributes
        self.Ahat = A / np.sum(A, axis = 0)
        self.labels = labels
        
        
    def linsolve(self, epsilon=0.85):
        """
        Computes the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # calculate vector of probabilities
        inv = la.inv(np.eye(self.n) - epsilon * self.Ahat)
        vec = ((1 - epsilon) / self.n) * self.ones
        p = inv @ vec
        # return dictionary mapping labels to probabilities
        return {self.labels[i]: p[i] for i in range(self.n)}

    def eigensolve(self, epsilon=0.85):
        """
        Computes the PageRank vector using the eigenvalue method.
        Normalizes the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # compute E matrix and coefficients
        E, coeff = np.ones((self.n, self.n)), ((1 - epsilon) / self.n)
        B = (epsilon * self.Ahat) + (coeff * E)
        # eigenvectors and values of B matrix
        eigs, vecs = la.eig(B)
        p = vecs[:, 0] / sum(vecs[:, 0])
        # return dictionary mapping labels to probabilities
        return {self.labels[i]: p[i] for i in range(self.n)}
        
        
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """
        Computes the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        p0 = self.ones / self.n # ones vec divided by length
        
        def next_p(p0):
            """helper function to compute next p value"""
            return (epsilon * self.Ahat @ p0) + (1 - epsilon) * self.ones / self.n
            
        # iterate through
        for i in range(maxiter):
            p1 = next_p(p0)
            if max(np.abs(p1 - p0)) < tol:
                p0 = p1
                break
            p0 = p1
        
        # return dictionary mapping labels to probabilities
        return {self.labels[i]: p0[i] for i in range(self.n)}



def get_ranks(d):
    """
    Constructs a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    # get labels and values
    labels, vals = list(d.keys()), list(d.values()) 
    # return list mapping labels sorted by rank
    return [labels[i] for i in np.flip(np.argsort(vals), axis = 0)] 
    

def rank_websites(filename = "web_stanford.txt", epsilon = 0.85):
    """
    Reads the specified file and constructs a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Uses the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then ranks them with get_ranks(). If two webpages have the same rank,
    resolves ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    # read in data
    with open("web_stanford.txt", "r") as myfile:
        data = myfile.readlines()
    
    # make set of all pages
    d, labels = {}, set()
    for i in data:
        temp = list(map(int, i.split("/")))
        d.update({temp[0]: temp[1: ]})
        labels = labels.union(set(temp))
        
    # sort all the labels
    labels = list(labels)
    labels.sort()
    
    # find total number of them and make A matrix 
    n = len(labels) 
    A = np.zeros((n, n))
    # maps keys to an index for matrix
    indices = {key: i for i, key in enumerate(labels)} 
    
    # updata A matrix based on how each site mapps to other sites
    for label in labels:
        try:
            index = indices[label]
            vals = d[label]
            for val in vals:
                A[:, index][indices[val]] = 1
        except:
            continue
        
    # return ranked sites via functions above
    graph = DiGraph(A, labels = labels)
    new_dict = graph.itersolve(epsilon = epsilon)
    return get_ranks(new_dict)
    
    

def rank_ncaa_teams(filename, epsilon=0.85):
    """
    Reads the specified file and constructs a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Uses the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then ranks them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    # read in data and do parse lines, get set of all teams
    with open(filename, "r") as myfile:
        data = myfile.read().strip()
        sites = sorted(set(data.replace("\n", ",").split(",")[2: ]))
        sites = list(sites)
        indices = {site: i for i, site in enumerate(sites)}
       
    # parse
    data = data.split("\n")[1: ]
    # how many teams there are
    n = len(sites)
    A = np.zeros((n, n)).astype(np.float64)
    
    # updata adjacency matrix
    for line in data:
        
        info = line.split(",")
        winner, loser = info[0], info[1]
        row, col = indices[winner], indices[loser] 
        A[row, col] += 1
        
    # put into graph object
    graph = DiGraph(A, labels = sites) 

    # return ranked teams via page-rank
    return get_ranks(graph.itersolve(epsilon = epsilon)) 
    

def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """
    Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    # init graph object 
    graph = nx.DiGraph() 
    
    # read in the data
    with open(filename, "r", encoding = "utf-8") as myfile:
        
        data = myfile.readlines()
        # iterate through the movies
        for movie in data:
            # find actors in each movie
            actors = movie.strip("\n").split("/")[1: ]
            
            for combo in combinations(actors, 2):
                
                a = combo[0]
                b = combo[1]
                # update graph accordingly
                if graph.has_edge(b, a):
                    
                    graph[b][a]["weight"] += 1
                
                else:
                    
                    graph.add_edge(b, a, weight = 1)
                    
    dictionary = nx.pagerank(graph, alpha = epsilon)
    
    # return ranks of actors via page-rank 
    return get_ranks(dictionary)
