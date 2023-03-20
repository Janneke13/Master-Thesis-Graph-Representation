import torch
from rdflib import Graph, Literal

"""
This file:
- reads in data
- creates adjacency matrix
- handles different file types
"""


def create_adjacency_matrix_nt(file_name, literal_representation="filtered"):
    """
    Creates an adjacency matrix with certain literal representations of a file of a graph (in .nt format).

    :param file_name: The file name of the graph that the adjacency matrix needs to be created of.
    :param literal_representation: The way the literals are structurally represented. "filtered" if they are filtered
        out, "collapsed" if literals with the same value are collapsed to the same node, "separate" if the literals are
        all separated.
    :return: The adjacency matrix of the given file, the mapping (ind->ent)
    """

    # Send an error message if the literal representation is not correct.
    assert literal_representation in ["filtered", "collapsed",
                                          "separate"], "Literal representation " + literal_representation + \
                                                       " is not valid. Please enter one of the following: 'filtered'," \
                                                       " 'collapsed', 'separate'."

    # parse the graph using rdflib
    graph = Graph()
    graph.parse(file_name)

    # get sets of the entities and literals, get the count of literals
    entities, literals, nr_literals = divide_entities_literals(graph)

    # get the number of nodes, for each type of representation:
    if literal_representation == "filtered":
        number_nodes = len(entities)
    elif literal_representation == "collapsed":
        number_nodes = len(entities) + len(literals)
    elif literal_representation == "separate":
        number_nodes = len(entities) + nr_literals

    # create an (empty) adjacency matrix in torch
    adjacency_matrix = torch.zeros(number_nodes, number_nodes)

    # create a mapping for rows/columns representing nodes--> for quicker look-up
    map_nod_to_ind = dict()
    map_ind_to_nod = dict()

    # increments so every entity gets a different index
    current = 0

    # create a mapping for each unique entity!
    for entity in list(entities):
        map_nod_to_ind[entity] = current
        map_ind_to_nod[current] = entity
        current += 1

    # create a mapping for each unique literal --> note: only for the "collapsed" mode
    if literal_representation == "collapsed":
        for literal in list(literals):
            map_nod_to_ind[literal] = current
            map_ind_to_nod[current] = literal
            current += 1

    # put all the adjacencies in the pre-made matrix
    for head, relation, tail in graph:

        # check whether this is a relational triple and if so, add the connection
        # if the literals are included in the mapping already, the connection can also be added
        if not isinstance(tail, Literal) or literal_representation == "collapsed":
            # find out where one needs to be added:
            row_selected = map_nod_to_ind[head]
            column_selected = map_nod_to_ind[tail]
            adjacency_matrix[row_selected, column_selected] += 1

        # if the literals are needed separately, it works slightly differently - create the mapping while adding them
        elif literal_representation == "separate":
            row_selected = map_nod_to_ind[head]
            column_selected = current  # select a new column for every literal

            # map the literal to an index as well - this is the only mapping returned later, so it is relevant
            map_ind_to_nod[current] = tail

            adjacency_matrix[row_selected, column_selected] += 1
            current += 1

    return adjacency_matrix, map_ind_to_nod


def divide_entities_literals(kg):
    """
    Makes separate sets of the entities and literals, counts the total number of literals.

    :param kg: The knowledge graph, as an rdflib Graph object.
    :return: The sets of entities and literals, as well as the total number of literals.
    """
    # creates sets of entities and literals, counts the number of literals
    entities = set()
    literals = set()
    nr_literals_total = 0

    for head, relation, tail in kg:
        # note: heads can ONLY be entities
        entities.add(head)

        # add the tail to its respective set:
        if isinstance(tail, Literal):
            literals.add(tail)
            nr_literals_total += 1
        else:
            entities.add(tail)

    return entities, literals, nr_literals_total
