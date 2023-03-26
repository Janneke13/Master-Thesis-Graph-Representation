import torch
from rdflib import Graph, Literal
import scipy

"""
Functionalities:
- reads in data
- creates adjacency matrix
    - literal options: "filtered", "collapsed", "separate", "all-to-one"
    - relational options: False and True
    - two types: dense and coo
        - coo provided in the sparse torch tensors
        - dense provided in torch tensors

Note: "create_adjacency_matrix_nt", and "divide_entities_relations_literals"
both run in O(n) time.
"""


def create_adjacency_matrix_nt(file_name, literal_representation="filtered", relational=False, sparse=False):
    """
    Creates an adjacency matrix with certain literal representations of a file of a graph (in .nt format).
    NOTE: dense uses more memory than sparse, so sparse might be preferred.

    :param file_name: The file name of the graph that the adjacency matrix needs to be created of.
    :param literal_representation: The way the literals are structurally represented. "filtered" if they are filtered
        out, "collapsed" if literals with the same value are collapsed to the same node, "separate" if the literals are
        all separated, "all-to-one" if all the literals (regardless of value) are mapped to the same index.
    :param relational: Boolean denoting whether the relations have to be taken into account.
    :param sparse: Boolean denoting whether the adjacency matrix needs to be in scipy.sparse.coo_matrix format, if not
    then torch tensors are used.
    :return: The adjacency matrix of the given file, the mapping (ind->ent), possibly the ind->rel mapping.
    """

    # Send an error message if the literal representation is not correct.
    assert literal_representation in ["filtered", "collapsed",
                                      "separate", "all-to-one"], "Literal representation " + literal_representation + \
                                                                 " is not valid. Please enter one of the following: " \
                                                                 "'filtered', 'collapsed', 'separate', 'all-to-one'."

    # parse the graph using rdflib
    graph = Graph()
    graph.parse(file_name)

    # get sets of the entities, relations, and literals, get the count of literals
    entities, literals, relations, relations_without_literals, nr_literals = divide_entities_relations_literals(graph)

    # create a variable for the number of nodes
    number_nodes = 0

    # get the number of nodes, for each type of representation:
    if literal_representation == "filtered":
        number_nodes = len(entities)
    elif literal_representation == "collapsed":
        number_nodes = len(entities) + len(literals)
    elif literal_representation == "separate":
        number_nodes = len(entities) + nr_literals
    elif literal_representation == "all-to-one":
        number_nodes = len(entities) + 1  # map all literals to the same index

    if sparse:
        # create three separate lists - input for the coo matrix later
        dim_0 = list()
        dim_1 = list()
        dim_2 = list()  # only used in case it is relational
        values = list()
    else:
        # create an (empty) adjacency matrix in torch for the dense matrix case:
        if not relational:
            adjacency_matrix = torch.zeros(number_nodes, number_nodes)
        else:
            # 3D matrix --> with number of relations
            if literal_representation == "filtered":
                adjacency_matrix = torch.zeros(number_nodes, number_nodes, len(relations_without_literals))
            else:
                adjacency_matrix = torch.zeros(number_nodes, number_nodes, len(relations))

    # create a mapping for rows/columns representing nodes, to make lookup faster
    map_nod_to_ind = dict()
    map_ind_to_nod = dict()

    # increments so every entity gets a different index in the adjacency matrix
    current = 0

    # create a mapping for each unique entity
    for entity in list(entities):
        map_nod_to_ind[entity] = current
        map_ind_to_nod[current] = entity
        current += 1

    # add a general literal node for all literals - only in this mapping, as it is done separately in the storing
    if literal_representation == "all-to-one":
        map_ind_to_nod[current] = "literal"

    # create a mapping for each unique literal --> note: only for the "collapsed" mode
    if literal_representation == "collapsed":
        for literal in list(literals):
            map_nod_to_ind[literal] = current
            map_ind_to_nod[current] = literal
            current += 1

    # if a relational adjacency matrix is being created, take relation into account --> map these in a similar way
    # will be empty if the matrix, to be created, is not relational
    map_rel_to_ind = dict()
    map_ind_to_rel = dict()

    if relational:
        current_rel = 0
        if literal_representation == "filtered":
            for relation in list(relations_without_literals):
                map_rel_to_ind[relation] = current_rel
                map_ind_to_rel[current_rel] = relation
                current_rel += 1
        else:
            for relation in list(relations):
                map_rel_to_ind[relation] = current_rel
                map_ind_to_rel[current_rel] = relation
                current_rel += 1

    # put all the relations in the pre-made adjacency matrix
    for head, relation, tail in graph:
        # find the location of the head node
        row_selected = map_nod_to_ind[head]

        # check whether the tail node is already in the mapping
        if not isinstance(tail, Literal) or literal_representation == "collapsed":
            column_selected = map_nod_to_ind[tail]

        # if it is a literal, and the mapping is all-to-one --> all literals to the same node
        elif literal_representation == "all-to-one":
            column_selected = current

        # if the literals are needed separately, it works slightly differently - create the mapping while adding them
        elif literal_representation == "separate":
            row_selected = map_nod_to_ind[head]
            column_selected = current  # select a new column for every literal

            # map the literal to an index as well - this is the only mapping returned later, so it is relevant
            map_ind_to_nod[current] = tail
            current += 1

        if (literal_representation == "filtered" and not isinstance(tail, Literal)) \
                or literal_representation != "filtered":

            if sparse:
                # add it to every list needed for the sparse format
                dim_0.append(row_selected)
                dim_1.append(column_selected)
                values.append(1.)
                if relational:
                    relation_selected = map_rel_to_ind[relation]
                    dim_2.append(relation_selected)
            else:
                # for dense matrices, we can add the number of relations directly
                # for relations, everything needs to be mapped to its own relation as well
                if not relational:
                    adjacency_matrix[row_selected, column_selected] += 1
                else:
                    relation_selected = map_rel_to_ind[relation]
                    adjacency_matrix[row_selected, column_selected, relation_selected] += 1

    # in case the sparse format is used, create the sparse adjacency matrices
    if sparse:
        if relational:
            # for relational, make a 3D coo matrix
            if literal_representation == "filtered":
                adjacency_matrix = torch.sparse_coo_tensor(indices=torch.tensor([dim_0, dim_1, dim_2]),
                                                           values=torch.tensor(values),
                                                           size=(number_nodes, number_nodes,
                                                                 len(relations_without_literals)))
            else:
                adjacency_matrix = torch.sparse_coo_tensor(indices=torch.tensor([dim_0, dim_1, dim_2]),
                                                           values=torch.tensor(values),
                                                           size=(number_nodes, number_nodes, len(relations)))
        else:
            # else, make a 2d matrix
            adjacency_matrix = torch.sparse_coo_tensor(indices=torch.tensor([dim_0, dim_1]),
                                                       values=torch.tensor(values),
                                                       size=(number_nodes, number_nodes))

    # return the relational mapping as well if a relational matrix is created
    if relational:
        return adjacency_matrix, map_ind_to_nod, map_ind_to_rel

    return adjacency_matrix, map_ind_to_nod


def divide_entities_relations_literals(kg):
    """
    Makes separate sets of the entities and literals, counts the total number of literals.

    :param kg: The knowledge graph, as a rdflib Graph object.
    :return: The sets of entities and literals, as well as the total number of literals.
    """
    # creates sets of entities and literals, counts the number of literals
    entities = set()
    relations = set()
    relations_without_literals = set()
    literals = set()
    nr_literals_total = 0

    for head, relation, tail in kg:
        # note: heads can ONLY be entities
        entities.add(head)
        relations.add(relation)

        # add the tail to its respective set:
        if isinstance(tail, Literal):
            literals.add(tail)
            nr_literals_total += 1
        else:
            entities.add(tail)
            relations_without_literals.add(relation)

    return entities, literals, relations, relations_without_literals, nr_literals_total
