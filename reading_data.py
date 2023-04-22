import torch
from rdflib import Graph, Literal
import gzip

"""
Functionalities:
- reads in data
    - can be either .gz or .nt files
- creates adjacency matrix
    - literal options: "filtered", "collapsed", "separate", "all-to-one"
    - relational options: False and True
    - two types: dense and coo
        - coo provided in the sparse torch tensors
        - dense provided in torch tensors
- create label tensor
    - also get the indices for the training, validation, and test indices

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

    assert file_name.endswith(".nt.gz") or file_name.endswith(".nt"), "Please put in a .nt or a .nt.gz file."

    # parse the graph using rdflib
    graph = Graph()

    # depends on whether a nt or a gzip file is read in:
    if file_name.endswith(".gz"):
        with gzip.open(file_name, 'r') as gf:
            graph.parse(data=gf.read(), format='nt')
    else:
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
        # create four separate lists - input for the coo matrix later
        dim_0 = list()  # for the head of the triple
        dim_1 = list()  # for the tail of the triple
        dim_2 = list()  # only used in case it is relational - for the relation
        values = list()  # will only contain 1's -- the strength of the connections
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

    # only for the entities, as it needs to be used to make the labels feature later:
    map_ent_to_ind = dict()

    # increments so every entity gets a different index in the adjacency matrix
    current = 0

    # create a mapping for each unique entity
    for entity in list(entities):
        # only used to create the adjacency matrix
        map_nod_to_ind[entity] = current

        # only used to return from the function
        map_ent_to_ind[entity] = current

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

    # put all the relations in the pre-made adjacency matrix - or in the lists defined before, if sparse
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
                values.append(1)
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
                # the filtered one has a different size (nr of relations) than the other ones
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
        return adjacency_matrix, map_ind_to_nod, map_ent_to_ind, map_ind_to_rel

    return adjacency_matrix, map_ind_to_nod, map_ent_to_ind


def training_valid_test_set(file_name_train, file_name_valid, file_name_test, mapping_ent_to_ind, number_nodes):
    """
    Create a training, validation, and test set for the labels, using the mapping from entity to index used before--
    to make sure that the correct indices are masked in the created adjacency matrix to compare with these labels.

    :param file_name_train: The file name of the triples representing the train indices.
    :param file_name_valid: The file name of the triples representing the validation indices.
    :param file_name_test: The file name of the triples representing the test indices.
    :param mapping_ent_to_ind: The pre-made mapping of the entities to their indices in the adjacency matrix.
    :param number_nodes: The number of nodes in the adjacency matrix.
    :return: The tensor with labels, the tensors with the indices of the training, validation, and test set respectively
    """

    # read in the files (either in .nt or .nt.gz format):
    graph_train = Graph()
    graph_valid = Graph()
    graph_test = Graph()

    if file_name_train.endswith(".gz"):
        with gzip.open(file_name_train, 'r') as gf:
            graph_train.parse(data=gf.read(), format='nt')
    else:
        graph_train.parse(file_name_train)

    if file_name_valid.endswith(".gz"):
        with gzip.open(file_name_valid, 'r') as gf:
            graph_valid.parse(data=gf.read(), format='nt')
    else:
        graph_valid.parse(file_name_valid)

    if file_name_test.endswith(".gz"):
        with gzip.open(file_name_test, 'r') as gf:
            graph_test.parse(data=gf.read(), format='nt')
    else:
        graph_test.parse(file_name_test)

    graph_all = graph_train + graph_valid + graph_test

    # create a zero-tensor for the labels
    labels = torch.neg(torch.ones(number_nodes))
    class_mapping = dict()
    current = 0

    # loop over all relations --> to create a tensor with all classes stored in it
    for head, relation, tail in graph_all:
        # get the index of head:
        head_index = mapping_ent_to_ind[head]

        # if current class not yet in map, add it and increment
        if tail not in class_mapping:
            class_mapping[tail] = current
            labels[head_index] = current

            current += 1

        # otherwise, add the class index to the tensor at that point
        else:
            label = class_mapping[tail]
            labels[head_index] = label

    # create lists for the indices of the training
    train_entities = list()
    valid_entities = list()
    test_entities = list()

    # add the mapped versions to the lists
    for head, relation, tail in graph_train:
        train_entities.append(mapping_ent_to_ind[head])

    for head, relation, tail in graph_valid:
        valid_entities.append(mapping_ent_to_ind[head])

    for head, relation, tail in graph_test:
        test_entities.append(mapping_ent_to_ind[head])

    # create tensors out of it
    train_entities = torch.LongTensor(train_entities)
    valid_entities = torch.LongTensor(valid_entities)
    test_entities = torch.LongTensor(test_entities)

    return labels, train_entities, valid_entities, test_entities, class_mapping


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
