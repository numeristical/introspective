import numpy as np

class graph_undirected(object):
    """This is a class to handle undirected graphs.  Still very much a work in progress.
    Defines a graph by a set of vertices and a set of "frozensets" representing the edges.
    """

    def __init__(self, edges, vertices={}):
        
        if vertices == {}:
            self.vertices = set([x for sublist in list(edges) for x in list(sublist) ])
        else:
            self.vertices = vertices
        new_edges = set()
        for edge in edges:
            new_edge = frozenset(edge)
            if len(new_edge)>1:
                new_edges.add(new_edge)
        self.edges = set(new_edges)


    def adjacent_edges(self, target_vertex):
        return set([x for x in self.edges if target_vertex in x])

    def adjacent_vertices(self, target_vertex):
        neighbors_and_self = set([x for sublist in self.adjacent_edges(target_vertex) for x in sublist])
        return set(neighbors_and_self)-set([target_vertex])

    def adjacent_vertices_to_set(self, target_vertex_set):
        templist = [list(self.adjacent_vertices(x)) for x in target_vertex_set]
        neighbors_and_self = [x for sublist in templist for x in sublist]
        return set(neighbors_and_self)-target_vertex_set
    
    def vertex_degree(self, target_vertex):
        return len(self.adjacent_vertices(target_vertex))

    def contract_edge(self, edge):
        return contract_edge(self, edge)

    def delete_vertex(self, vertex):
        return delete_vertex(self, vertex)

    def delete_vertices(self, vertex_set):
        return delete_vertices(self, vertex_set)

    def get_induced_subgraph(self, vertex_set):
        return get_induced_subgraph(self, vertex_set)


    def enumerate_all_partitions(self, max_size=0, memory_save=False, verbose=False):
        if(len(self.vertices)==0):
            return []
        if((len(self.vertices)>=1) and (len(self.vertices)<=2)):
            return [frozenset([list(self.vertices)[0]])]
        if max_size==0:
            max_size=int(np.floor(len(self.vertices)/2))
        good_partitions = []
        failed_partitions = []
        good_partitions.append(set())
        failed_partitions.append(set())
        num_good_partition_list = []
        num_failed_partition_list = []
        if verbose:
            print('Evaluating components of size 1')
        for vert in self.vertices:
            if is_connected(delete_vertex(self, vert)):
                good_partitions[0].add(frozenset({vert}))
            else:
                failed_partitions[0].add(frozenset({vert}))
        num_good_partition_list.append(len(good_partitions[0]))
        num_failed_partition_list.append(len(failed_partitions[0]))
        if verbose:
            print('num_good_partitions of comp_size 1 = {}'.format(num_good_partition_list[0]))
            print('num_failed_partitions of comp_size 1 = {}'.format(num_failed_partition_list[0]))
            print('Evaluating components of size 2')
        good_partitions.append(set())
        failed_partitions.append(set())
        for edge in self.edges:
            if is_connected(delete_vertices(self, edge)):
                good_partitions[1].add(edge)
            else:
                failed_partitions[1].add(edge)
        num_good_partition_list.append(len(good_partitions[1]))
        num_failed_partition_list.append(len(failed_partitions[1]))
        if verbose:
            print('num_good_partitions of comp_size 2 = {}'.format(num_good_partition_list[1]))
            print('num_failed_partitions of comp_size 2 = {}'.format(num_failed_partition_list[1]))
            print('num_total_good_partitions of comp_size<=2 is {}'.format(np.sum(num_good_partition_list)))
            print('num_total_failed_partitions of comp_size <=2 is {}'.format(np.sum(num_failed_partition_list)))

        
        for comp_size in range(3, max_size+1):
            good_partitions.append(set())
            failed_partitions.append(set())

            if verbose:
                print('Evaluating components of size {}'.format(comp_size))
            base_components = good_partitions[comp_size-2]
            for base_comp in base_components:
                neighbors_to_add = self.adjacent_vertices_to_set(base_comp)
                for neighbor in neighbors_to_add:
                    new_comp = set(base_comp)
                    new_comp.add(neighbor)
                    new_comp = frozenset(new_comp)
                    if ((not new_comp in good_partitions[comp_size-1]) and (not new_comp in failed_partitions[comp_size-1])):
                        if is_connected(delete_vertices(self,new_comp)):
                            good_partitions[comp_size-1].add(new_comp)
                        else:
                            failed_partitions[comp_size-1].add(new_comp)
            num_good_partition_list.append(len(good_partitions[comp_size-1]))
            num_failed_partition_list.append(len(failed_partitions[comp_size-1]))
            if memory_save:
                good_partitions[comp_size-2]={}
                failed_partitions[comp_size-2]={}
                
            if verbose:
                print('num_good_partitions of comp_size {} = {}'.format(comp_size,num_good_partition_list[comp_size-1]))
                print('num_failed_partitions of comp_size {} = {}'.format(comp_size,num_failed_partition_list[comp_size-1]))
                print('num_total_good_partitions of comp_size<={} is {}'.format(comp_size, np.sum(num_good_partition_list)))
                print('num_total_failed_partitions of comp_size <={} is {}'.format(comp_size, np.sum(num_failed_partition_list)))
             
        good_partitions = [k for templist in good_partitions for k in templist]
        return good_partitions

    
def contract_edge(graph, edge):
    edge_alph = list(edge)
    edge_alph.sort()
    contracted_vertex = '_'.join((edge_alph))
    #new_vertices = (set(graph.vertices) - set(edge)).union({contracted_vertex})
    new_edges = [[contracted_vertex if y==edge_alph[0] or y==edge_alph[1] else y for y in this_edge] 
                 if edge_alph[0] in this_edge  or edge_alph[1] in this_edge else this_edge for this_edge in graph.edges]
    return graph_undirected(new_edges)

def delete_vertex(graph, vertex):
    new_edges = set([edge for edge in graph.edges if vertex not in edge])
    new_vertices = graph.vertices - {vertex}
    return graph_undirected(new_edges, new_vertices)

def delete_vertices(graph, vertex_set):
    new_edges = set([edge for edge in graph.edges if not vertex_set.intersection(edge)])
    new_vertices = graph.vertices - vertex_set
    return graph_undirected(new_edges, new_vertices)

def get_induced_subgraph(graph, vertex_set):
    vertex_set = set(vertex_set)
    new_edges = set([edge for edge in graph.edges if edge <= vertex_set])
    new_vertices = vertex_set
    return graph_undirected(new_edges, new_vertices)


def is_connected(graph):
    initial_vertex = list(graph.vertices)[0]
    visited_vertices = [initial_vertex]
    unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
    while unexplored_vertices:
        curr_vertex = unexplored_vertices.pop()
        visited_vertices.append(curr_vertex)
        new_vertices = graph.adjacent_vertices(curr_vertex)
        unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
    return len(set(visited_vertices)) == len(set(graph.vertices))

def num_connected_comp(graph):
    initial_vertex = list(graph.vertices)[0]
    visited_vertices = [initial_vertex]
    unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
    while unexplored_vertices:
        curr_vertex = unexplored_vertices.pop(0)
        visited_vertices.append(curr_vertex)
        new_vertices = graph.adjacent_vertices(curr_vertex)
        unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
    if len(set(visited_vertices)) == len(set(graph.vertices)):
        return 1
    else:
        remainder_vertices = list(set(graph.vertices)-set(visited_vertices))
        remainder_edges = [edge for edge in graph.edges if edge.issubset(set(remainder_vertices))]
        return 1 + num_connected_comp(graph_undirected(remainder_edges, remainder_vertices))
    
def connected_comp_list(graph):
    initial_vertex = list(graph.vertices)[0]
    visited_vertices = [initial_vertex]
    unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
    while unexplored_vertices:
        curr_vertex = unexplored_vertices.pop(0)
        visited_vertices.append(curr_vertex)
        new_vertices = graph.adjacent_vertices(curr_vertex)
        unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
    if len(set(visited_vertices)) == len(set(graph.vertices)):
        return [graph]
    else:
        cc_vertices = set(visited_vertices)
        cc_edges = [edge for edge in graph.edges if edge.issubset(set(visited_vertices))]
        cc_graph = graph_undirected(cc_edges, cc_vertices)
        remainder_vertices = list(set(graph.vertices)-set(visited_vertices))
        remainder_edges = [edge for edge in graph.edges if edge.issubset(set(remainder_vertices))]
        return [cc_graph] + connected_comp_list(graph_undirected(remainder_edges, remainder_vertices))
    
def get_all_distances_from_vertex(graph, start_vertex):
    vertex_path_dist_dict={}
    vertex_path_dist_dict[start_vertex] = 0
    unexplored_vertices = list(graph.adjacent_vertices(start_vertex))
    for vert in unexplored_vertices:
        vertex_path_dist_dict[vert]=1
    visited_vertices = [start_vertex]
    
    while unexplored_vertices and (len(vertex_path_dist_dict.keys())<len(graph.vertices)):
        curr_vertex = unexplored_vertices.pop()
        curr_dist = vertex_path_dist_dict[curr_vertex]
        visited_vertices.append(curr_vertex)
        curr_neighbors = graph.adjacent_vertices(curr_vertex)
        new_vertices = curr_neighbors - set(vertex_path_dist_dict.keys())
        for vert in new_vertices:
            vertex_path_dist_dict[vert]=curr_dist+1
        unexplored_vertices = list(new_vertices) + unexplored_vertices
    return vertex_path_dist_dict

def separate_by_two_vertices(graph, vert_1, vert_2):
    dict_1 = get_all_distances_from_vertex(graph, vert_1)
    dict_2 = get_all_distances_from_vertex(graph, vert_2)
    comp_2 = set([vert for vert in dict_2.keys() if dict_2[vert]<dict_1[vert]])
    comp_1 = graph.vertices - comp_2
    return comp_1, comp_2
