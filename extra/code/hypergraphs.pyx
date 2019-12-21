import numpy as np
import json

class hypergraph(object):
    """This is a class to handle hypergraphs. 
    Defines a hypergraph by a set of vertices and a set of "frozensets" representing the edges.
    """

    def __init__(self, edges, vertices=set(), add_singletons=False):
        
        if vertices == set():
            self.vertices = set([x for sublist in list(edges) for x in list(sublist) ])
        else:
            self.vertices = vertices
        new_edges = set()
        for edge in edges:
            new_edge = frozenset(edge)
            if len(new_edge)>=1:
                new_edges.add(new_edge)
        if add_singletons:
            for vertex in self.vertices:
                new_edges.add(frozenset([vertex]))
        self.edges = set(new_edges)
        self.mc_partitions = []
        self.mc_partitions_max_size = 0
        self.all_connected_sets = []
        self.partition_dict = {}


    def adjacent_edges(self, target_vertex):
        return set([x for x in self.edges if target_vertex in x])

    def sets_adjacent(self, frozenset vert_name_set_1, frozenset vert_name_set_2):
        return(bool(vert_name_set_1.intersection(vert_name_set_2)))


    def adjacent_vertices(self, target_vertex):
        neighbors_and_self = set([x for sublist in self.adjacent_edges(target_vertex) for x in sublist])
        return set(neighbors_and_self)-set([target_vertex])

    def adjacent_vertices_to_set(self, target_vertex_set):
        templist = [list(self.adjacent_vertices(x)) for x in target_vertex_set]
        neighbors_and_self = [x for sublist in templist for x in sublist]
        return set(neighbors_and_self)-target_vertex_set

    # def adjacent_vertices_to_set_2(self, target_vertex_set):
    #     out_set = {x for subset in self.edges if self.sets_adjacent(subset, target_vertex_set) for x in subset}
    #     return(out_set - set(target_vertex_set))
    
    def vertex_degree(self, target_vertex):
        return len(self.adjacent_vertices(target_vertex))

    # def contract_edge(self, edge, sep_str='_'):
    #     return contract_edge(self, edge, sep_str)

    def delete_vertex(self, vertex):
        return delete_vertex(self, vertex)

    def delete_vertices(self, vertex_set):
        return delete_vertices(self, vertex_set)

    def get_induced_subgraph(self, vertex_set):
        return get_induced_subgraph(self, vertex_set)

    def generate_small_size_partitions(self, max_size=2):
        edge_list = list(self.edges)
        self.partition_dict[(1,1)] 
        if (self.num_vertices, 2) not in self.partition_dict.keys():
            self.partition_dict[(self.num_vertices, 2)] = {frozenset([i,j]) for i in edge_list for j in edge_list  if (not i.intersection(j) and i.union(j)==self.vertices)}
        for curr_size in range(3, max_size):
            if (0,curr_size-1) not in self.partition_dict.keys():
                self.partition_dict[(0, curr_size-1)] = {frozenset([i,j]) for i in edge_list for j in edge_list  if (not i.intersection(j))}

    # def return_mc_partitions(self):
    #     if self.mc_partitions==[]:
    #         self.enumerate_mc_partitions()
    #     return(self.mc_partitions)


#     def enumerate_mc_partitions(self, max_size=0, verbose=False):
#         """This method will examine every connected set S of size up to max_size and
#         determine whether or not the complement of the set is also connected.  If the
#         complement is also connected, then the partition {S, S^C} is added to the list
#         self.mc_partitions"""
        
#         # Default behavior is to find all maximally coarse partitions which
#         # requires searching components up to size floor(n_vertices/2)
#         if max_size==0:
#             max_size=int(np.floor(len(self.vertices)/2))
        
#         # Initialize some variables
#         # The two lists below are sets of sets by size.
#         # i.e. conn_sets_with_conn_complements_by_size[5] will be a set that contains
#         # the connected sets of size 5 whose complements are also connected
#         conn_sets_with_conn_complements_by_size = []
#         conn_sets_with_disconn_complements_by_size = []
        
#         # These two contain the sizes of each entry in the above lists
#         num_conn_sets_with_conn_complements_list = []
#         num_conn_sets_with_disconn_complements_list = []
        
#         # Initialize the list with an empty set
#         conn_sets_with_conn_complements_by_size.append(set())
#         conn_sets_with_disconn_complements_by_size.append(set())


#         # Corner case handling
#         if(len(self.vertices)==0):
#             return []
#         if((len(self.vertices)>=1) and (len(self.vertices)<=2)):
#             return [frozenset([list(self.vertices)[0]])]
        
#         # The connected components of size 1 are exactly the vertices
#         if verbose:
#             print('Evaluating connected sets of size 1')
#         for vert in self.vertices:
#             if is_connected(delete_vertex(self, vert)):
#                 conn_sets_with_conn_complements_by_size[0].add(frozenset({vert}))
#             else:
#                 conn_sets_with_disconn_complements_by_size[0].add(frozenset({vert}))
#         num_conn_sets_with_conn_complements_list.append(len(conn_sets_with_conn_complements_by_size[0]))
#         num_conn_sets_with_disconn_complements_list.append(len(conn_sets_with_disconn_complements_by_size[0]))
#         if verbose:
#             print('num conn sets of comp_size 1 with connected complements = {}'.format(num_conn_sets_with_conn_complements_list[0]))
#             print('num conn sets of comp_size 1 with disconnected complements = {}'.format(num_conn_sets_with_disconn_complements_list[0]))
#             print('Evaluating connected sets of size 2')
#         conn_sets_with_conn_complements_by_size.append(set())
#         conn_sets_with_disconn_complements_by_size.append(set())
        
#         # The connected components of size 2 are exactly the edges
#         for edge in self.edges:
#             if is_connected(delete_vertices(self, edge)):
#                 conn_sets_with_conn_complements_by_size[1].add(edge)
#             else:
#                 conn_sets_with_disconn_complements_by_size[1].add(edge)
#         num_conn_sets_with_conn_complements_list.append(len(conn_sets_with_conn_complements_by_size[1]))
#         num_conn_sets_with_disconn_complements_list.append(len(conn_sets_with_disconn_complements_by_size[1]))
#         if verbose:
#             print('num conn sets of comp_size 2 with connected complements = {}'.format(num_conn_sets_with_conn_complements_list[1]))
#             print('num conn sets of comp_size 2 with disconnected complements = {}'.format(num_conn_sets_with_disconn_complements_list[1]))
#             print('num conn sets of comp_size <=2 with connected complements = {}'.format(np.sum(num_conn_sets_with_conn_complements_list)))
#             print('num conn sets of comp_size <=2 with disconnected complements = {}'.format(np.sum(num_conn_sets_with_disconn_complements_list)))

        
#         for comp_size in range(3, max_size+1):
#             conn_sets_with_conn_complements_by_size.append(set())
#             conn_sets_with_disconn_complements_by_size.append(set())

#             if verbose:
#                 print('Evaluating connected sets of size {}'.format(comp_size))
#             base_components = conn_sets_with_conn_complements_by_size[comp_size-2].union(conn_sets_with_disconn_complements_by_size[comp_size-2])
#             for base_comp in base_components:
#                 neighbors_to_add = self.adjacent_vertices_to_set(base_comp)
#                 for neighbor in neighbors_to_add:
#                     new_comp = set(base_comp)
#                     new_comp.add(neighbor)
#                     new_comp = frozenset(new_comp)
#                     if ((not new_comp in conn_sets_with_conn_complements_by_size[comp_size-1]) and (not new_comp in conn_sets_with_disconn_complements_by_size[comp_size-1])):
#                         if is_connected(delete_vertices(self,new_comp)):
#                             conn_sets_with_conn_complements_by_size[comp_size-1].add(new_comp)
#                         else:
#                             conn_sets_with_disconn_complements_by_size[comp_size-1].add(new_comp)
#             num_conn_sets_with_conn_complements_list.append(len(conn_sets_with_conn_complements_by_size[comp_size-1]))
#             num_conn_sets_with_disconn_complements_list.append(len(conn_sets_with_disconn_complements_by_size[comp_size-1]))
                
#             if verbose:
#                 print('num conn set of comp_size {} with connected complements= {}'.format(comp_size,num_conn_sets_with_conn_complements_list[comp_size-1]))
#                 print('num conn set of comp_size {} with discconnected complements= {}'.format(comp_size,num_conn_sets_with_disconn_complements_list[comp_size-1]))
#                 print('num conn set of comp_size <= {} with connected complements= {}'.format(comp_size, np.sum(num_conn_sets_with_conn_complements_list)))
#                 print('num conn set of comp_size <= {} with disconnected complements= {}'.format(comp_size, np.sum(num_conn_sets_with_disconn_complements_list)))
             
#         self.mc_partitions = list(set([frozenset([conn_set, frozenset(self.vertices - conn_set)]) for templist in conn_sets_with_conn_complements_by_size for conn_set in templist]))
#         #self.mc_partitions = [[conn_set, self.vertices - conn_set] for conn_set in conn_sets_with_conn_complements]
#         self.mc_partitions_max_size = max_size
        

#     def save_partitions_to_file(self, file_name):
#         list_of_lists = [list(x) for x in self.all_partitions]
#         with open(file_name, "w") as write_file:
#             json.dump(list_of_lists, write_file)

#     def load_partitions_from_file(self, file_name):
#         with open(file_name, "r") as read_file:
#             list_of_lists = json.load(read_file)
#         self.all_partitions = [frozenset(x) for x in list_of_lists]

#     def enumerate_connected_sets(self, max_size=-1, verbose=False):
#         if self.all_connected_sets:
#             return self.all_connected_sets
#         if(len(self.vertices)==0):
#             return []
#         if((len(self.vertices)>=1) and (len(self.vertices)<=2)):
#             return [frozenset([list(self.vertices)[0]])]
#         if max_size==(-1):
#             max_size=len(self.vertices)
#         connected_sets = []
#         connected_sets.append(set())
#         num_connected_sets_list = []
#         if verbose:
#             print('Evaluating components of size 1')
#         for vert in self.vertices:
#             connected_sets[0].add(frozenset({vert}))
#         num_connected_sets_list.append(len(connected_sets[0]))
#         if verbose:
#             print('num connected sets of size 1 = {}'.format(num_connected_sets_list[0]))
#             print('Evaluating components of size 2')
#         connected_sets.append(set())
#         for edge in self.edges:
#             connected_sets[1].add(edge)
#         num_connected_sets_list.append(len(connected_sets[1]))
#         if verbose:
#             print('num_connected_sets of size 2 = {}'.format(num_connected_sets_list[1]))
#             print('num_connected_sets of size<=2 is {}'.format(np.sum(num_connected_sets_list)))

        
#         for comp_size in range(3, max_size+1):
#             connected_sets.append(set())

#             if verbose:
#                 print('Evaluating components of size {}'.format(comp_size))
#             base_components = connected_sets[comp_size-2]
#             for base_comp in base_components:
#                 neighbors_to_add = self.adjacent_vertices_to_set(base_comp)
#                 for neighbor in neighbors_to_add:
#                     new_comp = set(base_comp)
#                     new_comp.add(neighbor)
#                     new_comp = frozenset(new_comp)
#                     connected_sets[comp_size-1].add(new_comp)
#             num_connected_sets_list.append(len(connected_sets[comp_size-1]))
#             # if memory_save:
#             #     good_partitions[comp_size-2]=set()
#             #     failed_partitions[comp_size-2]=set()
                
#             if verbose:
#                 print('num_connected_sets of size {} = {}'.format(comp_size,num_connected_sets_list[comp_size-1]))
#                 print('num_total_connected_sets of size<={} is {}'.format(comp_size, np.sum(num_connected_sets_list)))
             
#         connected_sets = [k for templist in connected_sets for k in templist]
#         self.all_connected_sets = connected_sets
#         return connected_sets
    
#     def save_connected_sets_to_file(self, file_name):
#         list_of_lists = [list(x) for x in self.all_connected_sets]
#         with open(file_name, "w") as write_file:
#             json.dump(list_of_lists, write_file)

#     def load_connected_sets_from_file(self, file_name):
#         with open(file_name, "r") as read_file:
#             list_of_lists = json.load(read_file)
#         self.all_connected_sets = [frozenset(x) for x in list_of_lists]

#     def get_partitions_from_connected_sets(self, verbose=False, verbose_freq=1000):
#         part_list = []
#         conn_set_list = self.all_connected_sets.copy()
#         conn_set_set = set(self.all_connected_sets)
#         if verbose:
#             print('checking {} connected sets'.format(len(conn_set_list)))
#         for i,conn_set in enumerate(conn_set_list):
#             if len(conn_set) > (len(self.vertices)/2):
#                 break
#             complement_set = frozenset(self.vertices - conn_set)
#             if complement_set in conn_set_set:
#                 part_list.append(conn_set)
#                 conn_set_list.remove(complement_set)
#             if ((((i+1) % verbose_freq)) ==0):
#                 if verbose:
#                     print('Checked {} sets'.format(i+1))
#                     print('Found {} partitions'.format(len(part_list)))
#         self.all_partitions = part_list


# def contract_edge(graph, edge, sep_str='_'):
#     edge_alph = list(edge)
#     edge_alph.sort()
#     contracted_vertex = sep_str.join((edge_alph))
#     #new_vertices = (set(graph.vertices) - set(edge)).union({contracted_vertex})
#     new_edges = [[contracted_vertex if y==edge_alph[0] or y==edge_alph[1] else y for y in this_edge] 
#                  if edge_alph[0] in this_edge  or edge_alph[1] in this_edge else this_edge for this_edge in graph.edges]
#     return graph_undirected(new_edges)

def delete_vertex(graph, vertex):
    new_edges = set([frozenset(edge - set(vertex)) for edge in graph.edges])
    new_vertices = graph.vertices - {vertex}
    return hypergraph(new_edges, new_vertices)

def delete_vertices(graph, vertex_set):

    new_edges = set([frozenset(edge-set(vertex_set)) for edge in graph.edges])
    new_vertices = graph.vertices - vertex_set
    return hypergraph(new_edges, new_vertices)

def get_induced_subgraph(graph, vertex_set):
    vertex_set = set(vertex_set)
    new_edges = set([edge for edge in graph.edges if edge <= vertex_set])
    new_vertices = vertex_set
    new_graph = hypergraph(new_edges, new_vertices)
    new_graph.all_connected_sets = [x for x in graph.all_connected_sets if new_vertices.issuperset(x)]
    return new_graph


# def is_connected(graph):
#     initial_vertex = next(iter(graph.vertices))
#     visited_vertices = [initial_vertex]
#     unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
#     while unexplored_vertices:
#         curr_vertex = unexplored_vertices.pop()
#         visited_vertices.append(curr_vertex)
#         new_vertices = graph.adjacent_vertices(curr_vertex)
#         unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
#     return len(set(visited_vertices)) == len(set(graph.vertices))

# def num_connected_comp(graph):
#     initial_vertex = list(graph.vertices)[0]
#     visited_vertices = [initial_vertex]
#     unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
#     while unexplored_vertices:
#         curr_vertex = unexplored_vertices.pop(0)
#         visited_vertices.append(curr_vertex)
#         new_vertices = graph.adjacent_vertices(curr_vertex)
#         unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
#     if len(set(visited_vertices)) == len(set(graph.vertices)):
#         return 1
#     else:
#         remainder_vertices = list(set(graph.vertices)-set(visited_vertices))
#         remainder_edges = [edge for edge in graph.edges if edge.issubset(set(remainder_vertices))]
#         return 1 + num_connected_comp(graph_undirected(remainder_edges, remainder_vertices))
    
# def connected_comp_list(graph):
#     initial_vertex = list(graph.vertices)[0]
#     visited_vertices = [initial_vertex]
#     unexplored_vertices = list(graph.adjacent_vertices(initial_vertex))
#     while unexplored_vertices:
#         curr_vertex = unexplored_vertices.pop(0)
#         visited_vertices.append(curr_vertex)
#         new_vertices = graph.adjacent_vertices(curr_vertex)
#         unexplored_vertices = list(set(unexplored_vertices).union(new_vertices) - set(visited_vertices))
#     if len(set(visited_vertices)) == len(set(graph.vertices)):
#         return [graph]
#     else:
#         cc_vertices = set(visited_vertices)
#         cc_edges = [edge for edge in graph.edges if edge.issubset(set(visited_vertices))]
#         cc_graph = graph_undirected(cc_edges, cc_vertices)
#         remainder_vertices = list(set(graph.vertices)-set(visited_vertices))
#         remainder_edges = [edge for edge in graph.edges if edge.issubset(set(remainder_vertices))]
#         return [cc_graph] + connected_comp_list(graph_undirected(remainder_edges, remainder_vertices))
    
# def get_all_distances_from_vertex(graph, start_vertex):
#     vertex_path_dist_dict=set()
#     vertex_path_dist_dict[start_vertex] = 0
#     unexplored_vertices = list(graph.adjacent_vertices(start_vertex))
#     for vert in unexplored_vertices:
#         vertex_path_dist_dict[vert]=1
#     visited_vertices = [start_vertex]
    
#     while unexplored_vertices and (len(vertex_path_dist_dict.keys())<len(graph.vertices)):
#         curr_vertex = unexplored_vertices.pop()
#         curr_dist = vertex_path_dist_dict[curr_vertex]
#         visited_vertices.append(curr_vertex)
#         curr_neighbors = graph.adjacent_vertices(curr_vertex)
#         new_vertices = curr_neighbors - set(vertex_path_dist_dict.keys())
#         for vert in new_vertices:
#             vertex_path_dist_dict[vert]=curr_dist+1
#         unexplored_vertices = list(new_vertices) + unexplored_vertices
#     return vertex_path_dist_dict

# def separate_by_two_vertices(graph, vert_1, vert_2):
#     dict_1 = get_all_distances_from_vertex(graph, vert_1)
#     dict_2 = get_all_distances_from_vertex(graph, vert_2)
#     comp_2 = set([vert for vert in dict_2.keys() if dict_2[vert]<dict_1[vert]])
#     comp_1 = graph.vertices - comp_2
#     return comp_1, comp_2
