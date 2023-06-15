import networkx as nx
import numpy as np
import csv
import argparse
import glob
from multiprocessing import Pool
import CallGraphExtraction as CGE
import os
from functools import partial

def parse_options():
    parser = argparse.ArgumentParser(description='Obtaining the social network property values!')
    parser.add_argument('-d', '--dir', help='The path of a dir which contains some APK files.', required=True)
    parser.add_argument('-o', '--out', help='The path of output file.', required=True)
    args = parser.parse_args()

    return args

def Obtain_callGraph(apk, existing_files):
    return CGE.apk_to_callgraph(apk, existing_files)

def O_nodes(cg):
    return nx.number_of_nodes(cg)

def O_edges(cg):
    return nx.number_of_edges(cg)

def O_density(cg):
    return nx.density(cg)

# Assortativity
# Assortativity measures the similarity of connections
# in the graph with respect to the node degree.
# Return type float
def O_degree_assortativity_coefficient(cg):
    return nx.degree_assortativity_coefficient(cg)

# Bridges
# A bridge in a graph is an edge whose removal causes the number
# of connected components of the graph to increase. Equivalently,
# a bridge is an edge that does not belong to any cycle.
def O_bridge_number(cg):
    cg = nx.to_undirected(cg)

    if nx.has_bridges(cg):
        all_bridges = nx.bridges(cg)
        return len(list(all_bridges))
    else:
        return 0

# Centrality

def O_degree_centrality(cg):
    degree_cen_dict = nx.degree_centrality(cg)
    all_degree = [d for d in degree_cen_dict.values()]

    return (np.mean(all_degree), np.max(all_degree), np.min(all_degree))

def O_katz_centrality(cg):
    katz_cen_dict = nx.katz_centrality(cg)
    all_katz = [d for d in katz_cen_dict.values()]

    return (np.mean(all_katz), np.max(all_katz), np.min(all_katz))

def O_harmonic_centrality(cg):
    harmonic_cen_dict = nx.harmonic_centrality(cg)
    all_harmonic = [float(d)/float(nx.number_of_nodes(cg)-1) for d in harmonic_cen_dict.values()]

    return (np.mean(all_harmonic), np.max(all_harmonic), np.min(all_harmonic))

def O_closeness_centrality(cg):
    closeness_cen_dict = nx.closeness_centrality(cg)
    all_closeness = [d for d in closeness_cen_dict.values()]

    return (np.mean(all_closeness), np.max(all_closeness), np.min(all_closeness))

def O_betweenness_centrality(cg):
    betweenness_cen_dict = nx.betweenness_centrality(cg)
    all_betweenness = [d for d in betweenness_cen_dict.values()]

    return (np.mean(all_betweenness), np.max(all_betweenness), np.min(all_betweenness))

def O_load_centrality(cg):
    load_cen_dict = nx.load_centrality(cg)
    all_load = [d for d in load_cen_dict.values()]

    return (np.mean(all_load),np.max(all_load), np.min(all_load))

# Clique

def O_clique_number(cg):
    cg = nx.to_undirected(cg)
    all_cliques = nx.enumerate_all_cliques(cg)

    return len(list(all_cliques))

def O_maximal_clique_number(cg):
    cg = nx.to_undirected(cg)
    all_maximal_cliques = nx.find_cliques(cg)

    return len(list(all_maximal_cliques))

def O_largest_clique_size(cg):
    cg = nx.to_undirected(cg)
    return nx.graph_clique_number(cg)

# Clustering

def O_average_triangles_number(cg):
    cg = nx.to_undirected(cg)
    triangles_dict = nx.triangles(cg)
    all_triangles = [t for t in triangles_dict.values()]

    return np.mean(all_triangles)

def O_transitivity(cg):
    return nx.transitivity(cg)

def O_average_clustering_coefficient(cg):
    cg = nx.to_undirected(cg)
    return nx.average_clustering(cg)

# # Communicability --- Too Slow!!
#
# def O_communicability(cg):
#     cg = nx.to_undirected(cg)
#     communicability_dict = nx.communicability(cg)
#     all_communicability = [t for t in communicability_dict.values()]
#
#     return np.mean(all_communicability)

# Community
# from networkx.algorithms import community
#
# def O_label_propagation_community_number(cg):
#     cg = nx.to_undirected(cg)
#     communities = community.label_propagation_communities(cg)
#
#     return len(list(communities))
#
# def O_girvan_newman_community_number(cg):
#     communities = community.girvan_newman(cg)
#
#     return len(list(communities))

# Component

# def O_connected_components_number(cg):
#     cg = nx.to_undirected(cg)
#     return nx.number_connected_components(cg)

def O_strongly_connected_components_number(cg):
    return nx.number_strongly_connected_components(cg)

def O_weakly_connected_components_number(cg):
    return nx.number_weakly_connected_components(cg)

def O_attracting_components_number(cg):
    return nx.number_attracting_components(cg)

# Connectivity

def O_node_connectivity(cg):
    return nx.node_connectivity(cg)

def O_edge_connectivity(cg):
    return nx.edge_connectivity(cg)

# Cycles

def O_cycles_number(cg):
    cg = nx.to_undirected(cg)
    all_cycles = nx.cycle_basis(cg)
    return len(all_cycles)

def O_simple_cycles_number(cg):
    simple_cycles = nx.simple_cycles(cg)
    return len(list(simple_cycles))

# Distance Measures

# def O_center_number(cg):
#     cg = nx.to_undirected(cg)
#
#     if nx.is_connected(cg):
#         center_number = len(nx.center(cg))
#     else:
#         numbers = []
#         for component in (cg.subgraph(c) for c in nx.connected_components(cg)):
#             numbers.append(len(nx.center(component)))
#         center_number = np.mean(numbers)
#
#     return center_number

def O_diameter(cg):
    cg = nx.to_undirected(cg)

    if nx.is_connected(cg):
        diameter = nx.diameter(cg)
    else:
        numbers = []
        G = (cg.subgraph(c) for c in nx.connected_components(cg))
        for component in G:
            numbers.append(nx.diameter(component))
        diameter = np.mean(numbers)

    return diameter

def O_radius(cg):
    cg = nx.to_undirected(cg)

    if nx.is_connected(cg):
        radius = nx.radius(cg)
    else:
        numbers = []
        for component in (cg.subgraph(c) for c in nx.connected_components(cg)):
            numbers.append(nx.radius(component))
        radius = np.mean(numbers)

    return radius

def O_eccentricity_number(cg):
    cg = nx.to_undirected(cg)

    all_eccentricity = []
    if nx.is_connected(cg):
        eccentricity_dict = nx.eccentricity(cg)
        all_eccentricity = [e for e in eccentricity_dict.values()]
    else:
        for component in (cg.subgraph(c) for c in nx.connected_components(cg)):
            eccentricity_dict = nx.eccentricity(component)
            all_eccentricity.extend([e for e in eccentricity_dict.values()])

    return np.mean(all_eccentricity)

def O_periphery(cg):
    cg = nx.to_undirected(cg)

    if nx.is_connected(cg):
        periphery = len(nx.periphery(cg))
    else:
        numbers = []
        for component in (cg.subgraph(c) for c in nx.connected_components(cg)):
            numbers.append(len(nx.periphery(component)))
        periphery = np.mean(numbers)

    return periphery

# Efficiency --- Very very very very Slow!!!

# def O_average_local_efficiency(cg):
#     cg = nx.to_undirected(cg)
#     return nx.local_efficiency(cg)
#
# def O_average_global_efficiency(cg):
#     cg = nx.to_undirected(cg)
#     return nx.global_efficiency(cg)

# Isolates

def O_isolates_number(cg):
    return len(list(nx.isolates(cg)))

# Reciprocity

def O_reciprocity(cg):
    return nx.overall_reciprocity(cg)

# Rich Club --- ZeroDivisionError: float division by zero
#
# def O_rich_club_coefficient(cg):
#     cg = nx.to_undirected(cg)
#     rich_club_coefficient_dict = nx.rich_club_coefficient(cg)
#     all_rich_club_coefficient = [r for r in rich_club_coefficient_dict.values()]
#
#     return np.mean(all_rich_club_coefficient)

# Shortest Paths

def O_shortest_path(cg):
    sp = []

    nodes = list(nx.nodes(cg))
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            else:
                if nx.has_path(cg, i, j):
                    sp.append(nx.shortest_path(cg, i, j))
                else:
                    continue
    sp_length = [len(i)-1 for i in sp]

    return sp_length

def O_average_shortest_path_length(sp_length):
    return np.mean(sp_length)

def O_max_shortest_path_length(sp_length):
    return np.max(sp_length)

def O_min_shortest_path_length(sp_length):
    return np.min(sp_length)

def O_shortest_path_number(sp_length):
    return len(sp_length)

# Structural holes --- Too Slow!

# def O_constraint(cg):
#     constraint_dict = nx.constraint(cg)
#
#     constraints = []
#     for c in constraint_dict.values():
#         if str(c) == 'nan':
#             continue
#         else:
#             constraints.append(c)
#     # constraints = [c if str(c) != 'nan' else 0 for c in constraint_dict.values()]
#
#     return (np.mean(constraints), np.max(constraints), np.min(constraints))
#
# def O_effective_size(cg):
#     effective_size_dict = nx.effective_size(cg)
#
#     effective_sizes = []
#     for e in effective_size_dict.values():
#         if str(e) == 'nan':
#             continue
#         else:
#             effective_sizes.append(e)
#
#     return (np.mean(effective_sizes), np.max(effective_sizes), np.min(effective_sizes))

# Triads
# Return: the number of 16 possible types of triads
# Return type: list

def O_triads_number(cg):
    triads_dict = nx.triadic_census(cg)

    types = ['003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T',
             '030C', '201', '120D', '120U', '120C', '210', '300']
    triads = []
    for t in types:
        triads.append(triads_dict[t])

    return triads

# # Vitality
#
# def O_average_closeness_vitality(cg):
#     closeness_vitality_dict = dict(nx.closeness_vitality(cg))
#     closeness_vitalities = [cv for cv in closeness_vitality_dict.values()]
#
#     return closeness_vitalities

# Wiener index --- Too Slow!

# def O_wiener_index(cg):
#     cg = nx.to_undirected(cg)
#
#     if nx.is_connected(cg):
#         wiener_index = nx.wiener_index(cg)
#     else:
#         indexs = []
#         for component in (cg.subgraph(c) for c in nx.connected_components(cg)):
#             indexs.append(nx.wiener_index(component))
#         wiener_index = np.mean(indexs)
#
#     return wiener_index

# Algebraic Connectivity

from networkx.linalg import algebraicconnectivity
def O_algebraic_connectivity(cg):
    cg = nx.to_undirected(cg)
    return algebraicconnectivity.algebraic_connectivity(cg)

def Obtain_all_property_values(apk, existing_files):
    apk_name = apk.split('/')[-1].replace('.apk', '')
    cg = Obtain_callGraph(apk, existing_files)

    if cg == None:
        return None
    else:
        nodes_number = O_nodes(cg)
        # if nodes_number > 10000:
        #     return None
        # else:
        #print(apk_name)
        edges_number = O_edges(cg)
        density = O_density(cg)
        degree_assortativity_coefficient = O_degree_assortativity_coefficient(cg)
        bridge_number = O_bridge_number(cg)
        (ave_degree_centrality, max_degree_centrality, min_degree_centrality) = O_degree_centrality(cg)
        (ave_katz_centrality, max_katz_centrality, min_katz_centrality) = O_katz_centrality(cg)
        (ave_harmonic_centrality, max_harmonic_centrality, min_harmonic_centrality) = O_harmonic_centrality(cg)
        (ave_closeness_centrality, max_closeness_centrality, min_closeness_centrality) = O_closeness_centrality(cg)
        (ave_betweenness_centrality, max_betweenness_centrality, min_betweenness_centrality) = O_betweenness_centrality(cg)
        (ave_load_centrality, max_load_centrality, min_load_centrality) = O_load_centrality(cg)
        clique_number = O_clique_number(cg)
        maximal_clique_number = O_maximal_clique_number(cg)
        largest_clique_size = O_largest_clique_size(cg)
        average_triangles_number = O_average_triangles_number(cg)
        transitivity = O_transitivity(cg)
        average_clustering_coefficient = O_average_clustering_coefficient(cg)
        #connected_components_number = O_connected_components_number(cg)
        strongly_connected_components_number = O_strongly_connected_components_number(cg)
        weakly_connected_components_number = O_weakly_connected_components_number(cg)
        attracting_components_number = O_attracting_components_number(cg)
        node_connectivity = O_node_connectivity(cg)
        edge_connectivity = O_edge_connectivity(cg)
        cycles_number = O_cycles_number(cg)
        simple_cycles_number = O_simple_cycles_number(cg)
        #center_number = O_center_number(cg)
        diameter = O_diameter(cg)
        radius = O_radius(cg)
        eccentricity_number = O_eccentricity_number(cg)
        periphery = O_periphery(cg)
        #average_local_efficiency = O_average_local_efficiency(cg)
        #average_global_efficiency = O_average_global_efficiency(cg)
        isolates_number = O_isolates_number(cg)
        reciprocity = O_reciprocity(cg)
        # rich_club_coefficient = O_rich_club_coefficient(cg)
        # It is a list.
        sp_lengths = O_shortest_path(cg)
        average_shortest_path_length = O_average_shortest_path_length(sp_lengths)
        max_shortest_path_length = O_max_shortest_path_length(sp_lengths)
        min_shortest_path_length = O_min_shortest_path_length(sp_lengths)
        shortest_path_number = O_shortest_path_number(sp_lengths)
        #(ave_constraint, max_constraint, min_constraint) = O_constraint(cg)
        #(ave_effective_size, max_effective_size, min_effective_size) = O_effective_size(cg)
        # Return: the number of 16 possible types of triads
        # Return type: list
        triads_number = O_triads_number(cg)
        #wiener_index = O_wiener_index(cg)
        algebraic_connectivity = O_algebraic_connectivity(cg)

        all_info = [apk_name, nodes_number, edges_number, density, degree_assortativity_coefficient,
                    bridge_number, ave_degree_centrality, max_degree_centrality, min_degree_centrality,
                    ave_katz_centrality, max_katz_centrality, min_katz_centrality, ave_harmonic_centrality,
                    max_harmonic_centrality, min_harmonic_centrality, ave_closeness_centrality, max_closeness_centrality,
                    min_closeness_centrality, ave_betweenness_centrality, max_betweenness_centrality,
                    min_betweenness_centrality, ave_load_centrality, max_load_centrality, min_load_centrality,
                    clique_number, maximal_clique_number, largest_clique_size, average_triangles_number,
                    transitivity, average_clustering_coefficient,
                    strongly_connected_components_number, weakly_connected_components_number, attracting_components_number,
                    node_connectivity, edge_connectivity, cycles_number, simple_cycles_number,
                    diameter, radius, eccentricity_number, periphery,
                    isolates_number, reciprocity, average_shortest_path_length,
                    max_shortest_path_length, min_shortest_path_length, shortest_path_number]
        all_info.extend(list(triads_number))
        all_info.extend([algebraic_connectivity])
        print(apk_name)
        return all_info

def Record_info(apk, existing_files, out):
    apk_name = apk.split('/')[-1].replace('.apk', '')
    all_info = Obtain_all_property_values(apk, existing_files)
    if out[-1] == '/':
        out_txt = out + apk_name + '.txt'
    else:
        out_txt = out + '/' + apk_name + '.txt'

    if all_info == None:
        return None
    else:
        with open(out_txt, 'w') as f:
            f.write(str(all_info))

def main():
    args = parse_options()
    dir_path = args.dir
    out_path = args.out

    if dir_path[-1] == '/':
        apk_files = glob.glob(dir_path + '*.apk')
    else:
        apk_files = glob.glob(dir_path + '/*.apk')

    existing_files = os.listdir(out_path)
    pool = Pool(100)
    for apk in apk_files:
        Record_info(apk, existing_files, out_path)
    #pool.map(partial(Record_info, existing_files=existing_files, out=out_path), apk_files)

if __name__ == '__main__':
    main()