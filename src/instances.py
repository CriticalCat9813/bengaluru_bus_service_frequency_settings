import networkx as nx
import pandas as pd
import pickle
from src.config import *
import src.helper as helper


def main(n):

    g, stops_df, lines_df, trips_df = helper.preprocess_load()

    samples = []
    while len(samples) < n:
        sample = trips_df.sample()
        trips_df = trips_df.drop(sample.index)
        sample = sample.squeeze()
        if any(
                sample.o_node in line.dist.keys() and sample.d_node in line.dist.keys()
                for _, line in lines_df.iterrows()
        ):
            if int(sample.o_node) in AIRPORT_NODES or int(sample.d_node) in AIRPORT_NODES:
                continue
            if nx.shortest_path_length(g, sample.o_node, sample.d_node, 'length') <= 2 * LINES_DIST_TRGT:
                continue
            samples.append(sample)
    samples_df = pd.DataFrame(samples)
    samples_df.reset_index(inplace=True, drop=True)
    samples_df.to_csv('{0}/results/instances/samples_df_{1}_{2}.csv'.format(RELPATH, FILENAME, n), index=False)

    N = [i for i in range(samples_df.shape[0])]
    J = [j for j in range(lines_df.shape[0])]
    K = [k for k in range(1)]
    A = [lines_df['length'].to_list()]
    B = [[1] for _ in N]
    V = []
    for _, sample in samples_df.iterrows():
        V_i = []
        for _, line in lines_df.iterrows():
            if sample.o_node in line.dist.keys() and sample.d_node in line.dist.keys():
                o_dist = line.dist[sample.o_node]
                if o_dist <= LINES_DIST_TRGT:
                    o_term = 1
                elif o_dist >= LINES_DIST_CTFF:
                    o_term = 0
                else:
                    o_term = (LINES_DIST_CTFF - o_dist) / (LINES_DIST_CTFF - LINES_DIST_TRGT)
                d_dist = line.dist[sample.d_node]
                if d_dist <= LINES_DIST_TRGT:
                    d_term = 1
                elif d_dist >= LINES_DIST_CTFF:
                    d_term = 0
                else:
                    d_term = (LINES_DIST_CTFF - d_dist) / (LINES_DIST_CTFF - LINES_DIST_TRGT)
                V_ij = min(o_term, d_term)
            else:
                V_ij = 0
            V_i.append(V_ij)
        V.append(V_i)

    instance = N, J, K, A, B, V
    with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, n), 'wb') as file:
        pickle.dump(instance, file)


if __name__ == '__main__':
    main(n=1430)
