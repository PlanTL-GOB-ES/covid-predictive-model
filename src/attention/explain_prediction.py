import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from statistics import mean
import math
from collections import Counter


def explain_prediction(input1, input2, attention_weights, output_path, fname, draw_all_labels=False, input1_labels=None,
                       input2_labels=None, limit=150, big=False):
    G = nx.Graph()
    node_labels = {}
    for i in range(attention_weights.shape[0]):
        for j in range(attention_weights.shape[1]):
            iname = 'i'+str(i)
            jname = 'j' +str(j)
            if input1_labels and input2_labels:
                node_labels[iname] = input1_labels[i]
                node_labels[jname] = input2_labels[j]
            elif input1_labels:
                node_labels[iname] = input1_labels[i]
                node_labels[jname] = jname
            else:
                node_labels[iname] = iname
                node_labels[jname] = jname
            G.add_edge(iname, jname,
                       weight=float(0 if math.isnan(attention_weights[i][j]) else attention_weights[i][j]),
                       label=node_labels[iname])
    if len(G.edges()) == 0:
        return
    edge_weight_mean = mean([e[2]['weight'] for e in G.edges(data=True)])
    edges_draw = [edge for edge in G.edges(data=True) if edge[2]['weight'] > edge_weight_mean]
    sorted(edges_draw, key=lambda x: x[2]['weight'])
    edges_draw = edges_draw[:limit if limit < len(edges_draw) else len(edges_draw)]
    if not draw_all_labels:
        node_labels_selection = [edge[0] if edge[0][0] == 'i' else edge[1] for edge in edges_draw]
        node_labels = {node_label:node_labels[node_label] for node_label in node_labels_selection}
        #node_labels = dict((k, node_labels[k]) for k in node_labels_selection if k in node_labels)

    '''
    Gi = nx.Graph()
    top_nodes = set()
    for e in edges_draw:
        Gi.add_edge(e[0], e[1], **e[2])
        if e[0] in top:
            top_nodes.add(e[0])
        else:
            top_nodes.add(e[1])
    '''
    top = nx.bipartite.sets(G)[0]
    pos = nx.bipartite_layout(G, top)
    col1x, col2x = set([i[0] for i in list(pos.values())])
    if col1x > col2x:
        col1x, col2x = col2x, col1x
    col1ys = [i[1] for i in list(pos.values()) if i[0] == col1x]
    col2ys = [i[1] for i in list(pos.values()) if i[0] == col2x]
    sorted(col1ys)
    sorted(col2ys)

    pos = dict(zip(['i' + str(i) for i in range(attention_weights.shape[0])], zip([col1x]*len(col1ys), col1ys)))
    pos.update(dict(zip(['j' + str(j) for j in range(attention_weights.shape[1])], zip([col2x]*len(col2ys), col2ys))))
    if big == True:
        figsize = (40, 40)
    else:
        figsize = (12, 12)
    plt.plot(figure=figure(figsize=figsize))
    cmap = plt.cm.cool
    colors = [edge[2]['weight'] for edge in edges_draw]
    nx.draw(G, pos=pos, node_size=1, width=0.4, node_shape='s', edge_color=colors,
            labels=node_labels, edgelist=edges_draw, font_size=6, edge_cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(colors), vmax=max(colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_path, fname))
    plt.show()
    relevant_graph_features = Counter([e[2]['label'] for e in edges_draw])
    return dict(relevant_graph_features)


def show_relevant_graph_features(relevant_features, output_path, fname):
    labels = list()
    values = list()
    for key, value in zip(relevant_features.keys(), relevant_features.values()):
        labels.append(key)
        values.append(value)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.bar(range(len(values)), values)
    ax.set_ylabel('Count')
    ax.set_title('Feature relevance')
    ax.set_xticks(range(len(labels)))
    plt.xticks(rotation=90)
    ax.set_xticklabels(labels)
    plt.savefig(os.path.join(output_path, fname))
    plt.show()


def show_attention_matrix(attention_weights, data_columns, output_path, fname):
    fig, ax = plt.subplots(figsize=(50, 50))
    im = ax.imshow(attention_weights.numpy())
    ax.set_yticks(range(attention_weights.shape[0]))
    ax.set_xticks(range(attention_weights.shape[1]))
    ax.set_yticklabels(data_columns)
    ax.set_xticklabels([str(i) for i in list(range(attention_weights.shape[1]))])
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    fig.gca().set_aspect('auto')
    ax.set_title("Attention matrix")
    fig.tight_layout()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_path, fname))
    plt.show()
