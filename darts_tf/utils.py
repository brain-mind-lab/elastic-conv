import numpy as np
from operations import OPS, ECONV_WEIGHT_TO_SIZE

import pydot

step_node_style_kwargs = dict(
    style='filled',
    fillcolor='#AAAAFF',
    shape='rect'
)
in_node_style_kwargs = dict(
    style='filled',
    fillcolor='#AAFFAA',
    shape='rect'
)
out_node_style_kwargs = dict(
    style='filled',
    fillcolor='#FFFFAA',
    shape='rect'
)

def plot_cells(alpha_normal, alpha_normal_econv, alpha_reduction, alpha_reduction_econv, dest_file):

    def _cell_to_pydot_cluster(alpha, alpha_econv, name, debug=False):
        G = pydot.Cluster(name, label=name)

        c_k2 = pydot.Node('%s_C_[k-2]' % name, label='C_[k-2]', **in_node_style_kwargs)
        c_k1 = pydot.Node('%s_C_[k-1]' % name, label='C_[k-1]', **in_node_style_kwargs)
        c_k = pydot.Node('%s_C_[k]' % name, label='C_[k]', **out_node_style_kwargs)
        steps = [c_k2, c_k1]
        for i in range(alpha.shape[0] - 2):
            step = pydot.Node('%s_%i' % (name, i), label='%i' % i, **step_node_style_kwargs)
            steps.append(step)
            G.add_edge(pydot.Edge(step, c_k))


        for n in steps:
            G.add_node(n)
        G.add_node(c_k)

        for i in range(2, alpha.shape[0]):
            for j in range(i):
                if debug:
                    softmax_weights = np.exp(alpha[i, j]) / np.exp(alpha[i, j]).sum()
                    
                    for op_name, op_weight in zip(OPS.keys(), softmax_weights):
                        if op_name == 'none':
                            continue
                        if op_name == 'skip_connect':
                            op_name = ''
                            
                        op_name = '%s w=%.2f' % (op_name, op_weight)

                        edge = pydot.Edge(steps[j], steps[i], label=op_name, color='black', penwidth=float(1 * op_weight))
                        G.add_edge(edge)
                else:
                    op_name = list(OPS.keys())[alpha[i, j].argmax()]
                    if op_name == 'none':
                        continue
                    if op_name == 'skip_connect':
                        op_name = ''
                    if op_name == 'elastic_conv':
                        w = alpha_econv[i, j]
                        filter_size = ECONV_WEIGHT_TO_SIZE[w]
                        op_name = 'conv_%dx%d' % filter_size
                    edge = pydot.Edge(steps[j], steps[i], label=op_name, color='black', penwidth=1)
                    G.add_edge(edge)
                    

        return G
    
    G = pydot.Dot(graph_type="digraph")
    G_n = pydot.Dot(graph_type="digraph")
    G_r = pydot.Dot(graph_type="digraph")
    G_debug = pydot.Dot(graph_type="digraph", rankdir="LR")
    
    normal = _cell_to_pydot_cluster(alpha_normal, alpha_normal_econv, 'Normal')
    reduction = _cell_to_pydot_cluster(alpha_reduction, alpha_reduction_econv, 'Reduction')
    normal_debug = _cell_to_pydot_cluster(alpha_normal, alpha_normal_econv, 'Normal', debug=True)
    reduction_debug = _cell_to_pydot_cluster(alpha_reduction, alpha_reduction_econv, 'Reduction', debug=True)
    
    G_n.add_subgraph(normal)
    G_r.add_subgraph(reduction)
    
    G.add_subgraph(normal)
    G.add_subgraph(reduction)
    
    G_debug.add_subgraph(normal_debug)
    G_debug.add_subgraph(reduction_debug)
    
    G_n.write_svg(dest_file + '_n.svg')
    G_r.write_svg(dest_file + '_r.svg')
    G.write_svg(dest_file + '.svg')
    G_debug.write_svg(dest_file + '_debug.svg')

import os
def _fprint(filename):
    def fprint(message):
        print(message)
        with open(filename, 'a') as f:
            f.write(str(message) + os.linesep)
    return fprint