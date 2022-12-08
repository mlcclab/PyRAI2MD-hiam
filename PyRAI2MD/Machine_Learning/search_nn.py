######################################################
#
# PyRAI2MD 2 module for grid search nn/demo hyper space
#
# Author Jingbai Li
# Dec 8 2022
#
######################################################

import copy

class SearchNN:
    """ NN grid search class

        Parameters:          Type:
            keywords         dict        keyword dictionary

        Attribute:           Type:
            layers           list        number of layers
            nodes            list        number of nodes
            batch            list        number of batch size
            l1               list        list of l1 regularization factor
            l2               list        list of l2 regularization factor
            dropout          list        list of dropout ratio

        Functions:           Returns:
            nsearch          int         number of searches
            queue            list        a list of hyper combinations
            update_hypers    tuple       a dict of update hypers and a key string
            summarize        tuple       a string of normal completed output and the path to the crashed jobs
    """

    def __init__(self, keywords=None):
        grids = keywords['search']
        gr_layers = grids['depth']
        gr_nodes = grids['nn_size']
        gr_batch = grids['batch_size']
        gr_l1 = grids['reg_l1']
        gr_l2 = grids['reg_l2']
        gr_dropout = grids['dropout']

        self.queue = []
        for a in gr_layers:
            for b in gr_nodes:
                for c in gr_batch:
                    for d in gr_l1:
                        for e in gr_l2:
                            for f in gr_dropout:
                                self.queue.append([a, b, c, d, e, f])
        self.keywords = keywords

    def nsearch(self):
        return len(self.queue)

    def queue(self):
        return self.queue

    def update_hypers(self, hypers):
        variables = copy.deepcopy(self.keywords)
        layers, nodes, batch, l1, l2, dropout = hypers
        key = '%s_%s_%s_%s_%s_%s' % (layers, nodes, batch, l1, l2, dropout)
        variables['nn']['eg']['depth'] = layers
        variables['nn']['eg']['nn_size'] = nodes
        variables['nn']['eg']['batch_size'] = batch
        variables['nn']['eg']['reg_l1'] = l1
        variables['nn']['eg']['reg_l2'] = l2
        variables['nn']['eg']['dropout'] = dropout
        variables['nn']['nac']['depth'] = layers
        variables['nn']['nac']['nn_size'] = nodes
        variables['nn']['nac']['batch_size'] = batch
        variables['nn']['nac']['reg_l1'] = l1
        variables['nn']['nac']['reg_l2'] = l2
        variables['nn']['nac']['dropout'] = dropout
        variables['nn']['soc']['depth'] = layers
        variables['nn']['soc']['nn_size'] = nodes
        variables['nn']['soc']['batch_size'] = batch
        variables['nn']['soc']['reg_l1'] = l1
        variables['nn']['soc']['reg_l2'] = l2
        variables['nn']['soc']['dropout'] = dropout
        variables['nn']['eg2']['depth'] = layers
        variables['nn']['eg2']['nn_size'] = nodes
        variables['nn']['eg2']['batch_size'] = batch
        variables['nn']['eg2']['reg_l1'] = l1
        variables['nn']['eg2']['reg_l2'] = l2
        variables['nn']['eg2']['dropout'] = dropout
        variables['nn']['nac2']['depth'] = layers
        variables['nn']['nac2']['nn_size'] = nodes
        variables['nn']['nac2']['batch_size'] = batch
        variables['nn']['nac2']['reg_l1'] = l1
        variables['nn']['nac2']['reg_l2'] = l2
        variables['nn']['nac2']['dropout'] = dropout
        variables['nn']['soc2']['depth'] = layers
        variables['nn']['soc2']['nn_size'] = nodes
        variables['nn']['soc2']['batch_size'] = batch
        variables['nn']['soc2']['reg_l1'] = l1
        variables['nn']['soc2']['reg_l2'] = l2
        variables['nn']['soc2']['dropout'] = dropout
        variables['demo'] = variables['nn']

        return variables, key

    def summarize(self, metrics):
        summary = '  Layers   Nodes   Batch    L1        L2       Dropout    Energy1    Gradient1    NAC1        SOC1        Energy2    Gradient2    NAC2        SOC2        Time     Walltime\n'
        crashed = ''
        for n, hypers in enumerate(self.queue):

            if metrics[n]['status'] == 0:
                crashed += '%s\n' % (metrics[n]['path'])
                continue

            layers, nodes, batch, l1, l2, dropout = hypers
            e1 = metrics[n]['e1']
            g1 = metrics[n]['g1']
            n1 = metrics[n]['n1']
            s1 = metrics[n]['s1']
            e2 = metrics[n]['e2']
            g2 = metrics[n]['g2']
            n2 = metrics[n]['n2']
            s2 = metrics[n]['s2']
            t = metrics[n]['time']
            wt = metrics[n]['walltime']
            summary += '%8d%8d%8d%10.2e%10.2e%10.2e%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%10.2e %s\n' % (
                layers,
                nodes,
                batch,
                l1,
                l2,
                dropout,
                e1,
                g1,
                n1,
                s1,
                e2,
                g2,
                n2,
                s2,
                t,
                wt)
        return summary, crashed
