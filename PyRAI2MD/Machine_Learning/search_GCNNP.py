######################################################
#
# PyRAI2MD 2 module for grid search nn/demo hyper space
#
# Author Jingbai Li
# Dec 8 2022
#
######################################################

import copy

class SearchE2N2:
    """ E2N2 grid search class

        Parameters:          Type:
            keywords         dict        keyword dictionary

        Attribute:           Type:
            n_features       list        list of node feature numbers
            n_blocks         list        list of interaction block numbers
            l_max            list        list of rotation order
            n_rbf            list        list of rbf basis numbers
            rbf_layers       list        list of rbf layer numbers
            rbf_neurons      list        list of rbf neuron numbers

        Functions:           Returns:
            nsearch          int         number of searches
            queue            list        a list of hyper combinations
            update_hypers    tuple       a dict of update hypers and a key string
            summarize        tuple       a string of normal completed output and the path to the crashed jobs
    """

    def __init__(self, keywords=None):
        grids = keywords['search']
        gr_features = grids['n_features']
        gr_blocks = grids['n_blocks']
        gr_lmax = grids['l_max']
        gr_nrbf = grids['n_rbf']
        gr_rbf_layers = grids['rbf_layers']
        gr_rbf_neurons = grids['rbf_neurons']

        self.queue = []
        for a in gr_features:
            for b in gr_blocks:
                for c in gr_lmax:
                    for d in gr_nrbf:
                        for e in gr_rbf_layers:
                            for f in gr_rbf_neurons:
                                self.queue.append([a, b, c, d, e, f])
        self.keywords = keywords

    def nsearch(self):
        return len(self.queue)

    def queue(self):
        return self.queue

    def update_hypers(self, hypers):
        variables = copy.deepcopy(self.keywords)
        n_features, n_blocks, l_max, n_rbf, rbf_layers, rbf_neurons = hypers
        key = 'f%s_b%s_l%s_n%s_%s_%s' % (n_features, n_blocks, l_max, n_rbf, rbf_layers, rbf_neurons)
        variables['e2n2']['e2n2_eg']['n_features'] = n_features
        variables['e2n2']['e2n2_eg']['n_blocks'] = n_blocks
        variables['e2n2']['e2n2_eg']['n_features'] = l_max
        variables['e2n2']['e2n2_eg']['n_blocks'] = n_rbf
        variables['e2n2']['e2n2_eg']['n_features'] = rbf_layers
        variables['e2n2']['e2n2_eg']['n_blocks'] = rbf_neurons
        variables['e2n2']['e2n2_nac']['n_features'] = n_features
        variables['e2n2']['e2n2_nac']['n_blocks'] = n_blocks
        variables['e2n2']['e2n2_nac']['n_features'] = l_max
        variables['e2n2']['e2n2_nac']['n_blocks'] = n_rbf
        variables['e2n2']['e2n2_nac']['n_features'] = rbf_layers
        variables['e2n2']['e2n2_nac']['n_blocks'] = rbf_neurons
        variables['e2n2']['e2n2_soc']['n_features'] = n_features
        variables['e2n2']['e2n2_soc']['n_blocks'] = n_blocks
        variables['e2n2']['e2n2_soc']['n_features'] = l_max
        variables['e2n2']['e2n2_soc']['n_blocks'] = n_rbf
        variables['e2n2']['e2n2_soc']['n_features'] = rbf_layers
        variables['e2n2']['e2n2_soc']['n_blocks'] = rbf_neurons

        return variables, key

    def summarize(self, metrics):
        summary = '  Feats   Blocks   l_max    RBFs    Rlayers     Rnodes    Energy1    Gradient1    NAC1        SOC1        Energy2    Gradient2    NAC2        SOC2        Time     Walltime\n'
        crashed = ''
        for n, hypers in enumerate(self.queue):

            if metrics[n]['status'] == 0:
                crashed += '%s\n' % (metrics[n]['path'])
                continue

            n_features, n_blocks, l_max, n_rbf, rbf_layers, rbf_neurons = hypers
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
                n_features,
                n_blocks,
                l_max,
                n_rbf,
                rbf_layers,
                rbf_neurons,
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
