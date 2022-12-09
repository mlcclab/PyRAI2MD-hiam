######################################################
#
# PyRAI2MD 2 module for grid search
#
# Author Jingbai Li
# Sep 25 2021
#
######################################################

import os
import time
import copy
import multiprocessing
import numpy as np

from PyRAI2MD.methods import QM
from PyRAI2MD.Machine_Learning.remote_train import RemoteTrain
from PyRAI2MD.Machine_Learning.training_data import Data
from PyRAI2MD.Machine_Learning.search_nn import SearchNN
from PyRAI2MD.Machine_Learning.search_GCNNP import SearchE2N2
from PyRAI2MD.Utils.timing import what_is_time
from PyRAI2MD.Utils.timing import how_long


class GridSearch:
    """ Grid search class

        Parameters:          Type:
            keywords         dict        keyword dictionary

        Attribute:           Type:
            layers           list        number of layers
            nodes            list        number of nodes
            batch            list        number of batch size
            l1               list        list of l1 regularization factor
            l2               list        list of l2 regularization factor
            dropout          list        list of dropout ratio
            use_hpc          int         use HPC (1) for calculation or not(0), like SLURM
            retrieve         int         retrieve training metrics


        Functions:           Returns:
            search           None        do grid-search
    """

    def __init__(self, keywords=None):
        self.keywords = copy.deepcopy(keywords)
        self.version = keywords['version']
        self.title = keywords['control']['title']
        self.ml = keywords['control']['qm'][0]
        self.ml_ncpu = keywords['control']['ml_ncpu']
        self.use_hpc = keywords['search']['use_hpc']
        self.retrieve = keywords['search']['retrieve']
        train_data = keywords[self.ml]['train_data']
        self.data = Data()
        self.data.load(train_data)
        self.data.stat()
        self.keywords[self.ml]['train_data'] = os.path.realpath(self.keywords[self.ml]['train_data'])
        if self.ml in ['nn', 'demo']:
            hspace = SearchNN(keywords=self.keywords)
        elif self.ml in ['e2n2']:
            hspace = SearchE2N2(keywords=self.keywords)
        else:
            hspace = None
            exit('\n KeywordError: grid search only supports nn, demo, or e2n2, found %s instead\n' % self.ml)
        self.hspace = hspace
        self.nsearch = hspace.nsearch()

    def _retrieve_data(self):
        ## retrieve training results in sequential or parallel mode
        variables_wrapper = [[n, x] for n, x in enumerate(self.hspace.queue())]

        ## adjust multiprocessing if necessary
        ncpu = 1
        if self.use_hpc > 0:
            ncpu = np.amin([self.nsearch, self.ml_ncpu])

        ## start multiprocessing
        results = [[] for _ in range(self.nsearch)]
        pool = multiprocessing.Pool(processes=ncpu)
        for val in pool.imap_unordered(self._search_wrapper_hpc, variables_wrapper):
            grid_id, grid_results = val
            results[grid_id] = grid_results
        pool.close()

        return results

    def _run_search_seq(self):
        ## run training in sequential mode
        variables_wrapper = [[n, x] for n, x in enumerate(self.hspace.queue())]

        ## sequential mode
        ncpu = 1

        ## start multiprocessing
        results = [[] for _ in range(self.nsearch)]
        pool = multiprocessing.Pool(processes=ncpu)
        for val in pool.imap_unordered(self._search_wrapper_seq, variables_wrapper):
            grid_id, grid_results = val
            results[grid_id] = grid_results
        pool.close()

        return results

    def _search_wrapper_seq(self, variables):
        grid_id, hypers = variables

        ## update hypers and add training data
        keywords, key = self.hspace.update_hypers(hypers)
        keywords[self.ml]['train_mode'] = 'training'
        keywords[self.ml]['data'] = self.data
        maindir = os.getcwd()
        calcdir = '%s/grid-search/NN-%s-%s' % (os.getcwd(), self.title, key)

        ## train on local machine
        if not os.path.exists(calcdir):
            os.makedirs(calcdir)

        os.chdir(calcdir)
        model = QM([self.ml], keywords=keywords)
        metrics = model.train()
        os.chdir(maindir)

        return grid_id, metrics

    def _run_search_hpc(self):
        ## wrap variables for multiprocessing
        variables_wrapper = [[n, x] for n, x in enumerate(self.hspace.queue())]

        ## adjust multiprocessing if necessary
        ncpu = np.amin([self.nsearch, self.ml_ncpu])

        ## start multiprocessing
        results = [[] for _ in range(self.nsearch)]
        pool = multiprocessing.Pool(processes=ncpu)
        for val in pool.imap_unordered(self._search_wrapper_hpc, variables_wrapper):
            grid_id, grid_results = val
            results[grid_id] = grid_results
        pool.close()

        return results

    def _search_wrapper_hpc(self, variables):
        grid_id, hypers = variables

        ## update hypers
        keywords, key = self.hspace.update_hypers(hypers)
        keywords[self.ml]['train_mode'] = 'training'
        calcdir = '%s/grid-search/NN-%s-%s' % (os.getcwd(), self.title, key)

        ## remote training in subprocess
        model = RemoteTrain(keywords=keywords, calcdir=calcdir, use_hpc=self.use_hpc, retrieve=self.retrieve)
        metrics = model.train()

        return grid_id, metrics

    def _write_summary(self, metrics):
        logpath = os.getcwd()
        summary, crashed = self.hspace.summarize(metrics)

        with open('%s/%s.log' % (logpath, self.title), 'a') as log:
            log.write(summary)

        if len(crashed) > 0:
            with open('%s/%s.crashed' % (logpath, self.title), 'w') as log:
                log.write(crashed)

        return self

    def _heading(self):

        headline = """
%s
 *---------------------------------------------------*
 |                                                   |
 |                   Grid Search                     |
 |                                                   |
 *---------------------------------------------------*

 Number of search: %s

""" % (self.version, self.nsearch)

        return headline

    def search(self):
        logpath = os.getcwd()
        start = time.time()
        heading = 'Grid Search Start: %20s\n%s' % (what_is_time(), self._heading())

        with open('%s/%s.log' % (logpath, self.title), 'w') as log:
            log.write(heading)

        if self.use_hpc > 0:
            search_func = self._run_search_hpc
        else:
            search_func = self._run_search_seq

        if self.retrieve == 0:
            results = search_func()
        else:
            results = self._retrieve_data()

        self._write_summary(results)
        end = time.time()
        walltime = how_long(start, end)
        tailing = 'Grid Search End: %20s Total: %20s\n' % (what_is_time(), walltime)

        with open('%s/%s.log' % (logpath, self.title), 'a') as log:
            log.write(tailing)

        return self
