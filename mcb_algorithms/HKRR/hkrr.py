from concurrent.futures import ProcessPoolExecutor as PPE
import numpy as np
from tqdm import tqdm, trange
import time


## MCB algorithm
class HKRRAlgorithm:
    """
    HKRR, Algorithm 1
    """

    def __init__(self, params):
        """
        Multicalibrate Predictions on Training Set
        """
        self.lmbda = params['lambda']
        self.alpha = params['alpha']
        self.v_hat_saved = []
        self.delta_iters = None
        self.subgroup_updated_iters = None
        self.v_updated_iters = None

    def fit(self, confs, labels, subgroups, use_oracle=True, randomized=True, max_iter=float('inf')):
        """
        confs: initial confs on positive class
        labels: labels for each data point
        subgroups: (ordered) list of lists where each entry is a list of all indices of data belonging to 
                    a certain subgroup
        max_iter: max # iterations before terminating
        """

        # init predictions
        p = confs.copy()
        n = len(confs)
        alpha = self.alpha
        lmbda = self.lmbda

        # count iterations
        iter = 0
        delta_iters = []
        subgroup_updated_iters = []
        v_updated_iters = []

        # get probability intervals and subgroups (including complements)
        V_range = np.arange(0, 1, lmbda)
        C = [(i, sg) for i, sg in enumerate(subgroups)]

        # shuffle subgroups if randomized
        if randomized:
            np.random.shuffle(C)
            np.random.shuffle(V_range)

        # repeat until no updates made
        updated = True
        while updated and iter < max_iter:
            updated = False
            iter += 1

            # track steps for test points
            delta = []
            subgroup_updated = []
            v_updated = []

            # for each S in C, for each v in Lambda[0,1] (S_v := subgroup intersect v)
            for S_idx, S in C:
                # skip empty subgroups
                if (len(S) == 0): continue
                
                for v in V_range:
                    S_v = [i for i in S if ((v < p[i] <= v + lmbda) or 
                                            (v == 0 and v <= p[i] <= v + lmbda))]

                    # if subset size smaller than tao, throw out
                    tao = alpha * lmbda * len(S)
                    if len(S_v) < tao:
                        continue

                    # retrieve offset from oracle
                    v_hat = np.mean(p[S_v]) # expected probability in S_v

                    if use_oracle:
                        r = self.oracle(subset=S_v, v_hat=v_hat, omega=(alpha/4), labels=labels)

                        # if no check, update predictions, projecting onto [0,1]
                        if r != 100:
                            p[S_v] = np.clip(p[S_v] + (r - v_hat), 0, 1)
                            updated = True

                            # update steps in procedure
                            delta.append(r - v_hat)
                            subgroup_updated.append(S_idx)
                            v_updated.append(v)
                    else:
                        dlta = np.mean(labels[S_v]) - v_hat
                        if (abs(dlta) < lmbda/10):
                            continue
                        p[S_v] = np.clip(p[S_v] + dlta, 0, 1)
                        updated = True

                        # update steps in procedure
                        delta.append(dlta)
                        subgroup_updated.append(S_idx)
                        v_updated.append(v)

            delta_iters.append(delta)
            subgroup_updated_iters.append(subgroup_updated)
            v_updated_iters.append(v_updated)

            # save v_hats for current iteration
            self.v_hat_saved.append({})
            for v in V_range:
                v_lmbda = [i for i in range(n) if ((v < p[i] <= v + lmbda) or 
                                                   (v == 0 and v <= p[i] <= v + lmbda))]

                # skip empty subgroups
                if (len(v_lmbda) == 0):
                    self.v_hat_saved[iter-1][v] = -1
                    continue

                v_hat = np.mean(p[v_lmbda])
                self.v_hat_saved[iter-1][v] = v_hat

        self.lmbda = lmbda
        self.delta_iters = delta_iters
        self.subgroup_updated_iters = subgroup_updated_iters
        self.v_updated_iters = v_updated_iters

        return p

    # oracle: Guess and check oracle to add noise
    def oracle(self, subset, v_hat, omega, labels):
        ps = np.mean(labels[subset])
        r=0
        
        # r == 100 indicates check
        if abs(ps-v_hat)<2*omega:
            r = 100
        if abs(ps-v_hat)>4*omega:
            r = np.random.uniform(0, 1)
        if r != 100:
            r = np.random.uniform(ps-omega, ps+omega)

        return r

    def predict(self, f_x, subgroups_containing_x, early_stop=None):
        """
        Adjust Test-Set Predictions with Deltas from Multicalibration Procedure
            for $x \in X$:
            > for $lvl$ in circuit:
            >> if $x \in \lambda(v) \cap subgroup(lvl)$:
            >>> apply update (delta)
            >>
            >>> project to $[0,1]$ if needed
            >
            return predictions

        :param f_x: initial prediction (float)
        """
        # name vars
        early_stop = early_stop if early_stop else len(self.subgroup_updated_iters)
        mcb_pred = f_x.copy()
        subgroup_updated_iters = self.subgroup_updated_iters
        v_updated_iters = self.v_updated_iters
        delta_iters = self.delta_iters
        lmbda = self.lmbda

        for subgroup_updated, v_updated, delta in zip(subgroup_updated_iters[:early_stop], v_updated_iters[:early_stop], delta_iters[:early_stop]):
            # for each lvl in circuit
            for lvl in range(len(subgroup_updated)):
                # check if datapoint belongs to $subgroup \cap lambda(v)$
                if subgroup_updated[lvl] in subgroups_containing_x:
                    v = v_updated[lvl]
                    if (v < mcb_pred <= v + lmbda) or (v == 0 and v <= mcb_pred <= v + lmbda):
                        # apply update, project onto [0, 1]
                        mcb_pred = np.clip(mcb_pred + delta[lvl], 0, 1)

        # get final prediciton from calib set v_hats
        V_range = np.arange(0, 1, lmbda)
        for v in V_range:
            if (v < mcb_pred <= v + lmbda) or (v == 0 and v <= mcb_pred <= v + lmbda):
                # if empty interval, return same prediction
                if self.v_hat_saved[-1][v] != -1:
                    mcb_pred = self.v_hat_saved[-1][v]
                break

        return mcb_pred
    
    def _batch_predict_regular(self, f_xs, groups, early_stop=None):
        """
        The batch_predict method without parallelization.
        Used in private _batch_predict method if max_workers=1.

        :param f_x: initial prediction (float)
        """
        # name vars
        early_stop = early_stop if early_stop else len(self.subgroup_updated_iters)
        mcb_preds = f_xs.copy()

        for i in trange(len(f_xs)):
            mcb_preds[i] = self.predict(f_xs[i], [j for j in range(len(groups)) if i in groups[j]], early_stop=early_stop)

        return mcb_preds

    def _idx_predict(self, idx_pair, f_xs, groups, early_stop=None):
        '''
        Same as _batch_predict_regular, but with custom indexing.
        '''
        # name vars
        idxs = list(range(*idx_pair))
        early_stop = early_stop if early_stop else len(self.subgroup_updated_iters)
        mcb_preds = f_xs.copy()

        for i in trange(len(f_xs)):
            mcb_preds[i] = self.predict(f_xs[i], [j for j in range(len(groups)) if idxs[i] in groups[j]], early_stop=early_stop)

        return mcb_preds

    def _batch_predict(self, f_xs, groups, early_stop=None, max_workers=None):
        '''
        Rewritten batch_predict method to parallelize prediction code.
        Called by public batch_predict method.
        '''
        print(f"Predicting with {max_workers} worker(s)")
        if max_workers == 1:
            return self._batch_predict_regular(f_xs, groups, early_stop)
        
        lmbda = (len(f_xs) // max_workers) + 1
        idx_pairs = [(i, min(i+lmbda, len(f_xs))) for i in range(0, len(f_xs), lmbda)]
        confs = [f_xs[i:j] for i, j in idx_pairs]

        with PPE(max_workers=max_workers) as executor:
            futures = [executor.submit(self._idx_predict, idx_pair, conf, groups, early_stop) 
                        for idx_pair, conf in zip(idx_pairs, confs)]
            results = list(tqdm(futures, total=len(futures)))
            results = [f.result() for f in results]
        
        return np.concatenate(results)
    
    def batch_predict(self, f_xs, groups, early_stop=None):
        '''
        Can choose number of workers here. Default is 1.
        '''
        nw = 1
        start = time.time()
        p = self._batch_predict(f_xs, groups, early_stop, max_workers=nw)
        print(f"Time for {nw} workers: {time.time()-start}")

        return p
