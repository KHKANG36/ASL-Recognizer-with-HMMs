import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        best_score = 100000000
        best_model = None
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                current_model = self.base_model(n_components)
                pval = (n_components ** 2) + 2 *len(self.X[0]) * n_components - 1
                logL = current_model.score(self.X, self.lengths)
                BIC_score = -2*logL + pval*(np.log(np.sum(self.lengths)))
                if BIC_score < best_score:
                    best_score = BIC_score 
                    best_model = current_model 
            except:
                pass
        return best_model 


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        
        best_score = -100000000
        best_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            other_score = 0
            M = 0 
            try:
                current_model = self.base_model(n_components)
                current_score = current_model.score(self.X, self.lengths) 
                for word in self.words:
                    if word != self.this_word:
                        other_X, other_length = self.hwords[word]
                        M += 1
                        other_score += current_model.score(other_X, other_length)
                DIC_score = current_score - (1/(M-1))*other_score
                if DIC_score > best_score:
                    best_score = DIC_score
                    best_model = current_model
            except:
                pass            
        return best_model 


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = -100000000
        best_num_state = 0
        best_model = None
        n_splits = min(3, len(self.sequences))
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            count = 0 
            total_score = 0
            try:
                splits = KFold(n_splits)
                current_model = self.base_model(n_components)
                for cv_train_idx, cv_test_idx in splits.split(self.sequences):
                    cv_train_X, cv_train_len = combine_sequences(cv_train_idx, self.sequences)
                    cv_test_X, cv_test_len =  combine_sequences(cv_test_idx, self.sequences)
                
                    current_model = current_model.fit(cv_train_X, cv_train_len)
                    score = current_model.score(cv_test_X, cv_test_len)
                    total_score += score
                    count += 1
                
                avg_score = total_score / count
            
                if avg_score > best_score: 
                    best_score = avg_score
                    best_model = current_model
            except:
                pass
        
        return best_model
