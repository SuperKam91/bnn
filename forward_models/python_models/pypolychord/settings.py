import os
import numpy


class PolyChordSettings:
    """
    PolyChord settings

    For full details of nested sampling and PolyChord, please refer to:

    Parameters
    ----------
    nDims: int
        Dimensionality of the model, i.e. the number of physical parameters.

    nDerived: int
        The number of derived parameters (can be 0).


    Keyword arguments
    -----------------
    nlive: int
        (Default: nDims*25)
        The number of live points.
        Increasing nlive increases the accuracy of posteriors and evidences,
        and proportionally increases runtime ~ O(nlive).

    num_repeats : int
        (Default: nDims*5)
        The number of slice slice-sampling steps to generate a new point.
        Increasing num_repeats increases the reliability of the algorithm.
        Typically
        * for reliable evidences need num_repeats ~ O(5*nDims).
        * for reliable posteriors need num_repeats ~ O(nDims)

    nprior : int
        (Default: nlive)
        The number of prior samples to draw before starting compression.

    do_clustering : boolean
        (Default: True)
        Whether or not to use clustering at run time.

    feedback : {0,1,2,3}
        (Default: 1)
        How much command line feedback to give

    precision_criterion : float
        (Default: 0.001)
        Termination criterion. Nested sampling terminates when the evidence
        contained in the live points is precision_criterion fraction of the
        total evidence.

    logzero : float
        (Default: -1e30)
        The loglikelihood value at which PolyChord considers points
        'unphysical', and excludes them at the prior level.

    max_ndead : int
        (Default: -1)
        Alternative termination criterion. Stop after max_ndead iterations.
        Set negative to ignore (default).

    boost_posterior : float
        (Default: 0.0)
        Increase the number of posterior samples produced.  This can be set
        arbitrarily high, but you won't be able to boost by more than
        num_repeats
        Warning: in high dimensions PolyChord produces _a lot_ of posterior
        samples. You probably don't need to change this

    posteriors : boolean
        (Default: True)
        Produce (weighted) posterior samples. Stored in <root>.txt.

    equals : boolean
        (Default: True)
        Produce (equally weighted) posterior samples. Stored in
        <root>_equal_weights.txt

    cluster_posteriors : boolean
        (Default: True)
        Produce posterior files for each cluster?
        Does nothing if do_clustering=False.

    write_resume : boolean
        (Default: True)
        Create a resume file.

    read_resume : boolean
        (Default: True)
        Read from resume file.

    write_stats : boolean
        (Default: True)
        Write an evidence statistics file.

    write_live : boolean
        (Default: True)
        Write a live points file.

    write_dead : boolean
        (Default: True)
        Write a dead points file.

    write_dead : boolean
        (Default: True)
        Write a prior points file.

    compression_factor : double
        (Default: exp(-1))
        How often to update the files and do clustering

    base_dir : string
        (Default: 'chains')
        Where to store output files.

    file_root : string
        (Default: 'test')
        Root name of the files produced.

    grade_frac : List[float]
        (Default: 1)
        The amount of time to spend in each speed.

    grade_dims : List[int]
        (Default: 1)
        The number of parameters within each speed.

    nlives : dict {double:int}
        (Default: {})
        Variable number of live points option. This dictionary is a mapping
        between loglike contours and nlive.
        You should still set nlive to be a sensible number, as this indicates
        how often to update the clustering, and to define the default value.
    seed : positive int
        (Default: system time in milliseconds)
        Choose the seed to seed the random number generator.
        Note **Positive seeds only**
        a negative seed indicates that you should use the system time in
        milliseconds

    """
    def __init__(self, nDims, nDerived, **kwargs):

        #some of default settings changed to match cpp values by kj
        #changed by kj
        self.nlive = kwargs.pop('nlive', 50)
        #self.nlive = kwargs.pop('nlive', nDims*25)
        self.num_repeats = kwargs.pop('num_repeats', nDims*5)
        self.nprior = kwargs.pop('nprior', -1)
        self.do_clustering = kwargs.pop('do_clustering', True)
        self.feedback = kwargs.pop('feedback', 1)
        self.precision_criterion = kwargs.pop('precision_criterion', 0.001)
        self.logzero = kwargs.pop('logzero', -1e30)
        self.max_ndead = kwargs.pop('max_ndead', -1)
        self.boost_posterior = kwargs.pop('boost_posterior', 0.0)
        self.posteriors = kwargs.pop('posteriors', True)
        #don't want equally weighted samples kj
        self.equals = kwargs.pop('equals', False)
        # self.equals = kwargs.pop('equals', True)
        #only bothered about posterior file of global evidence for now kj
        self.cluster_posteriors = kwargs.pop('cluster_posteriors', False)
        # self.cluster_posteriors = kwargs.pop('cluster_posteriors', True)
        #changed by kj
        # self.write_resume = kwargs.pop('write_resume', False)        
        self.write_resume = kwargs.pop('write_resume', True)
        #write .paramname file using method in pypolychord example
        self.write_paramnames = kwargs.pop('write_paramnames', False)
        #changed by kj
        # self.read_resume = kwargs.pop('read_resume', False)
        self.read_resume = kwargs.pop('read_resume', True)
        self.write_stats = kwargs.pop('write_stats', True)
        #don't generate extra output files for now, kj
        self.write_live = kwargs.pop('write_live', False)
        # self.write_live = kwargs.pop('write_live', True)
        self.write_dead = kwargs.pop('write_dead', False)
        # self.write_dead = kwargs.pop('write_dead', True)
        self.write_prior = kwargs.pop('write_prior', False)
        # self.write_prior = kwargs.pop('write_prior', True)
        self.compression_factor = kwargs.pop('compression_factor',
                                             numpy.exp(-1))
        self.base_dir = kwargs.pop('base_dir', 'chains')
        self.file_root = kwargs.pop('file_root', 'test')
        self.seed = kwargs.pop('seed', -1)
        self.grade_dims = list(kwargs.pop('grade_dims',
                                          [nDims]))
        self.grade_frac = list(kwargs.pop('grade_frac',
                                          [1.0]*len(self.grade_dims)))
        self.nlives = kwargs.pop('nlives', {})

        if kwargs:
            raise TypeError('Unexpected **kwargs in Contours constructor: %r'
                            % kwargs)

        if sum(self.grade_dims) != nDims:
            raise ValueError('grade_dims must sum to the total dimensionality:'
                             'sum(%i) /= %i' % (self.grade_dims, nDims))

    @property
    def cluster_dir(self):
        return os.path.join(self.base_dir, 'clusters')
