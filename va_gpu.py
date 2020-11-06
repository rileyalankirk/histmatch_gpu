# GPU implementations of VA functions
import cupy


def calc_info_gpu(im, niters=5, beta=None, alpha=None, gamma=None): # see original documentation for info about parameters
    from hist.exact.va import __check_args
    gamma, _, beta, alpha_1, alpha_2 = __check_args(gamma, beta, alpha, im.ndim)
    if niters <= 0: raise ValueError('niters') # niters is R in [3]
    im = im.astype(float)
    gamma = cupy.asarray(gamma)
    constants = cupy.array((beta, alpha_1, alpha_2), dtype=float)
    return __get_va_kernel(im.ndim, niters)(im, gamma, constants, im)

@cupy.util.memoize(for_each_device=True)
def __get_va_kernel(ndim, niters=5):
    """
    Create the cupy ElementwiseKernel for the VA algorithm with the given number of dimensions and
    iterations. The kernel requires passing the image, the gamma values (with 3**ndim values), and
    the constants as a 3-element array of beta, alpha_1, and alpha_2. Additionally, the image must
    be passed again as the output array.
   
    This could easily remove the niters parameters and only be dependent on the number of dimensions.
    """
   
    # This creates C code similar to the purpose of the Python __get_g_info() function.
    # The generated C code is a series of ndim nested loops and fills in the variables gamma, neighbors, and n_neighbors
    g_info = []
    for j in range(ndim):
        g_info.append('''for (int dim_{j} = 0; dim_{j} < 3; dim_{j}++) {{
    ni[{j}] = _ind.get()[{j}] + dim_{j} - 1;
    bool skip_{j} = ni[{j}] < 0 || ni[{j}] >= x.shape()[{j}];'''.format(j=j))
    g_info = '''int neighbor_full_ind = 0, ni[{ndim}];
    {loops}
        // inner-most loop
        if (gamma_[neighbor_full_ind] && !({skip})) {{
            gamma[n_neighbors] = gamma_[neighbor_full_ind];
            neighbors[n_neighbors++] = &x[ni];
        }}
        neighbor_full_ind++;
    {end_loops}'''.format(
        ndim=ndim,
        loops='\n'.join(g_info),
        skip='||'.join('skip_{}'.format(j) for j in range(ndim)),
        end_loops='}'*ndim)
   
    # pre-defined variables:
    #   CArray<double, {ndim}> x [see https://github.com/cupy/cupy/blob/89cb2dfc2ccb9b3b429623c88c7ccd5d9a720278/cupy/core/include/cupy/carray.cuh#L176]
    #   CIndexer<{ndim}> _ind    [see https://github.com/cupy/cupy/blob/89cb2dfc2ccb9b3b429623c88c7ccd5d9a720278/cupy/core/include/cupy/carray.cuh#L294]
    #   ptrdiff_t* i = _ind.get()
    return cupy.ElementwiseKernel('raw float64 x, raw float64 gamma_, raw float64 constants', 'float64 out',
        """
        // The gamma_ array is condensed to gamma, removing all 0s and out-of-bounds elements
        // The neighbors array is pointers to the data for neighbors, parallel to gamma
        int n_neighbors = 0; // actual number of neighbors being considered, <={max_neighbors}
        double gamma[{max_neighbors}];
        const double* neighbors[{max_neighbors}];
        {g_info}

        // Save constants to variables
        const double beta = constants[0];
        const double alpha_1 = constants[1];
        const double alpha_2 = constants[2];
        const double orig = x[i];

        // The actual algorithm is from this point forward
        for (int k = 0; k < {niters}; k++) {{
            double t = 0, value = x[i];
            for (int j = 0; j < n_neighbors; j++) {{
                double a = gamma[j] * (value - *neighbors[j]);
                t += a / (fabs(a) + alpha_2);
            }}
            t *= beta;
            double u = alpha_1 * t / (fabs(t) - 1);
            u += orig;
           
            __syncthreads();
            out = u;
            __syncthreads();
            // TODO: Maybe faster than just doing more iterations? (on GPU)
            // Should be only done once (not on all threads, actually should re-dispatch)
            // if (k >= 1 && is_unique(x)) {{ break; }}
        }}
        """.format(niters=niters, ndim=ndim, max_neighbors=3**ndim, g_info=g_info),
                                  'va_calc_info_{}_{}'.format(niters, ndim), reduce_dims=False)