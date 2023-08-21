import logging

class TestDiffsnorm:
    def test_dense(self):
        logging.info('running TestDiffsnorm.test_dense...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype)
                    if not isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype) \
                            + 1j * np.random.normal(size=(m, n)).astype(dtype)

                    (U, s, Va) = svd(A, full_matrices=False)
                    snorm = diffsnorm(A, U, s, Va)
                    logging.info(snorm)
                    assert snorm < prec * s[0]

    def test_sparse(self):
        logging.info('running TestDiffsnorm.test_sparse...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype)
                    if not isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype) \
                            * (1 + 1j)

                    A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
                    A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    A = A / s[0]

                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    snorm = diffsnorm(A, U, s, Va)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])


def diffsnormc(A, U, s, Va, n_iter=20):
    """
    2-norm approx error to a matrix upon centering.

    Computes an estimate snorm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of C(A) - U diag(s) Va, using
    n_iter iterations of the power method started with a random vector,
    where C(A) refers to A from the input, after centering its columns;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of C(A) - U diag(s) Va, where C(A) refers to A
    after centering its columns.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    U : array_like
        second matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    s : array_like
        vector in the column-centered A - U diag(s) Va whose spectral
        norm is being estimated
    Va : array_like
        fourth matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of the column-centered A
        - U diag(s) Va (the estimate fails to be accurate with
        exponentially low probability as n_iter increases; see
        references DC1_, DC2_, and DC3_ below)

    Examples
    --------
    >>> from fbpca import diffsnormc, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, False)
    >>> err = diffsnormc(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to the
    column-centered A such that the columns of U are orthonormal, as
    are the rows of Va, and the entries of s are nonnegative and
    nonincreasing. diffsnormc(A, U, s, Va) outputs an estimate of the
    spectral norm of the column-centered A - U diag(s) Va, which
    should be close to the machine precision.

    References
    ----------
    .. [DC1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DC2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DC3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    diffsnorm, pca
    """

    (m, n) = A.shape
    (m2, k) = U.shape
    k2 = len(s)
    l = len(s)
    (l2, n2) = Va.shape

    assert m == m2
    assert k == k2
    assert l == l2
    assert n == n2

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and \
            np.isrealobj(Va):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    #
    # Calculate the average of the entries in every column.
    #
    c = A.sum(axis=0) / m
    c = c.reshape((1, n))

    if m >= n:

        #
        # Generate a random vector x.
        #
        if isreal:
            x = np.random.normal(size=(n, 1)).astype(dtype)
        else:
            x = np.random.normal(size=(n, 1)).astype(dtype) \
                + 1j * np.random.normal(size=(n, 1)).astype(dtype)

        x = x / norm(x)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            #
            y = mult(A, x) - np.ones((m, 1), dtype=dtype).dot(c.dot(x)) \
                - U.dot(np.diag(s).dot(Va.dot(x)))
            #
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T \
                - c.conj().T.dot(np.ones((1, m), dtype=dtype).dot(y)) \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

            #
            # Normalize x, memorizing its Euclidean norm.
            #
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:

        #
        # Generate a random vector y.
        #
        if isreal:
            y = np.random.normal(size=(m, 1)).astype(dtype)
        else:
            y = np.random.normal(size=(m, 1)).astype(dtype) \
                + 1j * np.random.normal(size=(m, 1)).astype(dtype)

        y = y / norm(y)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T \
                - c.conj().T.dot(np.ones((1, m), dtype=dtype).dot(y)) \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            #
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            #
            y = mult(A, x) - np.ones((m, 1), dtype=dtype).dot(c.dot(x)) \
                - U.dot(np.diag(s).dot(Va.dot(x)))

            #
            # Normalize y, memorizing its Euclidean norm.
            #
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm


class TestDiffsnormc(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestDiffsnormc.test_dense...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype)
                    if not isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype) \
                            + 1j * np.random.normal(size=(m, n)).astype(dtype)

                    c = A.sum(axis=0) / m
                    c = c.reshape((1, n))
                    Ac = A - np.ones((m, 1), dtype=dtype).dot(c)

                    (U, s, Va) = svd(Ac, full_matrices=False)
                    snorm = diffsnormc(A, U, s, Va)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])

    def test_sparse(self):

        logging.info('running TestDiffsnormc.test_sparse...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype)
                    if not isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype) \
                            * (1 + 1j)

                    A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
                    A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    A = A / s[0]

                    Ac = A.todense()
                    c = Ac.sum(axis=0) / m
                    c = c.reshape((1, n))
                    Ac = Ac - np.ones((m, 1), dtype=dtype).dot(c)

                    (U, s, Va) = svd(Ac, full_matrices=False)
                    snorm = diffsnormc(A, U, s, Va)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])


class TestDiffsnorms:
    def test_dense(self):
        logging.info('running TestDiffsnorms.test_dense...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for isreal in [True, False]:

                    if isreal:
                        A = np.random.normal(size=(n, n)).astype(dtype)
                    if not isreal:
                        A = np.random.normal(size=(n, n)).astype(dtype) \
                            + 1j * np.random.normal(size=(n, n)).astype(dtype)

                    (U, s, Va) = svd(A, full_matrices=True)
                    T = (np.diag(s).dot(Va)).dot(U)
                    snorm = diffsnorms(A, T, U)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])

    def test_sparse(self):
        logging.info('running TestDiffsnorms.test_sparse...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for isreal in [True, False]:

                    if isreal:
                        A = 2 * spdiags(
                            np.arange(n) + 1, 0, n, n).astype(dtype)
                    if not isreal:
                        A = 2 * spdiags(
                            np.arange(n) + 1, 0, n, n).astype(dtype) * (1 + 1j)

                    A = A - spdiags(np.arange(n + 1), 1, n, n)
                    A = A - spdiags(np.arange(n) + 1, -1, n, n)
                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    A = A / s[0]
                    A = A.tocoo()

                    (U, s, Va) = svd(A.todense(), full_matrices=True)
                    T = (np.diag(s).dot(Va)).dot(U)
                    snorm = diffsnorms(A, T, U)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])


class TestEigenn:
    def test_dense(self):
        logging.info('running TestEigenn.test_dense...')

        errs = []
        err = []

        def eigenntesterrs(n, k, n_iter, isreal, l, dtype):

            if isreal:
                V = np.random.normal(size=(n, k)).astype(dtype)
            if not isreal:
                V = np.random.normal(size=(n, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(n, k)).astype(dtype)

            (V, _) = qr(V, mode='economic')

            d0 = np.zeros((k), dtype=dtype)
            d0[0] = 1
            d0[1] = .1
            d0[2] = .01

            A = V.dot(np.diag(d0).dot(V.conj().T))
            A = (A + A.conj().T) / 2

            (d1, V1) = eigh(A)
            (d2, V2) = eigenn(A, k, n_iter, l)

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [10, 20]:
                for k in [3, 9]:
                    for n_iter in [0, 2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa) = eigenntesterrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            self.assertTrue(erra < prec)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

    def test_sparse(self):
        logging.info('running TestEigenn.test_sparse...')

        errs = []
        err = []
        bests = []

        def eigenntestserrs(n, k, n_iter, isreal, l, dtype):

            n2 = int(round(n / 2))
            assert 2 * n2 == n

            A = 2 * spdiags(np.arange(n2) + 1, 0, n, n).astype(dtype)

            if isreal:
                A = A - spdiags(np.arange(n2 + 1), 1, n, n).astype(dtype)
                A = A - spdiags(np.arange(n2) + 1, -1, n, n).astype(dtype)

            if not isreal:
                A = A - 1j * spdiags(np.arange(n2 + 1), 1, n, n).astype(dtype)
                A = A + 1j * spdiags(np.arange(n2) + 1, -1, n, n).astype(dtype)

            A = A / diffsnorms(
                A, np.zeros((2, 2), dtype=dtype),
                np.zeros((n, 2), dtype=dtype))

            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)

            A = A.tocoo()

            P = np.random.permutation(n)
            A = coo_matrix((A.data, (P[A.row], P[A.col])), shape=(n, n))
            A = A.tocsr()

            (d1, V1) = eigh(A.toarray())
            (d2, V2) = eigenn(A, k, n_iter, l)

            bestsa = sorted(abs(d1))[-k - 1]

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa, bestsa

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for k in [30, 90]:
                    for n_iter in [2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa, bestsa) = eigenntestserrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            bests.append(bestsa)
                            self.assertTrue(erra < max(10 * bestsa, prec))

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))


class TestEigens:
    def test_dense(self):
        logging.info('running TestEigens.test_dense...')

        errs = []
        err = []

        def eigenstesterrs(n, k, n_iter, isreal, l, dtype):

            if isreal:
                V = np.random.normal(size=(n, k)).astype(dtype)
            if not isreal:
                V = np.random.normal(size=(n, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(n, k)).astype(dtype)

            (V, _) = qr(V, mode='economic')

            d0 = np.zeros((k), dtype=dtype)
            d0[0] = 1
            d0[1] = -.1
            d0[2] = .01

            A = V.dot(np.diag(d0).dot(V.conj().T))
            A = (A + A.conj().T) / 2

            (d1, V1) = eigh(A)
            (d2, V2) = eigens(A, k, n_iter, l)

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa

        for dtype in ['float64', 'float32', 'float16']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [10, 20]:
                for k in [3, 9]:
                    for n_iter in [0, 2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa) = eigenstesterrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            self.assertTrue(erra < prec)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

    def test_sparse(self):
        logging.info('running TestEigens.test_sparse...')

        errs = []
        err = []
        bests = []

        def eigenstestserrs(n, k, n_iter, isreal, l, dtype):
            n2 = int(round(n / 2))
            assert 2 * n2 == n

            A = 2 * spdiags(np.arange(n2) + 1, 0, n2, n2).astype(dtype)

            if isreal:
                A = A - spdiags(np.arange(n2 + 1), 1, n2, n2).astype(dtype)
                A = A - spdiags(np.arange(n2) + 1, -1, n2, n2).astype(dtype)

            if not isreal:
                A = A - 1j * spdiags(
                    np.arange(n2 + 1), 1, n2, n2).astype(dtype)
                A = A + 1j * spdiags(
                    np.arange(n2) + 1, -1, n2, n2).astype(dtype)

            A = A / diffsnorms(
                A, np.zeros((2, 2), dtype=dtype),
                np.zeros((n2, 2), dtype=dtype))

            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)

            A = A.tocoo()

            datae = np.concatenate([A.data, A.data])
            rowe = np.concatenate([A.row + n2, A.row])
            cole = np.concatenate([A.col, A.col + n2])
            A = coo_matrix((datae, (rowe, cole)), shape=(n, n))

            P = np.random.permutation(n)
            A = coo_matrix((A.data, (P[A.row], P[A.col])), shape=(n, n))
            A = A.tocsc()

            (d1, V1) = eigh(A.toarray())
            (d2, V2) = eigens(A, k, n_iter, l)

            bestsa = sorted(abs(d1))[-k - 1]

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa, bestsa

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for k in [30, 90]:
                    for n_iter in [2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa, bestsa) = eigenstestserrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            bests.append(bestsa)
                            self.assertTrue(erra < max(10 * bestsa, prec))

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))


class TestPCA:
    def test_dense(self):
        logging.info('running TestPCA.test_dense...')

        errs = []
        err = []

        def pcatesterrs(m, n, k, n_iter, raw, isreal, l, dtype):
            if isreal:

                U = np.random.normal(size=(m, k)).astype(dtype)
                (U, _) = qr(U, mode='economic')

                V = np.random.normal(size=(n, k)).astype(dtype)
                (V, _) = qr(V, mode='economic')

            if not isreal:

                U = np.random.normal(size=(m, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(m, k)).astype(dtype)
                (U, _) = qr(U, mode='economic')

                V = np.random.normal(size=(n, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(n, k)).astype(dtype)
                (V, _) = qr(V, mode='economic')

            s0 = np.zeros((k), dtype=dtype)
            s0[0] = 1
            s0[1] = .1
            s0[2] = .01

            A = U.dot(np.diag(s0).dot(V.conj().T))

            if raw:
                Ac = A
            if not raw:
                c = A.sum(axis=0) / m
                c = c.reshape((1, n))
                Ac = A - np.ones((m, 1), dtype=dtype).dot(c)

            (U, s1, Va) = svd(Ac, full_matrices=False)
            (U, s2, Va) = pca(A, k, raw, n_iter, l)

            s3 = np.zeros((min(m, n)), dtype=dtype)
            for ii in range(k):
                s3[ii] = s2[ii]
            errsa = norm(s1 - s3)

            if raw:
                erra = diffsnorm(A, U, s2, Va)
            if not raw:
                erra = diffsnormc(A, U, s2, Va)

            return erra, errsa

        for dtype in ['float64', 'float32', 'float16']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(20, 10), (10, 20), (20, 20)]:
                for k in [3, 9]:
                    for n_iter in [0, 2, 1000]:
                        for raw in [True, False]:
                            for isreal in [True, False]:
                                l = k + 1
                                (erra, errsa) = pcatesterrs(
                                    m, n, k, n_iter, raw, isreal, l, dtype)
                                err.append(erra)
                                errs.append(errsa)
                                self.assertTrue(erra < prec)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

    def test_sparse(self):
        logging.info('running TestPCA.test_sparse...')

        errs = []
        err = []
        bests = []

        def pcatestserrs(m, n, k, n_iter, raw, isreal, l, dtype):

            if isreal:
                A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n)
            if not isreal:
                A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n)
                A = A.astype(dtype) * (1 + 1j)

            A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
            A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
            A = A / diffsnorm(
                A, np.zeros((m, 2), dtype=dtype), [0, 0],
                np.zeros((2, n), dtype=dtype))
            A = A.dot(A.conj().T.dot(A))
            A = A.dot(A.conj().T.dot(A))
            A = A[np.random.permutation(m), :]
            A = A[:, np.random.permutation(n)]

            if raw:
                Ac = A
            if not raw:
                c = A.sum(axis=0) / m
                c = c.reshape((1, n))
                Ac = A - np.ones((m, 1), dtype=dtype).dot(c)

            if raw:
                (U, s1, Va) = svd(Ac.toarray(), full_matrices=False)
            if not raw:
                (U, s1, Va) = svd(Ac, full_matrices=False)

            (U, s2, Va) = pca(A, k, raw, n_iter, l)

            bestsa = s1[k]

            s3 = np.zeros((min(m, n)), dtype=dtype)
            for ii in range(k):
                s3[ii] = s2[ii]
            errsa = norm(s1 - s3)

            if raw:
                erra = diffsnorm(A, U, s2, Va)
            if not raw:
                erra = diffsnormc(A, U, s2, Va)

            return erra, errsa, bestsa

        for dtype in ['float64', 'float32', 'float16']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for k in [30, 90]:
                    for n_iter in [2, 1000]:
                        for raw in [True, False]:
                            for isreal in [True, False]:
                                l = k + 1
                                (erra, errsa, bestsa) = pcatestserrs(
                                    m, n, k, n_iter, raw, isreal, l, dtype)
                                err.append(erra)
                                errs.append(errsa)
                                bests.append(bestsa)
                                self.assertTrue(erra < max(10 * bestsa, prec))

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))
