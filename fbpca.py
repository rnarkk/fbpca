"""
Functions for principal component analysis (PCA) and accuracy checks

---------------------------------------------------------------------

This module contains eight functions:

pca
    principal component analysis (singular value decomposition)
eigens
    eigendecomposition of a self-adjoint matrix
eigenn
    eigendecomposition of a nonnegative-definite self-adjoint matrix
diffsnorm
    spectral-norm accuracy of a singular value decomposition
diffsnormc
    spectral-norm accuracy of a centered singular value decomposition
diffsnorms
    spectral-norm accuracy of a Schur decomposition
mult
    default matrix multiplication
set_matrix_mult
    re-definition of the matrix multiplication function "mult"

---------------------------------------------------------------------

Copyright 2018 Facebook Inc.
All rights reserved.

"Software" means the fbpca software distributed by Facebook Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

* Neither the name Facebook nor the names of its contributors may be
  used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Additional grant of patent rights:

Facebook hereby grants you a perpetual, worldwide, royalty-free,
non-exclusive, irrevocable (subject to the termination provision
below) license under any rights in any patent claims owned by
Facebook, to make, have made, use, sell, offer to sell, import, and
otherwise transfer the Software. For avoidance of doubt, no license
is granted under Facebook's rights in any patent claims that are
infringed by (i) modifications to the Software made by you or a third
party, or (ii) the Software in combination with any software or other
technology provided by you or a third party.

The license granted hereunder will terminate, automatically and
without notice, for anyone that makes any claim (including by filing
any lawsuit, assertion, or other action) alleging (a) direct,
indirect, or contributory infringement or inducement to infringe any
patent: (i) by Facebook or any of its subsidiaries or affiliates,
whether or not such claim is related to the Software, (ii) by any
party if such claim arises in whole or in part from any software,
product or service of Facebook or any of its subsidiaries or
affiliates, whether or not such claim is related to the Software, or
(iii) by any party relating to the Software; or (b) that any right in
any patent claim of Facebook is invalid or unenforceable.
"""

import logging
import math
import unittest

import numpy as np
from scipy.linalg import cholesky, eigh, lu, qr, svd, norm, solve
from scipy.sparse import coo_matrix, issparse, spdiags


def diffsnorm(A, U, s, Va, n_iter=20):
    """
    2-norm accuracy of an approx to a matrix.

    Computes an estimate snorm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of A - U diag(s) Va, using
    n_iter iterations of the power method started with a random vector;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of A - U diag(s) Va.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    U : array_like
        second matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    s : array_like
        vector in A - U diag(s) Va whose spectral norm is being
        estimated
    Va : array_like
        fourth matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of A - U diag(s) Va (the
        estimate fails to be accurate with exponentially low prob. as
        n_iter increases; see references DM1_, DM2_, and DM3_ below)

    Examples
    --------
    >>> from fbpca import diffsnorm, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, True)
    >>> err = diffsnorm(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to A such
    that the columns of U are orthonormal, as are the rows of Va, and
    the entries of s are all nonnegative and are nonincreasing.
    diffsnorm(A, U, s, Va) outputs an estimate of the spectral norm of
    A - U diag(s) Va, which should be close to the machine precision.

    References
    ----------
    .. [DM1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DM2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DM3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    diffsnormc, pca
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

        # Run n_iter iterations of the power method.
        for it in range(n_iter):
            # Set y = (A - U diag(s) Va)x.
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))
            # Set x = (A' - Va' diag(s)' U')y.
            x = mult(y.conj().T, A).conj().T \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

            # Normalize x, memorizing its Euclidean norm.
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:
        # Generate a random vector y.
        if isreal:
            y = np.random.normal(size=(m, 1)).astype(dtype)
        else:
            y = np.random.normal(size=(m, 1)).astype(dtype) \
                + 1j * np.random.normal(size=(m, 1)).astype(dtype)

        y = y / norm(y)

        # Run n_iter iterations of the power method.
        for it in range(n_iter):
            # Set x = (A' - Va' diag(s)' U')y.
            x = mult(y.conj().T, A).conj().T \
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            # Set y = (A - U diag(s) Va)x.
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))

            # Normalise y, memorising its Euclidean norm.
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm
