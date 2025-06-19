# matrix.py
# This file contains the implementation of a matrix class for basic linear algebra calculations.
# Some imported from Naesungmath (https://github.com/minseo0388/naesungmath)

import numpy as np

class Matrix:
    def vector(self, v):
        """
        Convert a list or numpy array to a numpy array.
        
        Parameters:
            v: List or numpy array.
        
        Returns:
            Numpy array.
        """
        return np.array(v, dtype=float)
    
    def add(self, A, B):
        """
        Add two matrices or vectors.
        
        Parameters:
            A: First matrix or vector.
            B: Second matrix or vector.
        
        Returns:
            Sum of A and B as a numpy array.
        """
        return np.add(self.vector(A), self.vector(B))
    
    def subtract(self, A, B):
        """
        Subtract two matrices or vectors.
        
        Parameters:
            A: First matrix or vector.
            B: Second matrix or vector.
        
        Returns:
            Difference of A and B as a numpy array.
        """
        return np.subtract(self.vector(A), self.vector(B))
    
    def multiply(self, A, B):
        """
        Multiply two matrices or vectors.
        
        Parameters:
            A: First matrix or vector.
            B: Second matrix or vector.
        
        Returns:
            Product of A and B as a numpy array.
        """
        return np.multiply(self.vector(A), self.vector(B))
    
    def divide(self, A, B):
        """
        Divide two matrices or vectors.
        
        Parameters:
            A: First matrix or vector.
            B: Second matrix or vector.
        
        Returns:
            Quotient of A and B as a numpy array.
        """
        return np.divide(self.vector(A), self.vector(B))
    
    def dot(self, A, B):
        """
        Compute the dot product of two vectors or matrices.
        
        Parameters:
            A: First vector or matrix.
            B: Second vector or matrix.
        
        Returns:
            Dot product of A and B as a numpy array.
        """
        return np.dot(self.vector(A), self.vector(B))
    
    def transpose(self, A):
        """
        Transpose a matrix.
        
        Parameters:
            A: Matrix to be transposed.
        
        Returns:
            Transposed matrix as a numpy array.
        """
        return np.transpose(self.vector(A))
    
    def inverse(self, A):
        """
        Compute the inverse of a matrix.
        
        Parameters:
            A: Matrix to be inverted.
        
        Returns:
            Inverse of A as a numpy array.
        
        Raises:
            ValueError: If A is not square or singular.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.linalg.inv(A)
    
    def determinant(self, A):
        """
        Compute the determinant of a matrix.
        
        Parameters:
            A: Matrix for which to compute the determinant.
        
        Returns:
            Determinant of A as a float.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.linalg.det(A)
    
    def solve(self, A, b):
        """
        Solve the linear equation Ax = b.
        
        Parameters:
            A: Coefficient matrix.
            b: Right-hand side vector.
        
        Returns:
            Solution vector x as a numpy array.
        
        Raises:
            ValueError: If A is not square or if dimensions do not match.
        """
        A = self.vector(A)
        b = self.vector(b)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimensions of A and b do not match.")
        return np.linalg.solve(A, b)
    
    def norm(self, A):
        """
        Compute the Frobenius norm of a matrix.
        
        Parameters:
            A: Matrix for which to compute the norm.
        
        Returns:
            Frobenius norm of A as a float.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.linalg.norm(A, 'fro')
    
    def identity(self, n):
        """
        Create an identity matrix of size n x n.
        
        Parameters:
            n: Size of the identity matrix.
        
        Returns:
            Identity matrix as a numpy array.
        """
        return np.eye(n, dtype=float)
    
    def zeros(self, shape):
        """
        Create a zero matrix of given shape.
        
        Parameters:
            shape: Tuple specifying the shape of the zero matrix.
        
        Returns:
            Zero matrix as a numpy array.
        """
        return np.zeros(shape, dtype=float)
    
    def ones(self, shape):
        """
        Create a ones matrix of given shape.
        
        Parameters:
            shape: Tuple specifying the shape of the ones matrix.
        
        Returns:
            Ones matrix as a numpy array.
        """
        return np.ones(shape, dtype=float)
    
    def random(self, shape):
        """
        Create a random matrix of given shape.
        
        Parameters:
            shape: Tuple specifying the shape of the random matrix.
        
        Returns:
            Random matrix as a numpy array.
        """
        return np.random.rand(*shape)
    
    def reshape(self, A, new_shape):
        """
        Reshape a matrix to a new shape.
        
        Parameters:
            A: Matrix to be reshaped.
            new_shape: Tuple specifying the new shape.
        
        Returns:
            Reshaped matrix as a numpy array.
        
        Raises:
            ValueError: If the total number of elements does not match.
        """
        A = self.vector(A)
        if np.prod(A.shape) != np.prod(new_shape):
            raise ValueError("Total number of elements must match.")
        return A.reshape(new_shape)
    
    def flatten(self, A):
        """
        Flatten a matrix to a 1D array.
        
        Parameters:
            A: Matrix to be flattened.
        
        Returns:
            Flattened array as a numpy array.
        """
        A = self.vector(A)
        return A.flatten()
    
    def diagonal(self, A):
        """
        Extract the diagonal of a matrix.
        
        Parameters:
            A: Matrix from which to extract the diagonal.
        
        Returns:
            Diagonal elements as a numpy array.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.diag(A)
    
    def trace(self, A):
        """
        Compute the trace of a matrix.
        
        Parameters:
            A: Matrix for which to compute the trace.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.trace(A)
    
    def rank(self, A):
        """
        Compute the rank of a matrix.
        
        Parameters:
            A: Matrix for which to compute the rank.
        
        Returns:
            Rank of A as an integer.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.linalg.matrix_rank(A)
    
    def vector(self, A):
        """
        Convert input to a numpy array if it is not already.
        
        Parameters:
            A: Input to convert (can be list, tuple, or numpy array).
        
        Returns:
            Numpy array representation of A.
        """
        if isinstance(A, np.ndarray):
            return A
        elif isinstance(A, (list, tuple)):
            return np.array(A)
        else:
            raise ValueError("Input must be a list, tuple, or numpy array.")
        
    def __call__(self, A):
        """
        Make the class callable to convert input to a numpy array.
        
        Parameters:
            A: Input to convert (can be list, tuple, or numpy array).
        
        Returns:
            Numpy array representation of A.
        """
        return self.vector(A)
    
    def projection(self, A, B):
        """
        Project vector A onto vector B.
        
        Parameters:
            A: Vector to be projected.
            B: Vector onto which A is projected.
        
        Returns:
            Projection of A onto B as a numpy array.
        
        Raises:
            ValueError: If A or B is not 1D.
        """
        A = self.vector(A)
        B = self.vector(B)
        if A.ndim != 1 or B.ndim != 1:
            raise ValueError("Both A and B must be 1D arrays.")
        return (np.dot(A, B) / np.dot(B, B)) * B
    
    def angle_between(self, A, B):
        """
        Compute the angle between two vectors A and B in radians.
        
        Parameters:
            A: First vector.
            B: Second vector.
        
        Returns:
            Angle in radians as a float.
        
        Raises:
            ValueError: If A or B is not 1D or has zero norm.
        """
        A = self.vector(A)
        B = self.vector(B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        if norm_A == 0 or norm_B == 0:
            raise ValueError("Cannot compute angle for zero vectors.")
        cos_theta = np.dot(A, B) / (norm_A * norm_B)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
class MatrixOperations:
    def __init__(self):
        pass

    def LUdecomposition(self, A):
        """
        Perform LU decomposition of a matrix.
        
        Parameters:
            A: Matrix to decompose.
        
        Returns:
            L: Lower triangular matrix.
            U: Upper triangular matrix.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        L, U = np.linalg.lu(A)
        return L, U

    def QRdecomposition(self, A):
        """
        Perform QR decomposition of a matrix.
        
        Parameters:
            A: Matrix to decompose.
        
        Returns:
            Q: Orthogonal matrix.
            R: Upper triangular matrix.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        Q, R = np.linalg.qr(A)
        return Q, R
    
    def SVD(self, A):
        """
        Perform Singular Value Decomposition (SVD) of a matrix.
        
        Parameters:
            A: Matrix to decompose.
        
        Returns:
            U: Left singular vectors.
            S: Singular values (as a 1D array).
            Vh: Right singular vectors (conjugate transpose).
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        return U, S, Vh 
    
    def cholesky(self, A):
        """
        Perform Cholesky decomposition of a matrix.
        
        Parameters:
            A: Matrix to decompose (must be symmetric positive definite).
        
        Returns:
            L: Lower triangular matrix such that A = L * L^T.
        
        Raises:
            ValueError: If A is not 2D or not symmetric positive definite.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        if not np.allclose(A, A.T):
            raise ValueError("Matrix must be symmetric.")
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix must be positive definite.")
        return L
    
    def eigendecomposition(self, A):
        """
        Perform eigendecomposition of a matrix.
        
        Parameters:
            A: Matrix to decompose (must be square).
        
        Returns:
            eigenvalues: Eigenvalues of A.
            eigenvectors: Eigenvectors of A.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors
    
    def eigenvalues(self, A):
        """
        Compute the eigenvalues of a matrix.
        
        Parameters:
            A: Matrix to compute eigenvalues for (must be square).
        
        Returns:
            Numpy array of eigenvalues.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.linalg.eigvals(A)
    
    def eigenvectors(self, A):
        """
        Compute the eigenvectors of a matrix.

        Parameters:
            A: Matrix to compute eigenvectors for (must be square).

        Returns:
            Numpy array of eigenvectors.

        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        _, vecs = np.linalg.eig(A)
        return vecs
    
    def solve_linear_system(self, A, b):
        """
        Solve a linear system of equations Ax = b.
        
        Parameters:
            A: Coefficient matrix (must be square).
            b: Right-hand side vector (1D array).
        
        Returns:
            Solution vector x.
        """
        A = self.vector(A)
        b = self.vector(b)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError("Right-hand side vector must be 1D and match the number of rows in A.")
        return np.linalg.solve(A, b)
    
    def frobenius_norm(self, A):
        """
        Compute the Frobenius norm of a matrix.
        
        Parameters:
            A: Matrix to compute the Frobenius norm for (must be 2D).
        
        Returns:
            Frobenius norm as a float.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.linalg.norm(A, 'fro')
    
    def cramer_rule(self, A):
        """
        Compute the determinant of a matrix using Cramer's rule.
        
        Parameters:
            A: Square matrix to compute the determinant for.
        
        Returns:
            Determinant as a float.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.linalg.det(A) 
    
    def linear_conformation(self, A, x):
        """
        Apply a linear transformation defined by matrix A to vector x.
        
        Parameters:
            A: Transformation matrix (must be 2D).
            x: Input vector (1D array).
        
        Returns:
            Transformed vector as a numpy array.
        
        Raises:
            ValueError: If A is not 2D or x is not 1D.
        """
        A = self.vector(A)
        x = self.vector(x)
        if A.ndim != 2:
            raise ValueError("Transformation matrix must be 2D.")
        if x.ndim != 1 or x.shape[0] != A.shape[1]:
            raise ValueError("Input vector must be 1D and match the number of columns in A.")
        return np.dot(A, x)
    
    def vector_to_matrix(self, A, new_shape):
        """
        Convert a vector to a matrix with a specified shape.
        
        Parameters:
            A: Vector to be reshaped (1D array).
            new_shape: Desired shape for the matrix (tuple).
        
        Returns:
            Reshaped matrix as a numpy array.
        Raises:
            ValueError: If A is not 1D or new_shape is incompatible with the size of A.
        """
        A = self.vector(A)
        if A.ndim != 1:
            raise ValueError("Input must be a 1D array.")
        if np.prod(new_shape) != A.size:
            raise ValueError("New shape is incompatible with the size of the input vector.")
        return A.reshape(new_shape)
    
    def turn_matrix(self, A):
        """
        Turn a matrix into a vector by flattening it.
        
        Parameters:
            A: Matrix to be flattened (2D array).
        
        Returns:
            Flattened vector as a 1D numpy array.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return A.flatten()
    
    def rotate_matrix(self, A, angle):
        """
        Rotate a 2D matrix by a given angle in radians.
        
        Parameters:
            A: Matrix to rotate (2D array).
            angle: Angle in radians to rotate the matrix.
        
        Returns:
            Rotated matrix as a numpy array.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                     [np.sin(angle), np.cos(angle)]])
        return np.dot(rotation_matrix, A)
    
    def scale_matrix(self, A, scale):
        """
        Scale a matrix by a given factor.
        
        Parameters:
            A: Matrix to scale (2D array).
            scale: Scaling factor (float).
        
        Returns:
            Scaled matrix as a numpy array.
        
        Raises:
            ValueError: If A is not 2D or scale is not a number.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        if not isinstance(scale, (int, float)):
            raise ValueError("Scale must be a number.")
        return A * scale
    
    def translate_matrix(self, A, translation):
        """
        Translate a matrix by a given vector.
        
        Parameters:
            A: Matrix to translate (2D array).
            translation: Translation vector (1D array).
        
        Returns:
            Translated matrix as a numpy array.
        
        Raises:
            ValueError: If A is not 2D or translation is not 1D.
        """
        A = self.vector(A)
        translation = self.vector(translation)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        if translation.ndim != 1 or translation.shape[0] != A.shape[1]:
            raise ValueError("Translation must be a 1D array matching the number of columns in A.")
        return A + translation
    
    def reflect_matrix(self, A, axis):
        """
        Reflect a matrix across a specified axis.
        
        Parameters:
            A: Matrix to reflect (2D array).
            axis: Axis of reflection ('x' or 'y').
        
        Returns:
            Reflected matrix as a numpy array.
        
        Raises:
            ValueError: If A is not 2D or axis is not 'x' or 'y'.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        if axis not in ['x', 'y']:
            raise ValueError("Axis must be 'x' or 'y'.")
        if axis == 'x':
            return np.flipud(A)
        else:
            return np.fliplr(A)

    def parseval(self, A, B):
        """
        Check if Parseval's theorem holds for two matrices or vectors.
        
        Parameters:
            A: First matrix or vector.
            B: Second matrix or vector.
        
        Returns:
            True if Parseval's theorem holds, False otherwise.
        
        Raises:
            ValueError: If A or B is not 1D or has zero norm.
        """
        A = self.vector(A)
        B = self.vector(B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        if norm_A == 0 or norm_B == 0:
            raise ValueError("Cannot check Parseval's theorem for zero vectors.")
        return np.isclose(norm_A**2, norm_B**2, atol=1e-10)
    
    def gram_schmidt(self, A):
        """
        Perform Gram-Schmidt orthogonalization on a set of vectors.
        
        Parameters:
            A: Set of vectors (2D numpy array).
        
        Returns:
            Orthogonalized set of vectors as a numpy array.
        
        Raises:
            ValueError: If A is not 2D or has zero norm.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        Q = np.zeros_like(A)
        for i in range(A.shape[1]):
            v = A[:, i]
            for j in range(i):
                v -= np.dot(Q[:, j], A[:, i]) * Q[:, j]
            Q[:, i] = v / np.linalg.norm(v)
        return Q

class MatrixProperties:
    def __init__(self):
        pass
    
    def is_square(self, A):
        """
        Check if a matrix is square.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is square, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return A.shape[0] == A.shape[1]
    
    def is_symmetric(self, A):
        """
        Check if a matrix is symmetric.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is symmetric, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.allclose(A, A.T, atol=1e-10) 
    
    def is_square(self, A):
        """
        Check if a matrix is square.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is square, False otherwise.
        """
        A = self.vector(A)
        return A.ndim == 2 and A.shape[0] == A.shape[1]
    
    def is_symmetric(self, A):
        """
        Check if a matrix is symmetric.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is symmetric, False otherwise.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.array_equal(A, A.T)
    
    def is_orthogonal(self, A):
        """
        Check if a matrix is orthogonal.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is orthogonal, False otherwise.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.allclose(np.dot(A, A.T), np.eye(A.shape[0]), atol=1e-10)
    
    def is_positive_definite(self, A):
        """
        Check if a matrix is positive definite.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is positive definite, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.all(np.linalg.eigvals(A) > 0)
    
    def is_hermitian(self, A):
        """
        Check if a matrix is Hermitian (self-adjoint).
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is Hermitian, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.allclose(A, A.conj().T, atol=1e-10)
    
    def is_unitary(self, A):
        """
        Check if a matrix is unitary.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is unitary, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.allclose(np.dot(A, A.conj().T), np.eye(A.shape[0]), atol=1e-10)
    
    def is_diagonal(self, A):
        """
        Check if a matrix is diagonal.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is diagonal, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.all(np.diag(np.diagonal(A)) == A)
    
    def is_identity(self, A):
        """
        Check if a matrix is an identity matrix.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is an identity matrix, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.array_equal(A, np.eye(A.shape[0]), atol=1e-10)
    
    def is_zero(self, A):
        """
        Check if a matrix is a zero matrix.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is a zero matrix, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.all(A == 0)
    
    def is_one(self, A):
        """
        Check if a matrix is a ones matrix.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is a ones matrix, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.all(A == 1)
    
    def is_random(self, A):
        """
        Check if a matrix is a random matrix.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is a random matrix, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.all((A >= 0) & (A <= 1))
    
    def is_singular(self, A):
        """
        Check if a matrix is singular (not invertible).
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is singular, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.linalg.matrix_rank(A) < A.shape[0]
    
    def is_non_singular(self, A):
        """
        Check if a matrix is non-singular (invertible).
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A is non-singular, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.linalg.matrix_rank(A) == A.shape[0]
    
    def is_orthonormal(self, A):
        """
        Check if a matrix has orthonormal columns.
        
        Parameters:
            A: Matrix to check.
        
        Returns:
            True if A has orthonormal columns, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return np.allclose(np.dot(A.T, A), np.eye(A.shape[1]), atol=1e-10)
    
    def normalize(self, A):
        """
        Normalize a matrix or vector.
        
        Parameters:
            A: Matrix or vector to normalize.
        
        Returns:
            Normalized matrix or vector as a numpy array.
        
        Raises:
            ValueError: If A is empty or has zero norm.
        """
        A = self.vector(A)
        norm = np.linalg.norm(A)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector/matrix.")
        return A / norm
    
    def is_normalized(self, A):
        """
        Check if a matrix or vector is normalized.
        
        Parameters:
            A: Matrix or vector to check.
        
        Returns:
            True if A is normalized, False otherwise.
        
        Raises:
            ValueError: If A is empty or has zero norm.
        """
        A = self.vector(A)
        norm = np.linalg.norm(A)
        if norm == 0:
            raise ValueError("Cannot check normalization of a zero vector/matrix.")
        return np.isclose(norm, 1.0, atol=1e-10)
    
    def is_orthogonal_to(self, A, B):
        """
        Check if two vectors or matrices are orthogonal.
        
        Parameters:
            A: First vector or matrix.
            B: Second vector or matrix.
        
        Returns:
            True if A and B are orthogonal, False otherwise.
        
        Raises:
            ValueError: If A or B is empty or has zero norm.
        """
        A = self.vector(A)
        B = self.vector(B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        if norm_A == 0 or norm_B == 0:
            raise ValueError("Cannot check orthogonality of a zero vector/matrix.")
        return np.isclose(np.dot(A, B), 0.0, atol=1e-10)
    
    def is_collinear(self, A, B):
        """
        Check if two vectors are collinear.
        
        Parameters:
            A: First vector.
            B: Second vector.
        
        Returns:
            True if A and B are collinear, False otherwise.
        
        Raises:
            ValueError: If A or B is empty or has zero norm.
        """
        A = self.vector(A)
        B = self.vector(B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        if norm_A == 0 or norm_B == 0:
            raise ValueError("Cannot check collinearity of a zero vector.")
        return np.isclose(np.cross(A, B), 0.0, atol=1e-10)
    
    def is_orthogonal_basis(self, A):
        """
        Check if a set of vectors forms an orthogonal basis.
        
        Parameters:
            A: Set of vectors (2D numpy array).
        
        Returns:
            True if A forms an orthogonal basis, False otherwise.
        
        Raises:
            ValueError: If A is not 2D or has zero norm.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        for i in range(A.shape[1]):
            for j in range(i + 1, A.shape[1]):
                if not self.is_orthogonal_to(A[:, i], A[:, j]):
                    return False
        return True
    
    def is_linearly_independent(self, A):
        """
        Check if a set of vectors is linearly independent.
        
        Parameters:
            A: Set of vectors (2D numpy array).
        
        Returns:
            True if A is linearly independent, False otherwise.
        
        Raises:
            ValueError: If A is not 2D or has zero norm.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        return np.linalg.matrix_rank(A) == A.shape[1]
    
    def is_spanning_set(self, A, B):
        """
        Check if a set of vectors spans another vector space.
        
        Parameters:
            A: Set of vectors (2D numpy array).
            B: Vector space to check (1D numpy array).
        
        Returns:
            True if A spans B, False otherwise.
        
        Raises:
            ValueError: If A is not 2D or B is not 1D.
        """
        A = self.vector(A)
        B = self.vector(B)
        if A.ndim != 2 or B.ndim != 1:
            raise ValueError("A must be a 2D array and B must be a 1D array.")
        return np.linalg.matrix_rank(A) == A.shape[0] and np.all(np.isfinite(np.linalg.lstsq(A, B, rcond=None)[0]))
    
    def is_subspace(self, A, B):
        """
        Check if a set of vectors is a subspace of another vector space.
        
        Parameters:
            A: Set of vectors (2D numpy array).
            B: Vector space to check (1D numpy array).
        
        Returns:
            True if A is a subspace of B, False otherwise.
        
        Raises:
            ValueError: If A is not 2D or B is not 1D.
        """
        A = self.vector(A)
        B = self.vector(B)
        if A.ndim != 2 or B.ndim != 1:
            raise ValueError("A must be a 2D array and B must be a 1D array.")
        return np.linalg.matrix_rank(A) <= np.linalg.matrix_rank(B.reshape(-1, 1))
    
    def is_basis(self, A):
        """
        Check if a set of vectors forms a basis for a vector space.
        
        Parameters:
            A: Set of vectors (2D numpy array).
        
        Returns:
            True if A forms a basis, False otherwise.
        
        Raises:
            ValueError: If A is not 2D or has zero norm.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        return self.is_linearly_independent(A) and self.is_spanning_set(A, np.zeros(A.shape[0]))
    
    def is_orthogonal_complement(self, A, B):
        """
        Check if a set of vectors is the orthogonal complement of another set.
        
        Parameters:
            A: Set of vectors (2D numpy array).
            B: Set of vectors (2D numpy array).
        
        Returns:
            True if A is the orthogonal complement of B, False otherwise.
        
        Raises:
            ValueError: If A or B is not 2D or has zero norm.
        """
        A = self.vector(A)
        B = self.vector(B)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Both A and B must be 2D arrays.")
        return np.all([self.is_orthogonal_to(a, b) for a in A.T for b in B.T])
    
    def is_orthogonal_projection(self, A, B):
        """
        Check if a set of vectors is the orthogonal projection of another set.
        
        Parameters:
            A: Set of vectors (2D numpy array).
            B: Set of vectors (2D numpy array).
        
        Returns:
            True if A is the orthogonal projection of B, False otherwise.
        
        Raises:
            ValueError: If A or B is not 2D or has zero norm.
        """
        A = self.vector(A)
        B = self.vector(B)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Both A and B must be 2D arrays.")
        return np.allclose(A, np.dot(B, np.linalg.pinv(B)), atol=1e-10)
    
    def is_orthogonal_basis_complement(self, A, B):
        """
        Check if a set of vectors is the orthogonal basis complement of another set.
        
        Parameters:
            A: Set of vectors (2D numpy array).
            B: Set of vectors (2D numpy array).
        
        Returns:
            True if A is the orthogonal basis complement of B, False otherwise.
        
        Raises:
            ValueError: If A or B is not 2D or has zero norm.
        """
        A = self.vector(A)
        B = self.vector(B)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Both A and B must be 2D arrays.")
        return self.is_orthogonal_complement(A, B) and self.is_basis(A)
    
    def is_diagonalizable(self, A):
        """
        Check if a matrix is diagonalizable.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is diagonalizable, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        eigenvalues, eigenvectors = self.eigendecomposition(A)
        return np.linalg.matrix_rank(eigenvectors) == A.shape[0]
    
    def is_symmetric_positive_definite(self, A):
        """
        Check if a matrix is symmetric positive definite.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric positive definite, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return np.all(np.linalg.eigvals(A) > 0) and np.allclose(A, A.T)
    
    def is_symmetric_orthogonal(self, A):
        """
        Check if a matrix is symmetric orthogonal.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric orthogonal, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_orthogonal(A)
    
    def is_symmetric_hermitian(self, A):
        """
        Check if a matrix is symmetric Hermitian.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric Hermitian, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_hermitian(A)
    
    def is_symmetric_unitary(self, A):
        """
        Check if a matrix is symmetric unitary.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric unitary, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_unitary(A)
    
    def is_symmetric_diagonal(self, A):
        """
        Check if a matrix is symmetric diagonal.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric diagonal, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_diagonal(A)
    
    def is_symmetric_identity(self, A):
        """
        Check if a matrix is symmetric identity.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric identity, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_identity(A)
    
    def is_symmetric_zero(self, A):
        """
        Check if a matrix is symmetric zero.
        
        Parameters:
            A: Matrix to check (must be 2D).
        
        Returns:
            True if A is symmetric zero, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return self.is_symmetric(A) and self.is_zero(A)
    
    def is_symmetric_one(self, A):
        """
        Check if a matrix is symmetric one.
        
        Parameters:
            A: Matrix to check (must be 2D).
        
        Returns:
            True if A is symmetric one, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return self.is_symmetric(A) and self.is_one(A)
    
    def is_symmetric_random(self, A):
        """
        Check if a matrix is symmetric random.
        
        Parameters:
            A: Matrix to check (must be 2D).
        
        Returns:
            True if A is symmetric random, False otherwise.
        
        Raises:
            ValueError: If A is not 2D.
        """
        A = self.vector(A)
        if A.ndim != 2:
            raise ValueError("Matrix must be 2D.")
        return self.is_symmetric(A) and self.is_random(A)
    
    def is_symmetric_singular(self, A):
        """
        Check if a matrix is symmetric singular.

        Parameters:
            A: Matrix to check (must be square).

        Returns:
            True if A is symmetric singular, False otherwise.

        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_singular(A)
    
    def is_symmetric_non_singular(self, A):
        """
        Check if a matrix is symmetric non-singular.

        Parameters:
            A: Matrix to check (must be square).

        Returns:
            True if A is symmetric non-singular, False otherwise.

        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_non_singular(A)
    
    def is_symmetric_orthonormal(self, A):
        """
        Check if a matrix is symmetric orthonormal.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric orthonormal, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_orthonormal(A)

    def is_symmetric_normal(self, A):
        """
        Check if a matrix is symmetric normal.
        
        Parameters:
            A: Matrix to check (must be square).
        
        Returns:
            True if A is symmetric normal, False otherwise.
        
        Raises:
            ValueError: If A is not square.
        """
        A = self.vector(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square.")
        return self.is_symmetric(A) and self.is_normal(A)
    
    def is_symmetric_orthogonal_complement(self, A, B):
        """
        Check if the orthogonal complement of B is symmetric with respect to A.

        Parameters:
            A: First matrix (must be square).
            B: Second matrix (must be square).

        Returns:
            True if the orthogonal complement of B is symmetric with respect to A, False otherwise.

        Raises:
            ValueError: If A or B is not square.
        """
        A = self.vector(A)
        B = self.vector(B)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise ValueError("Matrix B must be square.")

        # Compute the orthogonal complement of B with respect to A
        B_orthogonal = B - self.projection(B, A)

        # Check if the orthogonal complement is symmetric
        return self.is_symmetric(B_orthogonal)
