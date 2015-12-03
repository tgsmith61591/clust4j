package com.clust4j.utils;

import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;

public class MatUtils {
	public final static double TINY = 2.2250738585072014e-308;
	public final static double EPS = 2.2204460492503131e-16;
	public final static int MIN_ACCEPTABLE_MAT_LEN = 1;
	
	public static enum Axis {
		ROW, COL
	}
	
	final static protected void checkDims(final double[][] a) {
		if(a.length < MIN_ACCEPTABLE_MAT_LEN) throw new IllegalArgumentException("illegal mat row dim:" + a.length);
		VecUtils.checkDims(a[0]);
	}
	
	public static final double[][] add(final double[][] a, final double[][] b) {
		checkDims(a);
		checkDims(b);
		
		final int m = a.length, n = a[0].length;
		if(b.length != m)
			throw new DimensionMismatchException(b.length, m);
		if(b[0].length != n)
			throw new DimensionMismatchException(b[0].length, n);
			
		final double[][] c = new double[m][n];
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				c[i][j] = a[i][j] + b[i][j];
		return c;
	}
	
	
	/**
	 * The indices of the max along axes
	 * @param data
	 * @param axis: row or column wise
	 * @return
	 */
	public static int[] argMax(final double[][] data, final Axis axis) {
		if(data.length == 0)
			return new int[0];
		
		int[] out;
		final int m=data.length, n=data[0].length;
		if(axis.equals(Axis.COL)) {
			out = new int[n];
			double[] col;
			for(int i = 0; i < n; i++) {
				col = getColumn(data, i);
				out[i] = VecUtils.argMax(col);
			}
		} else {
			out = new int[m];
			for(int i = 0; i < m; i++)
				out[i] = VecUtils.argMax(data[i]);
		}
		
		return out;
	}
	
	/**
	 * The indices of the max along axes
	 * @param data
	 * @param axis: row or column wise
	 * @return
	 */
	public static int[] argMin(final double[][] data, final Axis axis) {
		if(data.length == 0)
			return new int[0];
		
		int[] out;
		final int m=data.length, n=data[0].length;
		if(axis.equals(Axis.COL)) {
			out = new int[n];
			double[] col;
			for(int i = 0; i < n; i++) {
				col = getColumn(data, i);
				out[i] = VecUtils.argMin(col);
			}
		} else {
			out = new int[m];
			for(int i = 0; i < m; i++)
				out[i] = VecUtils.argMin(data[i]);
		}
		
		return out;
	}
	
	
	public static double[] colSums(final double[][] data) {
		checkDims(data);
		
		final double[] out = new double[data[0].length];
		for(int i = 0; i < out.length; i++)
			out[i] = VecUtils.sum(getColumn(data, i));
		
		return out;
	}
	
	
	/**
	 * Copy a 2d double array
	 * @param data
	 * @return
	 */
	public static final double[][] copyMatrix(final double[][] data) {
		final double[][] copy = new double[data.length][];
		
		if(data.length != 0) {
			for(int i = 0; i < copy.length; i++)
				copy[i] = VecUtils.copy(data[i]);
		}
		
		return copy;
	}
	
	public static double[] diagFromSquare(final double[][] data) {
		checkDims(data);
		
		final int m = data.length, n = data[0].length;
		if(m!=n)
			throw new DimensionMismatchException(m, n);
		
		final double[] out = new double[n];
		for(int i = 0; i < m; i++) {
			if(data[i].length != n) // Check for jagged array
				throw new DimensionMismatchException(data[i].length, n);
			out[i] = data[i][i];
		}
		
		return out;
	}
	
	
	public static boolean equalsExactly(final double[][] a, final double[][] b) {
		return equalsWithTolerance(a,b,0.0);
	}
	
	public static boolean equalsWithTolerance(final double[][] a, final double[][] b) {
		return equalsWithTolerance(a,b,Precision.EPSILON);
	}
	
	public static boolean equalsWithTolerance(final double[][] a, final double[][] b, final double tol) {
		if(a.length != b.length)
			return false;
		if(a.length == 0) // Both are empty
			return true;
		
		for(int i = 0; i < a.length; i++)
			if(!VecUtils.equalsWithTolerance(a[i], b[i], tol))
				return false;
		return true;
	}
	
	public static double[] flatten(final double[][] a) {
		checkDims(a);
		
		final int m = a.length, n = a[0].length;
		final double[] out = new double[m * n];
		int ctr = 0;
		for(int i = 0; i < m; i++) {
			final double[] row = a[i];
			if(row.length != n) // Check for jaggedness
				throw new DimensionMismatchException(n, row.length);
			for(int j = 0; j < n; j++)
				out[ctr++] = a[i][j];
		}
		
		return out;
	}
	
	/**
	 * Build a matrix from a vector
	 * @param v
	 * @param axis: which axis each value in the vector represents
	 * @return
	 */
	public static double[][] fromVector(final double[] v, final int repCount, final Axis axis) {
		VecUtils.checkDims(v);
		
		if(repCount < 1)
			throw new IllegalArgumentException("repCount cannot be less than 1");
		
		double[][] out;
		if(axis.equals(Axis.ROW)) {
			out = new double[v.length][repCount];
			
			for(int i = 0; i < out.length; i++)
				for(int j = 0; j < out[0].length; j++)
					out[i][j] = v[i];
		} else {
			out = new double[repCount][v.length];
			
			for(int i = 0; i < out.length; i++)
				out[i] = VecUtils.copy(v);
		}
		
		return out;
	}
	
	public static double[] getColumn(final double[][] data, final int idx) {
		checkDims(data);
		
		final int m=data.length, n=data[0].length;
		if(idx >= n || idx < 0)
			throw new IndexOutOfBoundsException(idx+"");
		
		final double[] col = new double[m];
		for(int i = 0; i < m; i++)
			col[i] = data[i][idx];
		
		return col;
	}
	
	public static double[][] getColumns(final double[][] data, final int[] idcs) {
		final double[][] out = new double[data.length][idcs.length];
		
		int idx = 0;
		for(int i = 0; i < idcs.length; i++)
			setColumnInPlace(out, idx++, getColumn(data, idcs[i]));
		
		return out;
	}
	
	public static double[][] getRows(final double[][] data, final int[] idcs) {
		final double[][] out = new double[idcs.length][];
		
		int idx = 0;
		for(int i = 0; i < idcs.length; i++) {
			out[idx] = new double[data[i].length];
			setRowInPlace(out, idx++, data[idcs[i]]);
		}
		
		return out;
	}
	
	public static double[] max(final double[][] data, final Axis axis) {
		if(data.length == 0)
			return new double[0];
		
		double[] out;
		final int m=data.length, n=data[0].length;
		if(axis.equals(Axis.COL)) {
			out = new double[n];
			double[] col;
			for(int i = 0; i < n; i++) {
				col = getColumn(data, i);
				out[i] = VecUtils.max(col);
			}
		} else {
			out = new double[m];
			for(int i = 0; i < m; i++)
				out[i] = VecUtils.max(data[i]);
		}
		
		return out;
	}
	
	public static double[] min(final double[][] data, final Axis axis) {
		if(data.length == 0)
			return new double[0];
		
		double[] out;
		final int m=data.length, n=data[0].length;
		if(axis.equals(Axis.COL)) {
			out = new double[n];
			double[] col;
			for(int i = 0; i < n; i++) {
				col = getColumn(data, i);
				out[i] = VecUtils.min(col);
			}
		} else {
			out = new double[m];
			for(int i = 0; i < m; i++)
				out[i] = VecUtils.min(data[i]);
		}
		
		return out;
	}
	
	/**
	 * Returns the mean row from a matrix
	 * @param data
	 * @return
	 */
	public static double[] meanRecord(final double[][] data) {
		checkDims(data);
		
		// Note: could use VecUtils.mean(...) and getColumn(...)
		// in conjunction here, but this is a faster hack, though
		// somewhat code duplicative...
		
		final int m=data.length, n=data[0].length;
		final double[] sums = new double[n];
		
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				sums[j] += data[i][j];
				if(i == m-1)
					sums[j] /= m;
			}
		}
		
		return sums;
	}
	
	public static double[] medianRecord(final double[][] data) {
		checkDims(data);
		
		final int n = data[0].length;
		final double[] median = new double[n];
		for(int j = 0; j < n; j++)
			median[j] = VecUtils.median(getColumn(data, j));
		
		return median;
	}
	
	public static double[][] multiply(final double[][] a, final double[][] b) {
		/*checkDims(a);
		checkDims(b);
		
		final int m1=a.length, n1=a[0].length;
		final int m2=b.length, n2=b[0].length;
		
		if(n1 != m2)
			throw new DimensionMismatchException(n1, m2);
		
		final double[][] c = new double[m1][n2];
		for(int i = 0; i < m1; i++)
			for(int j = 0; j < n2; j++)
				for(int k = 0; k < n1; k++)
					c[i][j] += a[i][k] * b[k][j];
		
		return c;*/
		
		final Array2DRowRealMatrix aa = new Array2DRowRealMatrix(a, false);
		final Array2DRowRealMatrix bb = new Array2DRowRealMatrix(b, false);
		return aa.multiply(bb).getDataRef();
	}
	
	
	/**
	 * Invert the sign of every element in a matrix, return a copy
	 * @param data
	 * @return
	 */
	public static double[][] negative(final double[][] data) {
		final double[][] copy = MatUtils.copyMatrix(data);
		for(int i = 0; i < copy.length; i++)
			for(int j = 0; j < copy[i].length; j++)
				copy[i][j] = -copy[i][j];
		return copy;
	}
	
	
	public static double[][] randomGaussian(final int m, final int n) {
		return randomGaussian(m, n, new Random());
	}
	
	
	public static double[][] randomGaussian(final int m, final int n, final Random seed) {
		if(m < 0 || n < 0)
			throw new IllegalArgumentException("illegal dimensions");
		
		final double[][] out = new double[m][n];
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				out[i][j] = seed.nextGaussian();
		
		return out;
	}
	
	public static double[] rowSums(final double[][] data) {
		checkDims(data);
		
		final double[] out = new double[data.length];
		for(int i = 0; i < out.length; i++)
			out[i] = VecUtils.sum(data[i]);
		
		return out;
	}
	
	public static double[][] scalarAdd(final double[][] data, final double scalar) {
		checkDims(data);
		
		final double[][] copy = copyMatrix(data);
		final int m=copy.length, n=copy[0].length;
		
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				copy[i][j] += scalar;
		
		return copy;
	}
	
	public static double[][] scalarDivide(final double[][] data, final double scalar) {
		checkDims(data);
		
		final double[][] copy = copyMatrix(data);
		final int m=copy.length, n=copy[0].length;
		
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				copy[i][j] /= scalar;
		
		return copy;
	}
	
	public static double[][] scalarMultiply(final double[][] data, final double scalar) {
		checkDims(data);
		
		final double[][] copy = copyMatrix(data);
		final int m=copy.length, n=copy[0].length;
		
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				copy[i][j] *= scalar;
		
		return copy;
	}
	
	public static double[][] scalarSubtract(final double[][] data, final double scalar) {
		checkDims(data);
		
		final double[][] copy = copyMatrix(data);
		final int m=copy.length, n=copy[0].length;
		
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				copy[i][j] -= scalar;
		
		return copy;
	}
	
	public static void setColumnInPlace(final double[][] a, final int idx, final double[] v) {
		checkDims(a);
		
		final int m = a.length, n = a[0].length;
		VecUtils.checkDims(getColumn(a, 0), v);
		
		if(idx < 0 || idx >= n)
			throw new IndexOutOfBoundsException("illegal idx: " + idx);
		
		for(int i = 0; i < m; i++)
			a[i][idx] = v[i];
	}
	
	public static void setRowInPlace(final double[][] a, final int idx, final double[] v) {
		checkDims(a);
		
		final int m = a.length, n = a[0].length;
		VecUtils.checkDims(a[0], v);

		if(idx < 0 || idx >= m)
			throw new IndexOutOfBoundsException("illegal idx: " + idx);
		
		for(int i = 0; i < n; i++)
			a[idx][i] = v[i];
	}
	
	public static final double[][] subtract(final double[][] a, final double[][] b) {
		checkDims(a);
		checkDims(b);
		
		final int m = a.length, n = a[0].length;
		if(b.length != m)
			throw new DimensionMismatchException(b.length, m);
		if(b[0].length != n)
			throw new DimensionMismatchException(b[0].length, n);
			
		final double[][] c = new double[m][n];
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				c[i][j] = a[i][j] - b[i][j];
		return c;
	}
}
