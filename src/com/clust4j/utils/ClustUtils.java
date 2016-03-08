package com.clust4j.utils;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.SimilarityMetric;

public class ClustUtils {
	
	/**
	 * Computes a flattened upper triangular distance matrix in a much more space efficient manner,
	 * however traversing it requires intermittent calculations using {@link #navigateFlattenedMatrix(double[], int, int, int)}
	 * @param data
	 * @param dist
	 * @return a flattened distance vector
	 */
	public static double[] distanceFlatVector(final AbstractRealMatrix data, GeometricallySeparable dist) {
		return distanceFlatVector(data.getData(), dist);
	}
	
	/**
	 * Computes a flattened upper triangular distance matrix in a much more space efficient manner,
	 * however traversing it requires intermittent calculations using {@link #navigateFlattenedMatrix(double[], int, int, int)}
	 * @param data
	 * @param dist
	 * @return a flattened distance vector
	 */
	public static double[] distanceFlatVector(final double[][] data, GeometricallySeparable dist) {
		final int m = data.length;
		final int s = m*(m-1)/2; // The shape of the flattened upper triangular matrix (m choose 2)
		final double[] vec = new double[s];
		for(int i = 0, r = 0; i < m - 1; i++)
			for(int j = i + 1; j < m; j++, r++)
				vec[r] = dist.getDistance(data[i], data[j]);
		
		return vec;
	}

	public static double minDist(final double[][] data) {
		final int m = data.length;
		double min = Double.MAX_VALUE;
		
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) {
				final double current = data[i][j];
				if(current < min)
					min = current;
			}
		}
		
		return min;
	}
	
	/**
	 * For a flattened upper triangular matrix...
	 * 
	 * <p>
	 * Original:
	 * <p>
	 * <table>
	 * <tr><td>0 </td><td>1 </td><td>2 </td><td>3</td></tr>
	 * <tr><td>0 </td><td>0 </td><td>1 </td><td>2</td></tr>
	 * <tr><td>0 </td><td>0 </td><td>0 </td><td>1</td></tr>
	 * <tr><td>0 </td><td>0 </td><td>0 </td><td>0</td></tr>
	 * </table>
	 * 
	 * <p>
	 * Flattened:
	 * <p>
	 * &lt;1 2 3 1 2 1&gt;
	 * 
	 * <p>
	 * ...and the parameters <tt>m</tt>, the original row dimension,
	 * <tt>i</tt> and <tt>j</tt>, will identify the corresponding index
	 * in the flattened vector such that mat[0][3] corresponds to vec[2];
	 * this method, then, would return 2 (the index in the vector 
	 * corresponding to mat[0][3]) in this case.
	 * 
	 * @param m
	 * @param i
	 * @param j
	 * @return the corresponding vector index
	 */
	public static int getIndexFromFlattenedVec(final int m, final int i, final int j) {
		if(i < j)
			return m * i - (i * (i + 1) / 2) + (j - i - 1);
		else if(i > j)
			return m * j - (j * (j + 1) / 2) + (i - j - 1);
		throw new IllegalArgumentException(i+", "+j+"; i should not equal j");
	}
	
	/**
	 * For a flattened upper triangular matrix...
	 * 
	 * <p>
	 * Original:
	 * <p>
	 * <table>
	 * <tr><td>0 </td><td>1 </td><td>2 </td><td>3</td></tr>
	 * <tr><td>0 </td><td>0 </td><td>1 </td><td>2</td></tr>
	 * <tr><td>0 </td><td>0 </td><td>0 </td><td>1</td></tr>
	 * <tr><td>0 </td><td>0 </td><td>0 </td><td>0</td></tr>
	 * </table>
	 * 
	 * <p>
	 * Flattened:
	 * <p>
	 * &lt;1 2 3 1 2 1&gt;
	 * 
	 * <p>
	 * ...and the parameters <tt>m</tt>, the original row dimension,
	 * <tt>i</tt> and <tt>j</tt>, will identify the corresponding value
	 * in the flattened vector such that mat[0][3] corresponds to vec[2];
	 * this method, then, would return 3, the value at index 2, in this case.
	 * 
	 * @param m
	 * @param i
	 * @param j
	 * @return the corresponding vector index
	 */
	public static double navigateFlattenedMatrix(final double[] vector, final int m, final int i, final int j) {
		return vector[getIndexFromFlattenedVec(m,i,j)];
	}
	
	/**
	 * Computes a flattened upper triangular similarity matrix in a much more space efficient manner,
	 * however traversing it requires intermittent calculations using {@link #navigateFlattenedMatrix(double[], int, int, int)}
	 * @param data
	 * @param dist
	 * @return a flattened similarity vector
	 */
	public static double[] similarityFlatVector(final AbstractRealMatrix data, SimilarityMetric sim) {
		return similarityFlatVector(data.getData(), sim);
	}
	
	/**
	 * Computes a flattened upper triangular similarity matrix in a much more space efficient manner,
	 * however traversing it requires intermittent calculations using {@link #navigateFlattenedMatrix(double[], int, int, int)}
	 * @param data
	 * @param dist
	 * @return a flattened similarity vector
	 */
	public static double[] similarityFlatVector(final double[][] data, SimilarityMetric sim) {
		final int m = data.length;
		final int s = m*(m-1)/2; // The shape of the flattened upper triangular matrix (m choose 2)
		final double[] vec = new double[s];
		for(int i = 0, r = 0; i < m - 1; i++)
			for(int j = i + 1; j < m; j++, r++)
				vec[r] = sim.getSimilarity(data[i], data[j]);
		
		return vec;
	}
}
