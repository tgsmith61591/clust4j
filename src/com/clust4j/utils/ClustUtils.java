package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

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
	
	public static double[][] distanceFullMatrix(final AbstractRealMatrix data, GeometricallySeparable dist) {
		return distanceFullMatrix(data.getData(), dist);
	}
	
	public static double[][] distanceFullMatrix(final double[][] data, GeometricallySeparable dist) {
		final int m = data.length;
		
		// Runs in O(m choose 2)
		final double[][] dist_mat = new double[m][m];
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) {
				final double d = dist.getDistance(data[i], data[j]);
				dist_mat[i][j] = d;
				dist_mat[j][i] = d;
			}
		}
		
		// One extra pass of M for the diagonal
		for(int i = 0; i < m; i++)
			dist_mat[i][i] = dist.getDistance(data[i], data[i]);
		
		return dist_mat;
	}
	
	/**
	 * Calculate the upper triangular distance matrix given an AbstractRealMatrix
	 * and an instance of GeometricallySeparable.
	 * @param data
	 * @param dist
	 * @return
	 */
	final public static double[][] distanceUpperTriangMatrix(final AbstractRealMatrix data, GeometricallySeparable dist) {
		return distanceUpperTriangMatrix(data.getData(), dist);
	}
	
	/**
	 * Calculate the upper triangular distance matrix given an AbstractRealMatrix
	 * and an instance of GeometricallySeparable.
	 * @param data
	 * @param dist
	 * @return
	 */
	final public static double[][] distanceUpperTriangMatrix(final double[][] data, GeometricallySeparable dist) {
		final int m = data.length;
		
		// Compute distance matrix, which is O(N^2) space, O(Nc2) time
		// We do this in KMedoids and not KMeans, because KMedoids uses
		// real points as medoids and not means for centroids, thus
		// the recomputation of distances is unnecessary with the dist mat
		final double[][] dist_mat = new double[m][m];
		for(int i = 0; i < m - 1; i++)
			for(int j = i + 1; j < m; j++)
				dist_mat[i][j] = dist.getDistance(data[i], data[j]);
		
		return dist_mat;
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
	
	public static double[][] similarityFullMatrix(final AbstractRealMatrix data, SimilarityMetric sim) {
		return similarityFullMatrix(data.getData(), sim);
	}
	
	public static double[][] similarityFullMatrix(final double[][] data, SimilarityMetric sim) {
		final int m = data.length;
		
		// Runs in O(m choose 2)
		final double[][] dist_mat = new double[m][m];
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) {
				final double d = sim.getSimilarity(data[i], data[j]);
				dist_mat[i][j] = d;
				dist_mat[j][i] = d;
			}
		}
		
		// One extra pass of M for the diagonal
		for(int i = 0; i < m; i++)
			dist_mat[i][i] = sim.getSimilarity(data[i], data[i]);
		
		return dist_mat;
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
	
	/**
	 * Calculate the upper triangular similarity matrix given an AbstractRealMatrix
	 * and an instance of SimilarityMetric.
	 * @param data
	 * @param dist
	 * @return
	 */
	final public static double[][] similarityUpperTriangMatrix(final AbstractRealMatrix data, SimilarityMetric sim) {
		return similarityUpperTriangMatrix(data.getData(),sim);
	}
	
	/**
	 * Calculate the upper triangular similarity matrix given an AbstractRealMatrix
	 * and an instance of SimilarityMetric.
	 * @param data
	 * @param dist
	 * @return
	 */
	final public static double[][] similarityUpperTriangMatrix(final double[][] data, SimilarityMetric sim) {
		final int m = data.length;
		
		// Compute similarity matrix, which is O(N^2) space, O(Nc2) time
		// We do this in KMedoids and not KMeans, because KMedoids uses
		// real points as medoids and not means for centroids, thus
		// the recomputation of distances is unnecessary with the dist mat
		final double[][] sim_mat = new double[m][m];
		for(int i = 0; i < m - 1; i++)
			for(int j = i + 1; j < m; j++)
				sim_mat[i][j] = sim.getSimilarity(data[i], data[j]);
		
		return sim_mat;
	}
	
	
	
	final public static <K,V extends Comparable<? super V>> SortedSet<Map.Entry<K,V>> sortEntriesByValue(Map<K,V> map) {
		return sortEntriesByValue(map, false);
	}
	
	
	
	final public static <K,V extends Comparable<? super V>> 
		SortedSet<Map.Entry<K,V>> sortEntriesByValue(Map<K,V> map, final boolean desc) 
	{
		SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(
			new Comparator<Map.Entry<K,V>>() {
				@Override public int compare(Map.Entry<K,V> e1, Map.Entry<K,V> e2) {
					int res = e1.getValue().compareTo(e2.getValue());
					return (res != 0 ? res : 1) * (desc ? -1 : 1);
				}
			}
		);
		
		sortedEntries.addAll(map.entrySet());
		return sortedEntries;
	}
	
	
	final public static <K,V extends Comparable<? super V>> 
			Collection<Map.Entry<K,V>> sortEntriesByValue(Collection<? extends Map.Entry<K,V>> col) {
		
		return sortEntriesByValue(col, false);
	}
	
	
	
	final public static <K,V extends Comparable<? super V>> 
			Collection<Map.Entry<K,V>> sortEntriesByValue(Collection<? extends Map.Entry<K,V>> col, 
					final boolean desc) {
		
		Comparator<Map.Entry<K,V>> comp = new Comparator<Map.Entry<K,V>>() {
			@Override public int compare(Map.Entry<K,V> e1, Map.Entry<K,V> e2) {
				int res = e1.getValue().compareTo(e2.getValue());
				int scale = desc ? -1 : 1;
				return res * scale;
			}
		};
		
		ArrayList<Map.Entry<K,V>> sorted = new ArrayList<>(col);
		Collections.sort(sorted, comp);
		return sorted;
	}
	
	
	
	final public static <K,V extends Comparable<? super V>> 
		SortedSet<Map.Entry<K,V>> sortEntriesByValue(
				Map<K,V> map, 
				final Comparator<Map.Entry<K,V>> cmp) 
	{
		SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(cmp);
		sortedEntries.addAll(map.entrySet());
		return sortedEntries;
	}
}
