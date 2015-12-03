package com.clust4j.utils;

import java.util.Comparator;
import java.util.Iterator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public class ClustUtils {
	/**
	 * This utility class is used in some cluster classes to keep
	 * track of medoid indices. It overrides TreeSet's equals()
	 * method and returns true if all of the integer values are equal,
	 * but exploits the underlying hashCode() method to check in contained
	 * in hashable collections.
	 * @author Taylor G Smith
	 */
	final public static class SortedHashableIntSet extends TreeSet<Integer> {
		private static final long serialVersionUID = -5206257978720286064L;
		
		@Override public boolean equals(Object o) {
			if(this == o)
				return true;
			if(o instanceof SortedHashableIntSet) {
				SortedHashableIntSet s = (SortedHashableIntSet) o;
				if(s.size() == this.size()) {
					Iterator<Integer> aIter = this.iterator();
					Iterator<Integer> bIter = s.iterator();
					
					while(aIter.hasNext() && bIter.hasNext()) {
						if( aIter.next().intValue() != bIter.next().intValue() )
							return false;
					}
					
					return true;
				}
			}
			
			return false;
		}
		
		public static SortedHashableIntSet fromArray(final int[] a) {
			final SortedHashableIntSet s = new SortedHashableIntSet();
			for(int i: a)
				s.add(i);
			return s;
		}
	}
	
	public static double[][] distanceFullMatrix(final AbstractRealMatrix data, GeometricallySeparable dist) {
		return distanceFullMatrix(data.getData(), dist);
	}
	
	public static double[][] distanceFullMatrix(final double[][] data, GeometricallySeparable dist) {
		final int m = data.length;
		
		final double[][] dist_mat = new double[m][m];
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) {
				final double d = dist.getDistance(data[i], data[j]);
				dist_mat[i][j] = d;
				dist_mat[j][i] = d;
			}
		}
		
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
	
	public static double[][] similarityFullMatrix(final AbstractRealMatrix data, SimilarityMetric sim) {
		return similarityFullMatrix(data.getData(), sim);
	}
	
	public static double[][] similarityFullMatrix(final double[][] data, SimilarityMetric sim) {
		final int m = data.length;
		
		final double[][] dist_mat = new double[m][m];
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) {
				final double d = sim.getSimilarity(data[i], data[j]);
				dist_mat[i][j] = d;
				dist_mat[j][i] = d;
			}
		}
		
		return dist_mat;
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
		SortedSet<Map.Entry<K,V>> sortEntriesByValue(
				Map<K,V> map, 
				final Comparator<Map.Entry<K,V>> cmp) 
	{
		SortedSet<Map.Entry<K,V>> sortedEntries = new TreeSet<Map.Entry<K,V>>(cmp);
		sortedEntries.addAll(map.entrySet());
		return sortedEntries;
	}
}
