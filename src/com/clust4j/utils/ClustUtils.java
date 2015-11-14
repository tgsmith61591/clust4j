package com.clust4j.utils;

import java.util.Comparator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public class ClustUtils {
	final public static double[][] distanceMatrix(final AbstractRealMatrix data, GeometricallySeparable dist) {
		final int m = data.getRowDimension();
		
		// Compute distance matrix, which is O(N^2) space, O(Nc2) time
		// We do this in KMedoids and not KMeans, because KMedoids uses
		// real points as medoids and not means for centroids, thus
		// the recomputation of distances is unnecessary with the dist mat
		final double[][] dist_mat = new double[m][m];
		for(int i = 0; i < m - 1; i++)
			for(int j = i + 1; j < m; j++)
				dist_mat[i][j] = dist.distance(data.getRow(i), data.getRow(j));
		
		return dist_mat;
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
