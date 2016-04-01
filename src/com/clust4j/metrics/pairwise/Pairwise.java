package com.clust4j.metrics.pairwise;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public class Pairwise {
	public static double[][] getDistance(AbstractRealMatrix a,
			GeometricallySeparable geo,
			boolean upperTriang, boolean partial) {
		return getDistance(a.getData(), geo, upperTriang, partial);
	}
	
	public static double[][] getDistance(double[][] a, 
			GeometricallySeparable geo, 
			boolean upperTriang, boolean partial) {
		
		return pairwise(a, geo, upperTriang, partial, 1.0);
	}
	
	public static double[][] getSimilarity(AbstractRealMatrix a,
			GeometricallySeparable geo,
			boolean upperTriang, boolean partial) {
		return getSimilarity(a.getData(), geo, upperTriang, partial);
	}
	
	public static double[][] getSimilarity(double[][] a, 
			GeometricallySeparable geo, 
			boolean upperTriang, boolean partial) {

		return pairwise(a, geo, upperTriang, partial, -1.0);
	}
	
	private static double[][] pairwise(double[][] a,
			GeometricallySeparable geo,
			boolean upper, boolean partial, double scalar) {
		
		/*
		 * Don't need to check dims, because that happens in each
		 * getDistance call. Any non-uniformity should be handled 
		 * there.
		 */
		
		final int m = a.length;
		final double[][] out = new double[m][m];
		double dist;
		
		/*
		 * First loop: O(M choose 2). Do computations
		 */
		for(int i = 0; i < m - 1; i++) {
			for(int j = i + 1; j < m; j++) {
				
				dist = scalar * (partial ? 
					geo.getPartialDistance(a[i], a[j]) : 
						geo.getDistance(a[i], a[j]));
				out[i][j] = dist;
				
				// We want the full matrix
				if(!upper) {
					out[j][i] = dist;
				}
			}
		}
		
		/*
		 *  If we want the full matrix, we need to compute the diagonal...
		 *  O(M) -- just the diagonal elements
		 */
		if(!upper) {
			for(int i = 0; i < m; i++) {
				out[i][i] = scalar * (partial ? 
					geo.getPartialDistance(a[i], a[i]) : 
						geo.getDistance(a[i], a[i]));
			}
		}
		
		return out;
	}
}
