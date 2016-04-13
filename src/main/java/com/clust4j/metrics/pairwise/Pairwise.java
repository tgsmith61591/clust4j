/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.metrics.pairwise;

import org.apache.commons.math3.linear.AbstractRealMatrix;

public abstract class Pairwise {
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
