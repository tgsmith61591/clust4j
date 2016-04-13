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

import static org.junit.Assert.*;

import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.kernel.ANOVAKernel;
import com.clust4j.kernel.CauchyKernel;
import com.clust4j.kernel.CircularKernel;
import com.clust4j.kernel.ExponentialKernel;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.kernel.GeneralizedMinKernel;
import com.clust4j.kernel.HyperbolicTangentKernel;
import com.clust4j.kernel.InverseMultiquadricKernel;
import com.clust4j.kernel.Kernel;
import com.clust4j.kernel.LaplacianKernel;
import com.clust4j.kernel.LinearKernel;
import com.clust4j.kernel.LogKernel;
import com.clust4j.kernel.MinKernel;
import com.clust4j.kernel.MultiquadricKernel;
import com.clust4j.kernel.PolynomialKernel;
import com.clust4j.kernel.PowerKernel;
import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.kernel.RationalQuadraticKernel;
import com.clust4j.kernel.SphericalKernel;
import com.clust4j.kernel.SplineKernel;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class PairwiseTests {
	
	final static double[][] X = MatUtils.reshape(VecUtils.asDouble(VecUtils.arange(9)), 3, 3);
	final static double[][] Xh = MatUtils.reshape(VecUtils.asDouble(VecUtils.arange(24)), 12, 2); // for haversine
	
	final static Kernel[] kernels = new Kernel[]{
			new ANOVAKernel(), 
			new CauchyKernel(), 
			new CircularKernel(), 
			new ExponentialKernel(),
			new GaussianKernel(), 
			new GeneralizedMinKernel(),
			new HyperbolicTangentKernel(),
			new InverseMultiquadricKernel(),
			new LaplacianKernel(),
			new LinearKernel(),
			new LogKernel(),
			new MinKernel(),
			new MultiquadricKernel(),
			new PolynomialKernel(),
			new PowerKernel(),
			new RadialBasisKernel(), 
			new RationalQuadraticKernel(),
			new SphericalKernel(),
			new SplineKernel()
		};
	
	static DistanceMetric[] distances() {
		DistanceMetric[] d = new DistanceMetric[Distance.values().length + 1];
		for(int i = 0; i < d.length - 1; i++)
			d[i] = Distance.values()[i];
		d[d.length-1] = new MinkowskiDistance(1.5);
		return d;
	}
	
	static SimilarityMetric[] similarities() {
		SimilarityMetric[] d = new SimilarityMetric[kernels.length + 1];
		for(int i = 0; i < d.length - 1; i++)
			d[i] = kernels[i];
		d[d.length-1] = Similarity.COSINE;
		return d;
	}
	
	@Test
	public void testDistUpperTriangular() {
		/*
		 * Test for ALL distance metrics
		 */
		for(DistanceMetric metric: distances()) {
			double[][] completeDistance = Pairwise.getDistance(X, metric, true, false);
			double[][] partialDistance  = Pairwise.getDistance(X, metric, true, true);
			
			for(int i = 0; i < X.length - 1; i++) {
				for(int j = i + 1; j < X.length; j++) {
					/*
					 * Assert partial to full == complete
					 */
					assertTrue(Precision.equals(
						completeDistance[i][j], 
						metric.partialDistanceToDistance(partialDistance[i][j]),
						1e-8));
					
					/*
					 * Assert complete to partial == partial
					 */
					assertTrue(Precision.equals(
						partialDistance[i][j], 
						metric.distanceToPartialDistance(completeDistance[i][j]),
						1e-8));
					
					/*
					 * Assert lower-triangular portion is all zero
					 */
					assertTrue(partialDistance[j][i] == 0.0);
					assertTrue(completeDistance[j][i]== 0.0);
				}
			}
			
			// Assert the diagonal is entirely zero
			for(int i = 0; i < X.length; i++) {
				assertTrue(partialDistance[i][i] == 0.0);
				assertTrue(completeDistance[i][i]== 0.0);
			}
		}	
	}

	
	
	@Test
	public void testDistFull() {
		/*
		 * Test for ALL distance metrics
		 */
		for(DistanceMetric metric: distances()) {
			double[][] completeDistance = Pairwise.getDistance(X, metric, false, false);
			double[][] partialDistance  = Pairwise.getDistance(X, metric, false, true);
			
			for(int i = 0; i < X.length - 1; i++) {
				for(int j = i + 1; j < X.length; j++) {
					/*
					 * Assert partial to full == complete
					 */
					assertTrue(Precision.equals(
						completeDistance[i][j], 
						metric.partialDistanceToDistance(partialDistance[i][j]),
						1e-8));
					
					/*
					 * Assert complete to partial == partial
					 */
					assertTrue(Precision.equals(
						partialDistance[i][j], 
						metric.distanceToPartialDistance(completeDistance[i][j]),
						1e-8));
					
					/*
					 * Assert lower-triangular portion equals upper triangular portion reflected
					 */
					assertTrue(partialDistance[j][i]  == partialDistance[i][j] );
					assertTrue(completeDistance[j][i] == completeDistance[i][j]);
				}
			}
		}
	}
	
	@Test
	public void testSimUpperTriangular() {
		/*
		 * Test for ALL kernels
		 */
		for(SimilarityMetric kernel: similarities()) {
			double[][] completeSimil = Pairwise.getSimilarity(X, kernel, true, false);
			double[][] partialSimil  = Pairwise.getSimilarity(X, kernel, true, true);
			
			for(int i = 0; i < X.length - 1; i++) {
				for(int j = i + 1; j < X.length; j++) {
					/*
					 * Assert partial to full == complete
					 */
					assertTrue(Precision.equals(
						completeSimil[i][j], 
						kernel.partialSimilarityToSimilarity(partialSimil[i][j]),
						1e-8));
					
					/*
					 * Assert complete to partial == partial
					 */
					assertTrue(Precision.equals(
						partialSimil[i][j], 
						kernel.similarityToPartialSimilarity(completeSimil[i][j]),
						1e-8));
					
					/*
					 * DISTS:
					 * Assert partial to full == complete
					 */
					assertTrue(Precision.equals(
						-completeSimil[i][j], 
						kernel.partialDistanceToDistance(-partialSimil[i][j]),
						1e-8));
					
					/*
					 * DIST:
					 * Assert complete to partial == partial
					 */
					assertTrue(Precision.equals(
						-partialSimil[i][j], 
						kernel.distanceToPartialDistance(-completeSimil[i][j]),
						1e-8));
					
					/*
					 * Assert lower-triangular portion is all zero
					 */
					assertTrue(partialSimil[j][i] == 0.0);
					assertTrue(completeSimil[j][i]== 0.0);
				}
			}
			
			// Assert the diagonal is entirely zero
			for(int i = 0; i < X.length; i++) {
				assertTrue(partialSimil[i][i] == 0.0);
				assertTrue(completeSimil[i][i]== 0.0);
			}
		}	
	}

	
	
	@Test
	public void testSimFull() {
		/*
		 * Test for ALL kernels
		 */
		for(SimilarityMetric kernel: similarities()) {
			double[][] completeSimil = Pairwise.getSimilarity(X, kernel, false, false);
			double[][] partialSimil  = Pairwise.getSimilarity(X, kernel, false, true);
			
			for(int i = 0; i < X.length - 1; i++) {
				for(int j = i + 1; j < X.length; j++) {
					/*
					System.out.println(
						kernel.getName() + ", " + 
						completeSimil[i][j] + ", " + 
						kernel.partialSimilarityToSimilarity(partialSimil[i][j]));
					*/
					
					/*
					 * Assert partial to full == complete
					 */
					assertTrue(Precision.equals(
						completeSimil[i][j], 
						kernel.partialSimilarityToSimilarity(partialSimil[i][j]),
						1e-8));
					
					/*
					 * Assert complete to partial == partial
					 */
					assertTrue(Precision.equals(
						partialSimil[i][j], 
						kernel.similarityToPartialSimilarity(completeSimil[i][j]),
						1e-8));
					
					/*
					 * DISTS:
					 * Assert partial to full == complete
					 */
					assertTrue(Precision.equals(
						-completeSimil[i][j], 
						kernel.partialDistanceToDistance(-partialSimil[i][j]),
						1e-8));
					
					/*
					 * DIST:
					 * Assert complete to partial == partial
					 */
					assertTrue(Precision.equals(
						-partialSimil[i][j], 
						kernel.distanceToPartialDistance(-completeSimil[i][j]),
						1e-8));
					
					/*
					 * Assert lower-triangular portion equals upper triangular portion reflected
					 */
					assertTrue(partialSimil[j][i]  == partialSimil[i][j] );
					assertTrue(completeSimil[j][i] == completeSimil[i][j]);
				}
			}
		}	
	}
	
	@Test
	public void testHaverDistUpperTriangular() {
		/*
		 * Test for Haversine
		 */
		DistanceMetric metric = Distance.HAVERSINE.MI;
		double[][] completeDistance = Pairwise.getDistance(Xh, metric, true, false);
		double[][] partialDistance  = Pairwise.getDistance(Xh, metric, true, true);
		
		for(int i = 0; i < Xh.length - 1; i++) {
			for(int j = i + 1; j < Xh.length; j++) {
				/*
				 * Assert partial to full == complete
				 */
				assertTrue(Precision.equals(
					completeDistance[i][j], 
					metric.partialDistanceToDistance(partialDistance[i][j]),
					1e-8));
				
				/*
				 * Assert complete to partial == partial
				 */
				assertTrue(Precision.equals(
					partialDistance[i][j], 
					metric.distanceToPartialDistance(completeDistance[i][j]),
					1e-8));
				
				/*
				 * Assert lower-triangular portion is all zero
				 */
				assertTrue(partialDistance[j][i] == 0.0);
				assertTrue(completeDistance[j][i]== 0.0);
			}
		}
		
		// Assert the diagonal is entirely zero
		for(int i = 0; i < X.length; i++) {
			assertTrue(partialDistance[i][i] == 0.0);
			assertTrue(completeDistance[i][i]== 0.0);
		}
	}

	
	
	@Test
	public void testHaverDistFull() {
		/*
		 * Test for Haversine
		 */
		DistanceMetric metric = Distance.HAVERSINE.MI;
		double[][] completeDistance = Pairwise.getDistance(Xh, metric, false, false);
		double[][] partialDistance  = Pairwise.getDistance(Xh, metric, false, true);
			
		for(int i = 0; i < Xh.length - 1; i++) {
			for(int j = i + 1; j < Xh.length; j++) {
				/*
				 * Assert partial to full == complete
				 */
				assertTrue(Precision.equals(
					completeDistance[i][j], 
					metric.partialDistanceToDistance(partialDistance[i][j]),
					1e-8));
				
				/*
				 * Assert complete to partial == partial
				 */
				assertTrue(Precision.equals(
					partialDistance[i][j], 
					metric.distanceToPartialDistance(completeDistance[i][j]),
					1e-8));
				
				/*
				 * Assert lower-triangular portion equals upper triangular portion reflected
				 */
				assertTrue(partialDistance[j][i]  == partialDistance[i][j] );
				assertTrue(completeDistance[j][i] == completeDistance[i][j]);
			}
		}
	}
	
	@Test
	public void testBadMinkowski() {
		boolean a = false;
		try {
			Distance.MINKOWSKI(0.5);
		} catch(IllegalArgumentException iae) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testCosinePartial() {
		final double[] d = new double[]{1,2,3,4,5};
		assertTrue(Similarity.COSINE.getPartialSimilarity(d, d) == Similarity.COSINE.getSimilarity(d, d));
	}
}
