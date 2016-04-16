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
package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.metrics.pairwise.Pairwise;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.VecUtils;

public class KernelTestCases {
	
	final static MatrixFormatter formatter = new MatrixFormatter();
	final static Random rand = new Random();
	public final static Kernel[] all_kernels = new Kernel[]{
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
	
	public static double[] randomVector(int length) {
		final double[] a = new double[length];
		for(int i = 0; i < a.length; i++)
			a[i] = rand.nextDouble();
		return a;
	}
	
	public static void print(final String s) {
		System.out.println(s);
	}

	@Test
	public void testSmall() {
		final double[] a = new double[]{0,1};
		final double[] b = new double[]{1,0};
		
		// Perfectly orthogonal
		assertTrue(new LinearKernel().getSimilarity(a, b) == new LinearKernel().getConstant());
		assertTrue(VecUtils.isOrthogonalTo(a, b));
	}
	
	@Test
	public void testProjections() {
		final double[] a = new double[]{5,0};
		final double[] b = new double[]{3,0};
		assertTrue(new LinearKernel().getSimilarity(a, b) == 15 + new LinearKernel().getConstant());
	}
	
	@Test
	public void testLinearSeparability() {
		// Perfectly linearly separable
		final double[][] train_array = new double[][] {
			new double[] {0.0, 	 1.0},
			new double[] {2.0,   3.0},
			new double[] {2.0,   4.0}
		};
		
		final Array2DRowRealMatrix train = new Array2DRowRealMatrix(train_array);
		final double sigma = 0.05;
		double[][] simMat;

		
		// Look at the kernel matrices...
		Kernel kernel = new LinearKernel();
		simMat = Pairwise.getSimilarity(train_array, kernel, false, false);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{1,  3,  4},
			new double[]{3, 13, 16},
			new double[]{4, 16, 20}
		}));

		
		kernel = new LaplacianKernel(sigma);
		simMat = Pairwise.getSimilarity(train_array, kernel, false, false);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{1.0, 0.8681234453945849, 0.8350384028320718},
			new double[]{0.8681234453945849, 1.0, 0.951229424500714 },
			new double[]{0.8350384028320718, 0.951229424500714 , 1.0}
		}));
		
		
		kernel = new ANOVAKernel(sigma, 1);
		simMat = Pairwise.getSimilarity(train_array, kernel, false, false);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{2.0, 1.6374615061559636, 1.4563589046997552},
			new double[]{1.6374615061559636, 2.0, 1.951229424500714 },
			new double[]{1.4563589046997552, 1.951229424500714, 2.0 }
		}));
		
		
		kernel = new SplineKernel();
		simMat = Pairwise.getSimilarity(train_array, kernel, false, false);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{2.3333333333333335, 5.333333333333333,  6.833333333333333  },
			new double[]{5.333333333333333 , 145.66666666666666, 203.16666666666666 },
			new double[]{6.833333333333333 , 203.16666666666666, 293.88888888888886 }
		}));
		
		
		kernel = new PolynomialKernel();
		simMat = Pairwise.getSimilarity(train_array, kernel, false, false);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{2,  4,  5},
			new double[]{4, 14, 17},
			new double[]{5, 17, 21}
		}));
		
		
		kernel = new HyperbolicTangentKernel();
		simMat = Pairwise.getSimilarity(train_array, kernel, false, false);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{0.9640275800758169, 0.999329299739067,  0.9999092042625951 },
			new double[]{0.999329299739067 , 0.9999999999986171, 0.9999999999999966 },
			new double[]{0.9999092042625951 , 0.9999999999999966, 1.0 }
		}));
		
		kernel = new RadialBasisKernel(sigma);
		simMat = Pairwise.getSimilarity(train_array, kernel, false, false);
		assertTrue(MatUtils.equalsExactly(simMat, new double[][]{
			new double[]{1.0, 0.6703200460356393, 0.522045776761016 },
			new double[]{0.6703200460356393, 1.0, 0.951229424500714 },
			new double[]{0.522045776761016, 0.951229424500714, 1.0  }
		}));
		
		
		// Test with KMEANS
		new KMeansParameters(2)
			.setMetric(kernel)
			.setVerbose(true)
			.fitNewModel(train);
		System.out.println();
	}
	
	@Test
	public void testLaplacianAndRBFPartialSim() {
		// we want to test that the partial similarity maintains ordinality
		double[] partialSimilarities = new double[]{
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
		};
		
		Kernel[] kernels = new Kernel[]{
			new GaussianKernel(), 
			new RadialBasisKernel(), 
			new CauchyKernel(), 
			new ANOVAKernel(), 
			new CircularKernel(), 
			new SphericalKernel()
		};
		
		for(Kernel kernel: kernels) {
			double[] fullSimilarities = new double[partialSimilarities.length];
			for(int i = 0; i < fullSimilarities.length; i++)
				fullSimilarities[i] = kernel.partialSimilarityToSimilarity(partialSimilarities[i]);
			System.out.println(Arrays.toString(fullSimilarities));
			for(int i = 0; i < fullSimilarities.length - 1; i++)
				assertTrue(fullSimilarities[i+1] > fullSimilarities[i]);
		}
		
		// inverse the partials, make them negative. They should now DECREASE
		partialSimilarities = VecUtils.negative(partialSimilarities);
		for(Kernel kernel: kernels) {
			double[] fullSimilarities = new double[partialSimilarities.length];
			for(int i = 0; i < fullSimilarities.length; i++)
				fullSimilarities[i] = kernel.partialSimilarityToSimilarity(partialSimilarities[i]);
			System.out.println(Arrays.toString(fullSimilarities));
			for(int i = 0; i < fullSimilarities.length - 1; i++)
				assertTrue(fullSimilarities[i+1] < fullSimilarities[i]);
			
			// Test partial to full for sim and dist
			for(int i = 0; i < fullSimilarities.length; i++) {
				assertTrue(
					Precision.equals(
						kernel.distanceToPartialDistance( -fullSimilarities[i] ), 
						-partialSimilarities[i],
						1e-8));
				
				assertTrue(
					Precision.equals(
						kernel.partialDistanceToDistance( -partialSimilarities[i] ), 
						-fullSimilarities[i],
						1e-8));
				
				assertTrue(
					Precision.equals(
						kernel.similarityToPartialSimilarity( fullSimilarities[i] ), 
						partialSimilarities[i],
						1e-8));
			}
		}
	}
	
	@Test
	public void testForCoverage() {
		double[] a = new double[]{1,2,3,4,5};
		
		for(Kernel k: all_kernels) {
			assertNotNull(k.getName());
			k.getSimilarity(a, a); // ensure throws no exception with default constructor
			
			// Ensure all partial to full and vice versa work
			assertTrue(Precision.equals(
				k.partialSimilarityToSimilarity(k.getPartialSimilarity(a, a)), 
				k.getSimilarity(a, a), 
				1e-8));
			
			assertTrue(Precision.equals(
				k.similarityToPartialSimilarity(k.getSimilarity(a, a)), 
				k.getPartialSimilarity(a, a), 
				1e-8));
			
			assertTrue(Precision.equals(
				k.getDistance(a, a), 
				k.partialDistanceToDistance(k.getPartialDistance(a, a)), 
				1e-8));
		}
	}
	
	@Test
	public void testMatForDataSets() {
		String[] sets = new String[]{"iris","wine","breastcancer"};
		
		String set;
		int idx = 0;
		for(Array2DRowRealMatrix data: new Array2DRowRealMatrix[]
				{
					TestSuite.IRIS_DATASET.getData(),
					TestSuite.WINE_DATASET.getData(),
					TestSuite.BC_DATASET.getData()
				}) {
			set = sets[idx];
			double[][] mat;
			
			for(Kernel k: all_kernels) {
				for(boolean partial: new boolean[]{true, false}) {
					mat = Pairwise.getSimilarity(data, k, false, partial);
					if(MatUtils.containsNaN(mat)) {
						System.out.println(k + " yields NaN in "+set+", partial=" + partial);
					}
					
					if(MatUtils.containsInf(mat)) {
						System.out.println(k + " yields Inf in "+set+", partial=" + partial);
					}
				}
			}
			
			idx++;
		}
	}
	
	@Test
	public void testOther() {
		/*
		 * hilbert space will equal 2*115 - 129 - 174 = -73
		 */
		double[] a = new double[]{10,2,3,4}; // ip = 129
		double[] b = new double[]{5,6,7,8}; // ip = 174
		assertTrue(Kernel.toHilbertPSpace(a, b) == -73);
		Kernel c,d,e;
		
		/*
		 * Ensure the sigma param is working for Cauchy
		 */
		c = new CauchyKernel();
		d = new CauchyKernel(5.0);
		assertFalse(c.getSimilarity(a, b) == d.getSimilarity(a, b));
		
		/*
		 * Assert circular kernel 0
		 */
		c = new CircularKernel(-200.0);
		assertTrue(c.getSimilarity(a, b) == 0.0); // greater than sigma
		
		/*
		 * Ensure alpha, beta are used in GeneralizedMinKernel
		 */
		c = new GeneralizedMinKernel(-2.0, -3.0);
		d = new GeneralizedMinKernel();
		assertFalse(c.getSimilarity(a, b) == d.getSimilarity(a, b));
		
		/*
		 * test constant on inverse multi
		 */
		c = new InverseMultiquadricKernel();
		d = new InverseMultiquadricKernel(5.0);
		assertFalse(c.getSimilarity(a, b) == d.getSimilarity(a, b));
		
		/*
		 * Test laplacian cases
		 */
		c = new LaplacianKernel(2.0);
		d = new LaplacianKernel(2.0, 2.0);
		e = new LaplacianKernel(2.0, 2.0, 2.0);
		assertFalse(c.getSimilarity(a, b) == d.getSimilarity(a, b));
		assertFalse(c.getSimilarity(a, b) == e.getSimilarity(a, b));
		assertFalse(e.getSimilarity(a, b) == d.getSimilarity(a, b));
		
		/*
		 * Test log cases
		 */
		c = new LogKernel();
		d = new LogKernel(1000);
		double f = c.getSimilarity(a, b), g = d.getSimilarity(a, b);
		assertTrue((Double.isInfinite(f) && Double.isInfinite(g)) || (f != g));
		
		/*
		 * Test polynomial cases
		 */
		c = new PolynomialKernel();
		d = new PolynomialKernel(5.0,2.0);
		assertFalse(c.getSimilarity(a, b) == d.getSimilarity(a, b));
		
		/*
		 * Test spherical cases
		 */
		c = new SphericalKernel(-200.0);
		assertTrue(c.getSimilarity(a, b) == 0.0); // greater than sigma
		d = new SphericalKernel(1000.0);
		assertFalse(c.getSimilarity(a, b) == d.getSimilarity(a, b));
	}
}
