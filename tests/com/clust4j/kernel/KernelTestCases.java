package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.KernelKNN;
import com.clust4j.utils.VecUtils;

public class KernelTestCases {
	final static Random rand = new Random();
	
	private static double[] randomVector(int length) {
		final double[] a = new double[length];
		for(int i = 0; i < a.length; i++)
			a[i] = rand.nextDouble();
		return a;
	}

	@Test
	public void testSmall() {
		final double[] a = new double[]{0,1};
		final double[] b = new double[]{1,0};
		
		// Perfectly orthogonal
		assertTrue(new LinearKernel().distance(a, b) == 0);
		assertTrue(VecUtils.isOrthogonalTo(a, b));
	}

	@Test
	public void testBigger() {
		final double[] a = randomVector(10);
		final double[] b = randomVector(10);
		System.out.println(new LinearKernel().distance(a, b));
	}
	
	@Test
	public void testGaussianKernel() {
		final double[] a = new double[]{0, 1, 2, 3};
		final double[] b = new double[]{1, 0,-1,-2};
		System.out.println(new GaussianKernel(2).distance(a, b));
	}

	
	@Test
	public void KernelKNNTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.00504, 	 0.0001,    0.08172},
			new double[] {3.65816,   2.9471,    3.12331},
			new double[] {4.12344,   3.0001,    2.89002}
		};
		
		final double[][] test_array = new double[][] {
			new double[] {0.01302, 	 0.0012,    0.06948},
			new double[] {3.01837,   2.2293,    3.94812}
		};
		
		final int[] trainLabels = new int[] {0, 1, 1};
		
		final Array2DRowRealMatrix train = new Array2DRowRealMatrix(train_array);
		final Array2DRowRealMatrix test  = new Array2DRowRealMatrix(test_array);
		
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,2};
		
		KernelKNN knn = null;
		for(boolean b : scale) {
			for(int k : ks) {
				knn = new KernelKNN(train, test, trainLabels, 
						new KernelKNN.KernelKNNPlanner(k, 
								new LinearKernel())
						.setScale(b)
						.setVerbose(!b));
				knn.train();
				
				final int[] results = knn.getPredictedLabels();
				//assertTrue(results[0] == trainLabels[0]);
				//assertTrue(results[1] == trainLabels[1]);
			}
		}
		
		// Try with k = 3, labels will be 1 both ways:
		for(boolean b : scale) {
			// Only verbose if scaling just to avoid too many loggings from this one test
			knn = new KernelKNN(train, test, trainLabels, 
					new KernelKNN.KernelKNNPlanner(3, 
							new LinearKernel())
					.setScale(b));
			knn.train();
			
			final int[] results = knn.getPredictedLabels();
			assertTrue(results[0] == trainLabels[1]);
			assertTrue(results[1] == trainLabels[1]);
		}
		

		//knn.info("testing the KNN logger");
	}
}
