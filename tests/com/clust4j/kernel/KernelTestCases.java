package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.KNN;
import com.clust4j.algo.KNN.KNNPlanner;
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
	public void testProjections() {
		final double[] a = new double[]{5,0};
		final double[] b = new double[]{3,0};
		assertTrue(new LinearKernel().distance(a, b) == 15);
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
			new double[] {0.00502, 	 0.0003,    0.08148},
			new double[] {3.01837,   2.2293,    3.94812}
		};
		
		final int[] trainLabels = new int[] {0, 1, 1};
		
		final Array2DRowRealMatrix train = new Array2DRowRealMatrix(train_array);
		final Array2DRowRealMatrix test  = new Array2DRowRealMatrix(test_array);
		
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,2};
		
		KNN knn = null;
		for(boolean b : scale) {
			for(int k : ks) {
				knn = new KNN(train, test, trainLabels, 
						new KNNPlanner(k)
						.setDist(new GaussianKernel())
						.setScale(b)
						.setVerbose(!b));
				knn.train();
				
				final int[] results = knn.getPredictedLabels();
				assertTrue(results[0] == trainLabels[0]);
				assertTrue(results[1] == trainLabels[1]);
			}
		}
		
		// Try with k = 3, labels will be 1 both ways:
		for(boolean b : scale) {
			// Only verbose if scaling just to avoid too many loggings from this one test
			knn = new KNN(train, test, trainLabels, 
					new KNNPlanner(3)
					.setDist(new LinearKernel())
					.setScale(b));
			knn.train();
			
			final int[] results = knn.getPredictedLabels();
			assertTrue(results[0] == trainLabels[1]);
			assertTrue(results[1] == trainLabels[1]);
		}
		
	}
	
	@Test
	public void testLinearSeparability() {
		// Perfectly linearly separable
		final double[][] train_array = new double[][] {
			new double[] {0.0, 	 1.0},
			new double[] {2.0,   3.0},
			new double[] {2.0,   4.0}
		};
		
		final double[][] test_array = new double[][] {
			new double[] {0.0, 	 0.5},
			new double[] {2.0,   3.5}
		};
		
		final int[] trainLabels = new int[] {0, 1, 1};
		
		final Array2DRowRealMatrix train = new Array2DRowRealMatrix(train_array);
		final Array2DRowRealMatrix test  = new Array2DRowRealMatrix(test_array);
		
		// Test with no normalization
		KNN knn1 = new KNN(train, test, trainLabels, 
				new KNNPlanner(2)
					.setDist(new LinearKernel())
					.setVerbose(true));
		knn1.train();
		assertTrue(knn1.getPredictedLabels()[0] == 0 && knn1.getPredictedLabels()[1] == 1);
		
		
	}
}
