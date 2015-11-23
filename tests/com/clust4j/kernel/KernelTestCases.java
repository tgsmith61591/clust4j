package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.ClustTests;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.KNN;
import com.clust4j.algo.KNN.KNNPlanner;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.VecUtils;

public class KernelTestCases {
	final static MatrixFormatter formatter = new MatrixFormatter();
	final static Random rand = new Random();
	
	public static double[] randomVector(int length) {
		final double[] a = new double[length];
		for(int i = 0; i < a.length; i++)
			a[i] = rand.nextDouble();
		return a;
	}
	
	public static String formatKernelMatrix(final double[][] data, final Kernel kernel) {
		return formatter.format(new Array2DRowRealMatrix(kernel.kernelSimilarityMatrix(data), false));
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
						.setSep(new GaussianKernel())
						.setScale(b)
						.setVerbose(!b));
				knn.fit();
				
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
					.setSep(new LinearKernel())
					.setScale(b));
			knn.fit();
			
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
		
		// Look at the kernel matrix...
		Kernel kernel = new LinearKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 3.0);

		final double sigma = 0.05;
		
		kernel = new LaplacianKernel(sigma);
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 0.8681234453945849);
		
		kernel = new ANOVAKernel(sigma, 1);
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 1.6374615061559636);
		
		kernel = new SplineKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 5.333333333333333);
		
		kernel = new PolynomialKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 4.0);
		
		kernel = new HyperbolicTangentKernel();
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 0.999329299739067);
		
		kernel = new RadialBasisKernel(sigma);
		assertTrue(kernel.kernelSimilarityMatrix(train_array)[0][1] == 0.6703200460356393);
		
		
		// Test with no normalization
		KNN knn1 = new KNN(train, test, trainLabels, 
				new KNNPlanner(2)
					.setSep(kernel)
					.setVerbose(true));
		knn1.fit();
		assertTrue(knn1.getPredictedLabels()[0] == 0 && knn1.getPredictedLabels()[1] == 1);
		
		
		
		// Test with KMEANS
		KMeans km = new KMeans(train, 
				new KMeans.BaseKCentroidPlanner(2)
					.setSep(kernel)
					.setVerbose(true)
				);
		km.fit();
		System.out.println();
	}
	
	@Test
	public void KernelKMeansLoadTest1() {
		final Array2DRowRealMatrix mat = ClustTests.getRandom(5000, 10);
		final int[] ks = new int[] {1,3,5,7};
		Kernel kernel = new GaussianKernel(0.05);
		
		KMeans km = null;
		for(int k : ks) {
			km = new KMeans(mat, new KMeans
					.BaseKCentroidPlanner(k)
					.setSep(kernel)
					.setVerbose(true)
					.setScale(false));
			km.fit();
		}
		System.out.println();
	}
	
	@Test
	public void KernelKMedoidsLoadTest1() {
		final Array2DRowRealMatrix mat = ClustTests.getRandom(1000, 10);
		final int[] ks = new int[] {1,3,5,7};
		Kernel kernel = new LaplacianKernel(0.05);
		
		KMedoids km = null;
		for(int k : ks) {
			km = new KMedoids(mat, 
					new KMedoids.KMedoidsPlanner(k)
						.setSep(kernel)
						.setVerbose(true)
						.setScale(false));
			km.fit();
		}
		System.out.println();
	}
	
	@Test
	public void KernelKMedoidsLoadTest2() {
		final Array2DRowRealMatrix mat = ClustTests.getRandom(2000, 10);
		final int[] ks = new int[] {12};
		Kernel kernel = new SplineKernel();
		
		KMedoids km = null;
		for(int k : ks) {
			km = new KMedoids(mat, 
					new KMedoids.KMedoidsPlanner(k)
						.setSep(kernel)
						.setVerbose(true)
						.setScale(false));
			km.fit();
		}
		System.out.println();
	}
}
