package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.AffinityPropagation;
import com.clust4j.algo.ClustTests;
import com.clust4j.algo.DBSCAN;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.KNN;
import com.clust4j.algo.NearestNeighbors;
import com.clust4j.algo.KNN.KNNPlanner;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.algo.NearestNeighbors.RunMode;
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
		assertTrue(knn1.getLabels()[0] == 0 && knn1.getLabels()[1] == 1);
		
		
		
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
		Kernel kernel = new HyperbolicTangentKernel(); //SplineKernel();
		
		for(int k : ks) {
			new KMedoids(mat, 
				new KMedoids.KMedoidsPlanner(k)
					.setSep(kernel)
					.setVerbose(true)
					.setScale(false)).fit();
		}
		System.out.println();
	}
	
	@Test
	public void DBSCANTest1() {
		final Array2DRowRealMatrix mat = ClustTests.getRandom(1500, 10);
		Kernel kernel = new RadialBasisKernel(0.05);
		DBSCAN db = new DBSCAN(mat, 
				new DBSCAN.DBSCANPlanner(0.05)
					.setSep(kernel)
					.setScale(true)
					.setVerbose(true)).fit();
		System.out.println();
		assertTrue(db.hasWarnings());
	}
	
	@Test
	public void AffinityPropTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		final Random seed = new Random(5);
		final boolean[] b = new boolean[]{true, false};
		
		for(boolean bool: b) {
			AffinityPropagation a = 
					new AffinityPropagation(mat, new AffinityPropagation
						.AffinityPropagationPlanner()
							.addGaussianNoise(bool)
							.setVerbose(true)
							.setSep(new GaussianKernel())
							.setSeed(seed)).fit();
					
					final int[] labels = a.getLabels();
					assertTrue(labels.length == 5);
					assertTrue(labels[0] == labels[1]);
					assertTrue(labels[2] == labels[3]);
					if(bool) assertTrue(a.getNumberOfIdentifiedClusters() == 3);
					assertTrue(a.didConverge());
		}
	}
	
	@Test
	public void AffinityPropLoadTest() {
		final Array2DRowRealMatrix mat = ClustTests.getRandom(1500, 10);
		new AffinityPropagation(mat, new AffinityPropagation
			.AffinityPropagationPlanner()
				.setSep(new GaussianKernel())
				.setVerbose(true)).fit();
	}
	
	@Test
	public void NNTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  2.0,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		NearestNeighbors nn = new NearestNeighbors(mat, 
			new NearestNeighborsPlanner()
				.setVerbose(true)
				.setSep(new GaussianKernel())
				.setK(1)).fit();
		
		ArrayList<Integer>[] ne = nn.getNearest();
		assertTrue(ne[0].size() == 1);
		assertTrue(ne[0].get(0) == 1);
		System.out.println();
		
		nn = new NearestNeighbors(mat, 
			new NearestNeighborsPlanner(RunMode.RADIUS)
				.setVerbose(true)
				.setSep(new GaussianKernel())
				.setRadius(3d)).fit();
	}
	
	@Test
	public void NN_KNEAREST_LoadTest() {
		final Array2DRowRealMatrix mat = ClustTests.getRandom(1500, 10);
		
		final int[] ks = new int[]{1, 5, 10};
		for(int k: ks) {
			new NearestNeighbors(mat, 
				new NearestNeighborsPlanner()
					.setVerbose(true)
					.setSep(new GaussianKernel())
					.setK(k)).fit();
		}
	}
	
	@Test
	public void NN_RADIUS_LoadTest() {
		final Array2DRowRealMatrix mat = ClustTests.getRandom(1500, 10);
		
		final double[] radii = new double[]{0.5, 5.0, 10.0};
		for(double radius: radii) {
			new NearestNeighbors(mat, 
				new NearestNeighborsPlanner(RunMode.RADIUS)
					.setVerbose(true)
					.setSep(new GaussianKernel())
					.setRadius(radius)).fit();
			System.out.println();
		}
	}
}
