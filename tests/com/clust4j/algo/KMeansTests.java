package com.clust4j.algo;

import static com.clust4j.TestSuite.getRandom;
import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.AbstractCentroidClusterer.InitializationStrategy;
import com.clust4j.algo.KMeans.KMeansPlanner;
import com.clust4j.data.DataSet;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.kernel.Kernel;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class KMeansTests implements ClassifierTest, ClusterTest, ConvergeableTest, BaseModelTest {
	final Array2DRowRealMatrix data_ = ExampleDataSets.IRIS.getData();

	@Test
	@Override
	public void testItersElapsed() {
		assertTrue(new KMeans(data_).fit().itersElapsed() > 0);
		assertTrue(new KMeans(data_, 3).fit().itersElapsed() > 0);
		assertTrue(new KMeans(data_, new KMeansPlanner()).fit().itersElapsed() > 0);
		assertTrue(new KMeans(data_, new KMeansPlanner(3)).fit().itersElapsed() > 0);
	}

	@Test
	@Override
	public void testConverged() {
		assertTrue(new KMeans(data_).fit().didConverge());
		assertTrue(new KMeans(data_, 3).fit().didConverge());
		assertTrue(new KMeans(data_, new KMeansPlanner()).fit().didConverge());
		assertTrue(new KMeans(data_, new KMeansPlanner(3)).fit().didConverge());
	}

	@Test
	@Override
	public void testDefConst() {
		new KMeans(data_);
	}

	@Test
	@Override
	public void testArgConst() {
		new KMeans(data_, 3);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new KMeans(data_, new KMeansPlanner());
		new KMeans(data_, new KMeansPlanner(3));
	}

	@Test
	@Override
	public void testFit() {
		new KMeans(data_).fit();
		new KMeans(data_, 3).fit();
		new KMeans(data_, new KMeansPlanner()).fit();
		new KMeans(data_, new KMeansPlanner(3)).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new KMeansPlanner().buildNewModelInstance(data_);
		new KMeansPlanner(3).buildNewModelInstance(data_);
	}

	@Test
	@Override
	public void testScoring() {
		new KMeans(data_, 3).fit().silhouetteScore();
		new KMeans(data_, 5).fit().silhouetteScore();
	}
	
	/** Scale = false */
	@Test
	public void KMeansTest1() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, 2).fit();
		
		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
		//km.info("testing the kmeans logger");
	}
	
	/** Now scale = true */
	@Test
	public void KMeansTest2() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, new KMeans.KMeansPlanner(2).setScale(true)).fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = false and multiclass */
	@Test
	public void KMeansTest3() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, new KMeans.KMeansPlanner(3).setScale(false)).fit();
		
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = true and multiclass */
	@Test
	public void KMeansTest4() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMeans km = new KMeans(mat, new KMeans.KMeansPlanner(3).setScale(true)).fit();
		
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	// What if k = 1??
	@Test
	public void KMeansTest5() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final boolean[] scale = new boolean[]{true, false};
		
		KMeans km = null;
		for(boolean b : scale) {
			final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
			km = new KMeans(mat, new KMeans.KMeansPlanner(1).setScale(b)).fit();
			assertTrue(km.didConverge());

			System.out.println(Arrays.toString(km.getLabels()));
			System.out.println(km.totalCost());
			if(b)
				assertTrue(km.totalCost() == 9.0);
		}
		
		// Test predict function -- no longer part of API
		// assertTrue(km.predictCentroid(new double[]{100d, 201d, 101d}) == km.getLabels()[3]);
	}
	
	// Make sure it won't break on a tie...
	@Test
	public void KMeansTieTest() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		final boolean[] scale = new boolean[]{true, false};
		
		KMeans km = null;
		for(boolean b : scale) {
			final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
			km = new KMeans(mat, new KMeans.KMeansPlanner(2).setScale(b));
			km.fit();
		}
	}
	
	
	
	@Test
	public void KMeansLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(10000, 10);
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,3,5,7};
		
		KMeans km = null;
		for(boolean b : scale) {
			for(int k : ks) {
				km = new KMeans(mat, new KMeans.KMeansPlanner(k).setScale(b));
				km.fit();
			}
		}
	}
	
	
	@Test
	public void KMeansLoadTest2FullLogger() {
		final Array2DRowRealMatrix mat = getRandom(5000, 10);
		KMeans km = new KMeans(mat, new KMeans
				.KMeansPlanner(5)
					.setScale(true)
					.setVerbose(true)
				);
		km.fit();
		System.out.println();
	}
	
	@Test
	public void KernelKMeansLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(5000, 10);
		final int[] ks = new int[] {1,3,5,7};
		Kernel kernel = new GaussianKernel(0.05);
		
		KMeans km = null;
		for(int k : ks) {
			km = new KMeans(mat, new KMeans
					.KMeansPlanner(k)
					.setSep(kernel)
					.setVerbose(true)
					.setScale(false));
			km.fit();
			System.out.println();
		}
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		KMeans km = new KMeans(data_,
			new KMeans.KMeansPlanner(3)
				.setScale(true)
				.setVerbose(true)).fit();
		System.out.println();
		
		final double c = km.totalCost();
		km.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		KMeans km2 = (KMeans)KMeans.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(km2.totalCost() == c);
		assertTrue(km.equals(km2));
		Files.delete(TestSuite.path);
	}
	
	@Test
	public void testLabelCentroidReordering() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
			
		final KMeans km = new KMeans(new Array2DRowRealMatrix(data,false), 3).fit();

		// the labels should correspond to the index of centroid...
		assertTrue(VecUtils.equalsExactly(km.getLabels(), new int[]{ 0,1,2 }));
		final ArrayList<double[]> centroids = km.getCentroids();
		
		for(int i = 0; i < centroids.size(); i++) {
			assertTrue(VecUtils.equalsExactly(data[i], centroids.get(i)));
		}
	}
	
	@Test
	public void assertNaN() {
		assertTrue(Double.isNaN(Double.NaN - 5.0));
		assertFalse(Double.NaN < 5.0);
		assertFalse(Double.NaN > 5.0);
	}
	
	@Test
	public void testingKMAug() {
		double[][] Y = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9}
		};
		
		double[][] X = new double[][]{
			Y[0]
		};
		
		double[][] dists = AbstractCentroidClusterer.eucDists(X, Y);
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{0.0, 27.0, 108.0}
		}));
		
		
		X = new double[][]{ Y[0], Y[1] };
		dists = AbstractCentroidClusterer.eucDists(X, Y);
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{0.0, 27.0, 108.0},
			new double[]{27.0, 0.0, 27.0 }
		}));
		
		X = new double[][]{ Y[0], Y[1], Y[2] };
		dists = AbstractCentroidClusterer.eucDists(X, Y);
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{0.0, 27.0, 108.0},
			new double[]{27.0, 0.0, 27.0 },
			new double[]{108.0, 27.0, 0.0}
		}));
	}
	
	@Test
	public void testCumSumSearchSorted() {
		double[] cumSum = new double[]{
			0.0, 27.0, 135.0, 162.0, 162.0, 189.0, 297.0, 324.0, 324.0
		};
		
		double[] rands = new double[]{16.57957296,   49.95928975,  266.41666906,  265.12261977};
		int[] ss = AbstractCentroidClusterer.searchSortedCumSum(cumSum, rands);
		assertTrue(VecUtils.equalsExactly(ss, new int[]{1,2,6,6}));
	}
	
	@Test
	public void testOnIris() {
		DataSet iris = ExampleDataSets.IRIS.shuffle();
		Array2DRowRealMatrix data = iris.getData();
		int[] labels = iris.getLabels();
		
		InitializationStrategy[] strats = InitializationStrategy.values();
		
		for(InitializationStrategy s: strats) {
			KMeans model = new KMeans(data, new KMeansPlanner(3)
				.setInitializationStrategy(s)
				.setScale(true)
				.setVerbose(true)).fit();
			
			System.out.println("Silhouette score: " + model.silhouetteScore());
			System.out.println("Idx affinity score: " + model.indexAffinityScore(labels));
			System.out.println();
		}
		
	}
}
