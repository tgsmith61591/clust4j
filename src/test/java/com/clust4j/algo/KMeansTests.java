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
package com.clust4j.algo;

import static com.clust4j.TestSuite.getRandom;
import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.TestSuite;
import com.clust4j.algo.AbstractCentroidClusterer.InitializationStrategy;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.data.DataSet;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.except.NaNException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.kernel.Kernel;
//import com.clust4j.kernel.KernelTestCases;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class KMeansTests implements ClassifierTest, ClusterTest, ConvergeableTest, BaseModelTest {
	final Array2DRowRealMatrix data_ = TestSuite.IRIS_DATASET.getData();
	final Array2DRowRealMatrix wine = TestSuite.WINE_DATASET.getData();
	final Array2DRowRealMatrix bc = TestSuite.BC_DATASET.getData();

	@Test
	@Override
	public void testItersElapsed() {
		assertTrue(new KMeans(data_).fit().itersElapsed() > 0);
		assertTrue(new KMeans(data_, 3).fit().itersElapsed() > 0);
		assertTrue(new KMeans(data_, new KMeansParameters()).fit().itersElapsed() > 0);
		assertTrue(new KMeans(data_, new KMeansParameters(3)).fit().itersElapsed() > 0);
	}

	@Test
	@Override
	public void testConverged() {
		assertTrue(new KMeans(data_).fit().didConverge());
		assertTrue(new KMeans(data_, 3).fit().didConverge());
		assertTrue(new KMeans(data_, new KMeansParameters()).fit().didConverge());
		assertTrue(new KMeans(data_, new KMeansParameters(3)).fit().didConverge());
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
		new KMeans(data_, new KMeansParameters());
		new KMeans(data_, new KMeansParameters(3));
	}

	@Test
	@Override
	public void testFit() {
		new KMeans(data_).fit();
		new KMeans(data_, 3).fit();
		new KMeans(data_, new KMeansParameters()).fit();
		new KMeans(data_, new KMeansParameters(3)).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new KMeansParameters().fitNewModel(data_);
		new KMeansParameters(3).fitNewModel(data_);
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
		StandardScaler scaler = new StandardScaler().fit(mat);
		KMeans km = new KMeans(scaler.transform(mat), new KMeansParameters(2)).fit();

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
		KMeans km = new KMeans(mat, new KMeansParameters(3)).fit();
		
		assertTrue(km.getK() == 3);
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
		StandardScaler scaler = new StandardScaler().fit(mat);
		KMeans km = new KMeans(scaler.transform(mat), new KMeansParameters(3)).fit();
		
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
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		StandardScaler scaler = new StandardScaler().fit(mat);
		final RealMatrix X = scaler.transform(mat);
		final boolean[] scale = new boolean[]{true, false};
		
		KMeans km = null;
		for(boolean b : scale) {
			km = new KMeans(b ? X : mat, new KMeansParameters(1)).fit();
			assertTrue(km.didConverge());

			System.out.println(Arrays.toString(km.getLabels()));
			System.out.println(km.getTSS());
			if(b)
				assertTrue(km.getTSS() == 9.0);
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
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		StandardScaler scaler = new StandardScaler().fit(mat);
		final RealMatrix X = scaler.transform(mat);
		final boolean[] scale = new boolean[]{true, false};
		
		KMeans km = null;
		for(boolean b : scale) {
			km = new KMeans(b ? X : mat, new KMeansParameters(2));
			km.fit();
		}
	}
	
	
	
	@Test
	public void KMeansLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		StandardScaler scaler = new StandardScaler().fit(mat);
		RealMatrix X = scaler.transform(mat);
		
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,3,5,7};
		
		KMeans km = null;
		for(boolean b : scale) {
			for(int k : ks) {
				km = new KMeans(b ? X : mat, new KMeansParameters(k));
				km.fit();
			}
		}
	}
	
	
	@Test
	public void KMeansLoadTest2FullLogger() {
		final Array2DRowRealMatrix mat = getRandom(500, 10); // need to reduce size for travis CI
		KMeans km = new KMeans(mat, new KMeansParameters(5)
					.setVerbose(true)
				);
		km.fit();
		System.out.println();
	}
	
	@Test
	public void KernelKMeansLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(500, 10); // need to reduce size for travis CI
		final int[] ks = new int[] {1,3,5,7};
		Kernel kernel = new GaussianKernel(0.05);
		
		KMeans km = null;
		for(int k : ks) {
			km = new KMeans(mat, new KMeansParameters(k)
					.setMetric(kernel)
					.setVerbose(true));
			km.fit();
			System.out.println();
		}
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		KMeans km = new KMeans(data_,
			new KMeansParameters(3)
				.setVerbose(true)).fit();
		System.out.println();
		
		final double c = km.getTSS();
		km.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		KMeans km2 = (KMeans)KMeans.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(km2.getTSS() == c);
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
		DataSet iris = TestSuite.IRIS_DATASET.shuffle();
		Array2DRowRealMatrix data = iris.getData();
		int[] labels = iris.getLabels();
		
		InitializationStrategy[] strats = InitializationStrategy.values();
		
		for(InitializationStrategy s: strats) {
			KMeans model = new KMeans(data, new KMeansParameters(3)
				.setInitializationStrategy(s)
				.setVerbose(true)).fit();
			
			System.out.println("Silhouette score: " + model.silhouetteScore());
			System.out.println("Idx affinity score: " + model.indexAffinityScore(labels));
			System.out.println();
		}
		
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPartitionalClass1() {
		new KMeans(data_, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testPartitionalClass2() {
		new KMeans(data_, 151);
	}
	
	@Test
	public void testBestDistIris() {
		DataSet ds = TestSuite.IRIS_DATASET.shuffle();
		findBestDistMetric(ds,3);
	}
	
	@Test
	public void testBestDistWine() {
		DataSet ds = TestSuite.WINE_DATASET.shuffle();
		findBestDistMetric(ds,3);
	}
	
	@Test
	public void testBestDistBC() {
		DataSet ds = TestSuite.BC_DATASET.shuffle();
		findBestDistMetric(ds,2);
	}
	
	@Test
	public void testBestKernIris() {
		DataSet ds = TestSuite.IRIS_DATASET.shuffle();
		findBestKernelMetric(ds,3);
	}
	
	@Test
	public void testBestKernWine() {
		DataSet ds = TestSuite.WINE_DATASET.shuffle();
		findBestKernelMetric(ds,3);
	}
	
	@Test
	public void testBestKernBC() {
		DataSet ds = TestSuite.BC_DATASET.shuffle();
		findBestKernelMetric(ds,2);
	}
	
	static void findBestDistMetric(DataSet ds, int k) {
		final Array2DRowRealMatrix d = ds.getData();
		PreProcessor scaler = new StandardScaler().fit(d);
		final RealMatrix X = scaler.transform(d);
		
		final int[] actual = ds.getLabels();
		GeometricallySeparable best = null;
		double ia = 0;
		
		// it's not linearly separable, so most won't perform incredibly well...
		KMeans model;
		int count = 0;
		for(DistanceMetric dist: Distance.values()) {
			if(KMeans.UNSUPPORTED_METRICS.contains(dist.getClass()))
				continue;
			
			KMeansParameters km = new KMeansParameters(k).setMetric(dist);
			double i = -1;
			
			model = km.fitNewModel(X);
			if(model.getK() != k) // gets modified if totally equal
				continue;
			
			count++;
			i = model.indexAffinityScore(actual);
			

			//System.out.println(model.getSeparabilityMetric().getName() + ", " + i);
			if(i > ia) {
				ia = i;
				best = model.getSeparabilityMetric();
			}
		}
		
		
		System.out.println("BEST: " + best.getName() + ", " + ia + ", successfully tried " + count);
	}
	
	static void findBestKernelMetric(DataSet ds, int k) {
		Array2DRowRealMatrix d = ds.getData();
		StandardScaler scaler = new StandardScaler().fit(d);
		final RealMatrix X = scaler.transform(d);
		final int[] actual = ds.getLabels();
		
		GeometricallySeparable best = null;
		double ia = 0;
		
		// it's not linearly separable, so most won't perform incredibly well...
		int count = 0;
		KMeans model;
		for(Kernel dist: com.clust4j.kernel.KernelTestCases.all_kernels) {
			if(KMeans.UNSUPPORTED_METRICS.contains(dist.getClass()))
				continue;
			
			KMeansParameters km = new KMeansParameters(k).setMetric(dist);
			double i = -1;
			
			model = km.fitNewModel(X);
			if(model.getK() != k) // gets modified if totally equal
				continue;
			
			count++;
			i = model.indexAffinityScore(actual);
			

			//System.out.println(model.getSeparabilityMetric().getName() + ", " + i);
			if(i > ia) {
				ia = i;
				best = model.getSeparabilityMetric();
			}
		}
		
		
		System.out.println("BEST: " + best.getName() + ", " + ia + ", successfully tried " + count);
	}
	
	/**
	 * Assert that when all of the matrix entries are exactly the same,
	 * the algorithm will still converge, yet produce one label: 0
	 */
	@Override
	@Test
	public void testAllSame() {
		final double[][] x = MatUtils.rep(-1, 3, 3);
		final Array2DRowRealMatrix X = new Array2DRowRealMatrix(x, false);
		
		int[] labels = new KMeans(X, new KMeansParameters(3).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
		System.out.println();
	}
	
	/**
	 * Hamming is unsupported. Test that it falls back to Euclidean
	 */
	@Test
	public void testHamming() {
		final Array2DRowRealMatrix X = TestSuite.IRIS_DATASET.shuffle().getData();
		KMeans km = new KMeans(X, new KMeansParameters(3)
				.setVerbose(true).setMetric(Distance.HAMMING))
					.fit();
		assertTrue(km.hasWarnings());
		assertTrue(km.getSeparabilityMetric().equals(Distance.EUCLIDEAN));
		System.out.println();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testTooLowK() {
		int k = 0;
		new KMeans(data_, k);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testTooHighK() {
		int k = 155;
		new KMeans(data_, k);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit() {
		new KMeans(data_, 3).getLabels();
	}
	
	@Test
	public void testMethods() {
		KMeans k = new KMeans(data_, new KMeansParameters(150)
			.setInitializationStrategy(InitializationStrategy.RANDOM));
		assertTrue(k.getMaxIter() == KMeans.DEF_MAX_ITER);
		assertTrue(k.getConvergenceTolerance() == KMeans.DEF_CONVERGENCE_TOLERANCE);
		
		HashSet<Integer> idcs = new HashSet<Integer>();
		for(int i: k.init_centroid_indices)
			idcs.add(i);
		
		assertTrue(idcs.size() == 150);
	}
	
	@Test
	public void testPredict() {
		KMeans k = new KMeans(data_, 3).fit();
		System.out.println("KMeans prediction affinity: " + k.indexAffinityScore(k.predict(data_)));
	}
	
	@Test
	public void testBadKVals() {
		boolean a = false, b = false;
		try {
			new KMeans(data_, 0);
		} catch(IllegalArgumentException e) {
			a = true;
		}
		
		try {
			new KMeans(data_, 1500);
		} catch(IllegalArgumentException e) {
			b = true;
		}
		
		assertTrue(a && b);
	}
	
	@Test
	public void testAutoName() {
		/*
		 * Not very necessary... but just shows it's not null for coverage
		 */
		assertNotNull(AbstractCentroidClusterer.InitializationStrategy.AUTO.getName());
	}
	
	@Test
	public void testDMEEucDists() {
		/*
		 * Also not very necessary, as it's internal and shouldn't hit
		 * this snag, but it provides coverage. Yay.
		 */
		boolean a = false;
		try {
			KMeans.eucDists(
				new double[][]{new double[]{1,2,3}}, 
				new double[][]{new double[]{1,2}});
		} catch(DimensionMismatchException d) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testMNFE() {
		boolean a = false;
		try {
			new KMeans(data_, 3).getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testParallelConflict() {
		final boolean orig = GlobalState.ParallelismConf.PARALLELISM_ALLOWED;
		
		try {
			/*
			 * Set to false and try to force, watch it fail
			 */
			GlobalState.ParallelismConf.PARALLELISM_ALLOWED = false;
			KMeans k = new KMeans(data_, new KMeansParameters(3).setForceParallel(true));
			assertFalse(k.parallel); // can't be true given global prohibits it.
			
		} finally {
			/*
			 * Always need to make sure we reset it!
			 */
			GlobalState.ParallelismConf.PARALLELISM_ALLOWED = orig;
		}
	}
	
	@Test
	public void testNaNInput() {
		boolean a = false;
		Array2DRowRealMatrix d = new Array2DRowRealMatrix(new double[][]{
			new double[]{Double.NaN, 1, 2},
			new double[]{1, 2 , 3}
		}, false);
		
		try {
			new KMeans(d, 1);
		} catch(NaNException n) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testWSS() {
		KMeans model = new KMeansParameters(3).setVerbose(true).fitNewModel(data_);
		final double[] wss = model.getWSS();
		ArrayList<double[]> centroids = model.getCentroids();
		final int[] labels = model.getLabels();
		
		int i = 0, label;
		double[] wss_assert = new double[3], centroid;
		for(double[] row: model.data.getData()) {
			label = labels[i];
			centroid = centroids.get(label);
			
			double sum = 0;
			for(int j = 0; j < centroid.length; j++) {
				double diff = row[j] - centroid[j];
				sum += (diff * diff);
			}
			
			wss_assert[label] += sum;
			i++;
		}
		
		assertTrue(VecUtils.equalsExactly(wss, wss_assert));
		assertTrue(model.getTSS() == model.getBSS() + VecUtils.sum(model.getWSS()));
		assertTrue(new VecUtils.DoubleSeries(new KMeans(data_, 3).getWSS(), Inequality.EQUAL_TO, Double.NaN).all());
	}
	
	
	/**
	 * For testing synchronicity
	 * @author Taylor G Smith
	 */
	abstract static public class KMRunnable implements Runnable {
		boolean hasRun = false;
		final KMeans model;
		int[] labels = new int[0];
		
		public KMRunnable(KMeans model) {
			this.model = model;
		}
		
		public KMRunnable(KMRunnable runner) {
			this.model = runner.model;
		}
	}
	
	@Test
	public void testSynchronization() {
		/*
		 * This is hard to test!
		 */
		KMeans km = new KMeans(data_, new KMeansParameters(3).setVerbose(true));
		
		// create a 2 thread pool with a small buffer for the runnable jobs
		ExecutorService threadPool = Executors.newCachedThreadPool();
		
		// this job executes the fit after a brief nap
		KMRunnable first = new KMRunnable(km){
			@Override
			public void run() {
				try {
					Thread.sleep(1000);
					this.model.fit();
					this.hasRun = true;
				} catch(InterruptedException e) {
					System.out.println("First failed!");
				} finally {
					assertNotNull(this.model.getLabels());
				}
			}
		};

		// this tries to get the volatile objs too early...
		KMRunnable second = new KMRunnable(first){
			@Override
			public void run() {
				boolean was_fit = true;
				
				try {
					this.labels = model.getLabels();
					this.hasRun = true;
				} catch(ModelNotFitException m) {
					this.labels = null;
					was_fit = false;
				} finally {
					System.out.println("Second");
					assertFalse(was_fit);
					assertNull(this.labels);
				}
			}
		};
		
		
		// submit 2 jobs that take a while to run
		threadPool.execute(first);
		threadPool.execute(second);
	}
}
