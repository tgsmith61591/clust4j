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

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.TestSuite;
import com.clust4j.algo.MeanShiftParameters;
import com.clust4j.algo.MeanShift.MeanShiftSeed;
import com.clust4j.algo.NearestNeighborsParameters;
import com.clust4j.algo.RadiusNeighborsParameters;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.except.IllegalClusterStateException;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.except.NonUniformMatrixException;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.metrics.pairwise.Similarity;
import com.clust4j.metrics.pairwise.SimilarityMetric;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class MeanShiftTests implements ClusterTest, ClassifierTest, ConvergeableTest, BaseModelTest {
	final static Array2DRowRealMatrix data_ = TestSuite.IRIS_DATASET.getData();
	final static Array2DRowRealMatrix wine = TestSuite.WINE_DATASET.getData();

	@Test
	public void MeanShiftTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {5.0,  4.3,  19.0, 4.0},
			new double[] {9.06, 12.6, 3.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		MeanShift ms = new MeanShift(mat, new MeanShiftParameters(0.5)
				.setVerbose(true)).fit();
		System.out.println();
		
		assertTrue(ms.getNumberOfIdentifiedClusters() == 3);
		assertTrue(ms.getNumberOfNoisePoints() == 0);
	}
	
	@Test
	public void MeanShiftTest2() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		MeanShift ms = new MeanShift(mat, new MeanShiftParameters(0.5)
				.setMaxIter(100)
				.setMinChange(0.0005)
				.setSeed(new Random(100))
				.setVerbose(true)).fit();
		assertTrue(ms.getNumberOfIdentifiedClusters() == 4);
		assertTrue(ms.getLabels()[2] == ms.getLabels()[3]);
		System.out.println();
		
		
		ms = new MeanShift(mat, new MeanShiftParameters(0.05)
				.setVerbose(true)).fit();
		assertTrue(ms.getNumberOfIdentifiedClusters() == 5);
		System.out.println();
	}
	
	@Test
	public void MeanShiftTest3() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		MeanShift ms = new MeanShift(mat, 0.5).fit();
		assertTrue(ms.getNumberOfIdentifiedClusters() == 4);
		assertTrue(ms.getLabels()[2] == ms.getLabels()[3]);
		
		MeanShiftParameters msp = new MeanShiftParameters(0.5);
		MeanShift ms1 = msp.fitNewModel(mat).fit();
		
		assertTrue(ms1.getBandwidth() == 0.5);
		assertTrue(ms1.didConverge());
		assertTrue(MatUtils.equalsExactly(ms1.getKernelSeeds(), train_array));
		assertTrue(ms1.getMaxIter() == MeanShift.DEF_MAX_ITER);
		assertTrue(ms1.getConvergenceTolerance() == MeanShift.DEF_TOL);
		assertTrue(ms.getNumberOfIdentifiedClusters() == ms1.getNumberOfIdentifiedClusters());
		assertTrue(VecUtils.equalsExactly(ms.getLabels(), ms1.getLabels()));
		
		// check that we can get the centroids...
		assertTrue(null != ms.getCentroids());
	}
	
	@Test
	public void testMeanShiftMFE1() {
		boolean a = false;
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(MatUtils.randomGaussian(50, 2));
		MeanShift ms = new MeanShift(mat, 0.5);
		try {
			ms.getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testMeanShiftMFE2() {
		boolean a = false;
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(MatUtils.randomGaussian(50, 2));
		MeanShift ms = new MeanShift(mat, 0.5);
		try {
			ms.getCentroids();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMeanShiftIAEConst() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(MatUtils.randomGaussian(50, 2));
		new MeanShift(mat, 0.0);
	}

	
	@Test
	public void testChunkSizeMeanShift() {
		final int chunkSize = 500;
		assertTrue(ParallelChunkingTask.ChunkingStrategy.getNumChunks(chunkSize, 500) == 1);
		assertTrue(ParallelChunkingTask.ChunkingStrategy.getNumChunks(chunkSize, 501) == 2);
		assertTrue(ParallelChunkingTask.ChunkingStrategy.getNumChunks(chunkSize, 23) == 1);
		assertTrue(ParallelChunkingTask.ChunkingStrategy.getNumChunks(chunkSize, 10) == 1);
	}
	
	@Test
	public void testMeanShiftAutoBwEstimate1() {
		final double[][] x = TestSuite.bigMatrix;
		double bw = MeanShift.autoEstimateBW(new Array2DRowRealMatrix(x, false), 
				0.3, Distance.EUCLIDEAN, new Random(), false);
		new MeanShift(new Array2DRowRealMatrix(x), bw).fit();
	}
	
	@Test
	public void testMeanShiftAutoBwEstimate2() {
		final double[][] x = TestSuite.bigMatrix;
		MeanShift ms = new MeanShift(new Array2DRowRealMatrix(x), 
			new MeanShiftParameters().setVerbose(true)).fit();
		System.out.println();
		assertTrue(ms.itersElapsed() >= 1);
		
		ms.fit(); // re-fit
		System.out.println();
	}
	
	@Test
	public void testMeanShiftAutoBwEstimate3() {
		final double[][] x = TestSuite.bigMatrix;
		MeanShift ms = new MeanShift(new Array2DRowRealMatrix(x)).fit();
		assertTrue(ms.getBandwidth() == 0.9148381982960355);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMeanShiftAutoBwEstimateException1() {
		final double[][] x = TestSuite.bigMatrix;
		new MeanShift(new Array2DRowRealMatrix(x), 
			new MeanShiftParameters()
				.setAutoBandwidthEstimationQuantile(1.1));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMeanShiftAutoBwEstimateException2() {
		final double[][] x = TestSuite.bigMatrix;
		new MeanShift(new Array2DRowRealMatrix(x), 
			new MeanShiftParameters()
				.setAutoBandwidthEstimationQuantile(0.0));
	}

	@Test
	@Override
	public void testDefConst() {
		new MeanShift(data_);
	}

	@Test
	@Override
	public void testArgConst() {
		new MeanShift(data_, 0.05);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new MeanShift(data_, new MeanShiftParameters(0.05));
	}

	@Test
	@Override
	public void testFit() {
		new MeanShift(data_,
			new MeanShiftParameters()).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new MeanShiftParameters()
			.fitNewModel(data_);
	}

	@Test
	@Override
	public void testItersElapsed() {
		assertTrue(new MeanShift(data_, 
				new MeanShiftParameters()).fit().itersElapsed() > 0);
	}

	@Test
	@Override
	public void testConverged() {
		assertTrue(new MeanShift(data_, 
				new MeanShiftParameters()).fit().didConverge());
	}

	@Test
	@Override
	public void testScoring() {
		new MeanShift(data_,
			new MeanShiftParameters()).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		MeanShift ms = new MeanShift(data_,
			new MeanShiftParameters(0.5)
				.setVerbose(true)).fit();
		System.out.println();
		
		final double n = ms.getNumberOfNoisePoints();
		ms.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		MeanShift ms2 = (MeanShift)MeanShift.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(ms2.getNumberOfNoisePoints() == n);
		assertTrue(ms.equals(ms2));
		Files.delete(TestSuite.path);
	}
	
	@Test
	public void testAutoEstimation() {
		Array2DRowRealMatrix iris = data_;
		final double[][] X = iris.getData();
		
		// MS estimates bw at 1.2032034114912584
		final double bandwidth = 1.2032034114912584;
		assertTrue(MeanShift.autoEstimateBW(iris, 0.3, 
			Distance.EUCLIDEAN, GlobalState.DEFAULT_RANDOM_STATE, false) == bandwidth);
		
		// Asserting fit works without breaking things...
		RadiusNeighbors r = new RadiusNeighbors(iris,
			new RadiusNeighborsParameters(bandwidth)).fit();
		
		TreeSet<MeanShiftSeed> centers = new TreeSet<>();
		for(double[] seed: X)
			centers.add(MeanShift.singleSeed(seed, r, X, 300));
		
		assertTrue(centers.size() == 7);

		double[][] expected_dists = new double[][]{
			new double[]{6.2114285714285691, 2.8928571428571428, 4.8528571428571423, 1.6728571428571426},
			new double[]{6.1927536231884037, 2.8768115942028984, 4.8188405797101437, 1.6463768115942023},
			new double[]{6.1521739130434767, 2.850724637681159,  4.7405797101449272, 1.6072463768115937},
			new double[]{6.1852941176470564, 2.8705882352941177, 4.8058823529411754, 1.6397058823529407},
			new double[]{6.1727272727272711, 2.874242424242424,  4.7757575757575745, 1.6287878787878785},
			new double[]{5.0163265306122451, 3.440816326530614,  1.46734693877551,   0.24285714285714283},
			new double[]{5.0020833333333341, 3.4208333333333356, 1.4666666666666668, 0.23958333333333334}
		};
		
		int[] expected_centers= new int[]{
			70, 69, 69, 68, 66, 49, 48
		};
		
		int idx = 0;
		for(MeanShiftSeed seed: centers) {
			assertTrue(VecUtils.equalsWithTolerance(seed.dists, expected_dists[idx], 1e-1));
			assertTrue(seed.count == expected_centers[idx]);
			idx++;
		}
	}
	

	
	/*
	 * Test iris with no seeds
	 */
	@Test
	public void MeanShiftTestIris() {
		Array2DRowRealMatrix iris = data_;
		StandardScaler scaler = new StandardScaler().fit(iris);
		RealMatrix X = scaler.transform(iris);
		
		MeanShift ms = new MeanShift(X, 
			new MeanShiftParameters()).fit();
		
		// sklearn output
		int[] expected = new LabelEncoder(new int[]{
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		}).fit().getEncodedLabels();
		
		// Assert almost equal
		assertTrue(Precision.equals(ms.indexAffinityScore(expected), 1.0, 1e-2));
	}
	
	/*
	 * Test where none in range
	@Test(expected=com.clust4j.except.IllegalClusterStateException.class)
	public void MeanShiftTestIrisSeeded() {
		Array2DRowRealMatrix iris = data_;
		
		MeanShift ms =
		new MeanShift(iris, 
			new MeanShiftPlanner()
				.setScale(true)
				.setSeeds(iris.getData())).fit();
		System.out.println(Arrays.toString(ms.getLabels()));
	}
	*/
	
	@Test(expected=NonUniformMatrixException.class)
	public void testSeededNUME() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShiftParameters()
				.setSeeds(new double[][]{
					new double[]{1,2,3,4},
					new double[]{0}
				}));
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testSeededDME() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShiftParameters()
				.setSeeds(new double[][]{
					new double[]{1,2,3},
					new double[]{1,2,3}
				}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testSeededIAE1() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShiftParameters()
				.setSeeds(new double[][]{
				}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testSeededIAE2() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShiftParameters()
				.setSeeds(MatUtils.randomGaussian(iris.getRowDimension() + 1, 
					iris.getColumnDimension())));
	}
	
	@Test
	public void testSeededIrisFunctional() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShiftParameters()
				.setVerbose(true)
				.setSeeds(new double[][]{
					iris.getRow(3),
					iris.getRow(90),
					iris.getRow(120)
				})).fit();
		System.out.println();
	}
	
	@Test
	public void testAutoEstimationWithScale() {
		Array2DRowRealMatrix iris = (Array2DRowRealMatrix)new StandardScaler().fit(data_).transform(data_);
		final double[][] X = iris.getData();
		
		// MS estimates bw at 1.6041295821313855
		final double bandwidth = 1.6041295821313855;
		
		assertTrue(
			Precision.equals(
				MeanShift.autoEstimateBW(iris, 0.3, 
				Distance.EUCLIDEAN, GlobalState.DEFAULT_RANDOM_STATE, false), 
				bandwidth, 1e-9));
		
		assertTrue(
			Precision.equals(
				MeanShift.autoEstimateBW(iris, 0.3, 
				Distance.EUCLIDEAN, GlobalState.DEFAULT_RANDOM_STATE, true), 
				bandwidth, 1e-9));
		
		// Asserting fit works without breaking things...
		RadiusNeighbors r = new RadiusNeighbors(iris,
			new RadiusNeighborsParameters(bandwidth)).fit();
				
		TreeSet<MeanShiftSeed> centers = new TreeSet<>();
		for(double[] seed: X)
			centers.add(MeanShift.singleSeed(seed, r, X, 300));
		
		assertTrue(centers.size() == 4);
		
		double[][] expected_dists = new double[][]{
			new double[]{ 0.50161528154395962, -0.31685274298813487, 0.65388162422893481, 0.65270450741975761 },
			new double[]{ 0.52001211065400177, -0.29561728795619946, 0.67106269515983397, 0.67390853215763813 },
			new double[]{ 0.54861244890482475, -0.25718786696105495, 0.68964559485632182, 0.69326664641211422 },
			new double[]{-1.0595457115461515,   0.74408909010240054,-1.2995708885010491 ,-1.2545442961404225  }
		};
		
		int[] expected_centers= new int[]{
			82, 80, 77, 45
		};
		
		int idx = 0;
		for(MeanShiftSeed seed: centers) {
			assertTrue(VecUtils.equalsWithTolerance(seed.dists, expected_dists[idx], 1e-1));
			assertTrue(seed.count == expected_centers[idx]);
			idx++;
		}
		
		ArrayList<EntryPair<double[], Integer>> center_intensity = new ArrayList<>();
		for(MeanShiftSeed seed: centers) {
			if(null != seed) {
				center_intensity.add(seed.getPair());
			}
		}
		
		
		final ArrayList<EntryPair<double[], Integer>> sorted_by_intensity = center_intensity;
		
		// test getting the unique vals
		idx = 0;
		final int m_prime = sorted_by_intensity.size();
		final Array2DRowRealMatrix sorted_centers = new Array2DRowRealMatrix(m_prime, iris.getColumnDimension());
		for(Map.Entry<double[], Integer> e: sorted_by_intensity)
			sorted_centers.setRow(idx++, e.getKey());
		
		
		// Create a boolean mask, init true
		final boolean[] unique = new boolean[m_prime];
		for(int i = 0; i < unique.length; i++) unique[i] = true;
		
		// Fit the new neighbors model
		RadiusNeighbors nbrs = new RadiusNeighbors(sorted_centers,
			new RadiusNeighborsParameters(bandwidth)
				.setVerbose(false)).fit();
		
		
		// Iterate over sorted centers and query radii
		int[] indcs;
		double[] center;
		for(int i = 0; i < m_prime; i++) {
			if(unique[i]) {
				center = sorted_centers.getRow(i);
				indcs = nbrs.getNeighbors(new double[][]{center}, bandwidth, false)
						.getIndices()[0];
				
				for(int id: indcs) {
					unique[id] = false;
				}
				
				unique[i] = true; // Keep this as true
			}
		}
		
		
		// Now assign the centroids...
		int redundant_ct = 0;
		final ArrayList<double[]> centroids =  new ArrayList<>();
		for(int i = 0; i < unique.length; i++) {
			if(unique[i]) {
				centroids.add(sorted_centers.getRow(i));
			}
		}
		
		redundant_ct = unique.length - centroids.size();
		
		assertTrue(redundant_ct == 2);
		assertTrue(centroids.size() == 2);
		assertTrue(VecUtils.equalsWithTolerance(centroids.get(0), new double[]{
			0.4999404345258691, -0.3157948009929614, 0.6516983739795399, 0.6505251874544873
		}, 1e-6));

		assertTrue(VecUtils.equalsExactly(centroids.get(1), new double[]{
			-1.0560079864392702, 0.7416046454700266, -1.295231741534238, -1.2503554887998656
		}));
		
		
		// also put the centroids into a matrix. We have to
		// wait to perform this op, because we have to know
		// the size of centroids first...
		Array2DRowRealMatrix clust_centers = new Array2DRowRealMatrix(centroids.size(), iris.getColumnDimension());
		for(int i = 0; i < clust_centers.getRowDimension(); i++)
			clust_centers.setRow(i, centroids.get(i));
		
		// The final nearest neighbors model -- if this works, we are in the clear...
		new NearestNeighbors(clust_centers,
			new NearestNeighborsParameters(1)).fit();
	}
	
	@Test
	public void testOnWineDataNonScaled() {
		new MeanShift(wine,
			new MeanShiftParameters()
			.setVerbose(true)).fit();
	}
	
	@Test
	public void testOnWineDataScaled() {
		new MeanShift(new StandardScaler().fit(wine).transform(wine),
			new MeanShiftParameters()
				.setVerbose(true)).fit();
	}
	
	@Test
	public void testParallelSmall() {
		MeanShift iris_serial = new MeanShift(data_, new MeanShiftParameters().setVerbose(true)).fit();
		System.out.println();
		
		MeanShift iris_paral = null;
		try {
			iris_paral = new MeanShift(data_, new MeanShiftParameters().setForceParallel(true).setVerbose(true)).fit();
		} catch(OutOfMemoryError | RejectedExecutionException e) {
			// don't propagate these...
			return;
		}
		System.out.println();
		
		assertTrue(iris_serial.silhouetteScore() == iris_paral.silhouetteScore());
		assertTrue(VecUtils.equalsExactly(iris_serial.getLabels(), iris_paral.getLabels()));
	}
	
	@Test
	public void testParallelLarger() {
		MeanShift wine_serial = new MeanShift(wine, 
			new MeanShiftParameters().setVerbose(true)).fit();
		System.out.println();
		
		MeanShift wine_paral = null; 
		try {
			wine_paral = new MeanShift(wine, 
					new MeanShiftParameters().setVerbose(true).setForceParallel(true)).fit();
		} catch(OutOfMemoryError | RejectedExecutionException e) {
			// don't propagate these...
			return;
		}
		System.out.println();
		
		assertTrue(Precision.equals(wine_serial.getBandwidth(), wine_paral.getBandwidth(), 1e-9));
		assertTrue(wine_serial.silhouetteScore() == wine_paral.silhouetteScore());
		assertTrue(VecUtils.equalsExactly(wine_serial.getLabels(), wine_paral.getLabels()));
	}
	
	@Test
	public void testParallelHuge() {
		final int n = 10;
		
		// Construct a large matrix of two separate gaussian seeds
		double[][] A = MatUtils.randomGaussian(400, n);
		double[][] B = MatUtils.randomGaussian(400, n, 25.0);
		double[][] C = MatUtils.rbind(A, B);
		try {
			new MeanShift(new Array2DRowRealMatrix(C, false),
				new MeanShiftParameters()
					//.setScale(true)
					.setVerbose(true)
					.setForceParallel(true)).fit();
		} catch(OutOfMemoryError | RejectedExecutionException e) {
			// don't propagate these...
			return;
		}
		
		// This should result in one cluster. We are testing that that works.
	}
	
	/**
	 * Asser that when all of the matrix entries are exactly the same,
	 * the algorithm will still converge, yet produce one label: 0
	 */
	@Override
	@Test
	public void testAllSame() {
		final double[][] x = MatUtils.rep(-1, 3, 3);
		final Array2DRowRealMatrix X = new Array2DRowRealMatrix(x, false);
		
		int[] labels = new MeanShift(X, new MeanShiftParameters().setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
		
		labels = new MeanShift(X, new MeanShiftParameters(0.5).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
	}
	
	@Test
	public void testValidMetrics() {
		MeanShift model;
		MeanShiftParameters planner;
		Array2DRowRealMatrix small= TestSuite.IRIS_SMALL.getData();
		double bandwidth = 1.5;
		
		/*
		 * Estimate bw and not
		 */
		for(boolean b: new boolean[]{true, false}) {
			for(Distance d: Distance.values()) {
				planner = b ? new MeanShiftParameters() : new MeanShiftParameters(bandwidth);
				planner = planner.setMetric(d);
				model = planner.fitNewModel(data_).fit();
				assertTrue(model.dist_metric.equals(d)); // assert didn't change
			}
			
			// minkowski?
			DistanceMetric d = new MinkowskiDistance(1.5);
			planner = b ? new MeanShiftParameters() : new MeanShiftParameters(bandwidth);
			planner = planner.setMetric(d);
			model = planner.fitNewModel(data_).fit();
			assertTrue(model.dist_metric.equals(d)); // assert didn't change
			
			// haversine?
			d = Distance.HAVERSINE.MI;
			planner = b ? new MeanShiftParameters() : new MeanShiftParameters(bandwidth);
			planner = planner.setMetric(d);
			model = planner.fitNewModel(small).fit();
			assertTrue(model.dist_metric.equals(d)); // assert didn't change
			
			// prove that similarity gets rejected
			d = Distance.EUCLIDEAN;
			SimilarityMetric sim = Similarity.COSINE;
			planner = b ? new MeanShiftParameters() : new MeanShiftParameters(bandwidth);
			planner = planner.setMetric(sim);
			model = planner.fitNewModel(data_).fit();
			assertTrue(model.dist_metric.equals(d)); // assert DID change
		}
	}
	
	@Test
	public void testPredict() {
		MeanShift a = new MeanShift(data_).fit();
		System.out.println("MeanShift prediction affinity: " + a.indexAffinityScore(a.predict(data_)));
	}
	
	@Test //travis may not be able to handle this...
	public void testVerySmallParallelJob() {
		/*
		 * Travis CI is not too capable of extremely large parallel jobs,
		 * but we might be able to get away with small ones like this.
		 */
		final boolean orig = GlobalState.ParallelismConf.PARALLELISM_ALLOWED;
		try {
			/*
			 * No matter the specs of the system testing this, we 
			 * need to ensure it will be able to force parallelism
			 */
			GlobalState.ParallelismConf.PARALLELISM_ALLOWED = true;
			Array2DRowRealMatrix a= new Array2DRowRealMatrix(new double[][]{
				new double[]{1,2,1},
				new double[]{2,1,2},
				new double[]{1,1,1},
				new double[]{2,2,2},
				new double[]{100,101,102},
				new double[]{99,100,101},
				new double[]{98,103,100}
			}, false);
			
			/*
			 * Should obviously be two clusters here...
			 */
			int[] ms1 = new MeanShift(a, new MeanShiftParameters().setForceParallel(true)).fit().getLabels();
			int[] ms2 = new MeanShift(a, new MeanShiftParameters().setForceParallel(false)).fit().getLabels();
			assertTrue(VecUtils.equalsExactly(ms1, ms2));
		} finally {
			/*
			 * Reset
			 */
			GlobalState.ParallelismConf.PARALLELISM_ALLOWED = orig;
		}
	}
	
	@Test
	public void testExceptions() {
		boolean a = false;
		
		/*
		 * This should cause the planner.bandwidth <= 0.0 flag to be thrown
		 */
		try { new MeanShiftParameters(0.0).fitNewModel(data_); } 
		catch(IllegalArgumentException i){ a= true; } 
		finally{ assertTrue(a); a = false; }
		
		/*
		 * This should cause the empty seeds flag to be thrown
		 */
		try { new MeanShiftParameters().setSeeds(new double[][]{}).fitNewModel(data_); } 
		catch(IllegalArgumentException i){ a= true; } 
		finally{ assertTrue(a); a = false; }
		
		
		/*
		 * Bigger seed test
		 */
		try { 
			new MeanShiftParameters().setSeeds(new double[][]{
				new double[]{0.0,0.01},
				new double[]{60.0,12.1}
			}).fitNewModel(data_);
		} 
		catch(DimensionMismatchException i){ a= true; } 
		finally{ assertTrue(a); a = false; }
		
		/* Try seeds that exceed iris in length */
		try { new MeanShiftParameters().setSeeds(MatUtils.randomGaussian(160, 4)).fitNewModel(data_); } catch(IllegalArgumentException i){ a= true; } finally{ assertTrue(a); a = false; }
		/* Try a bad quantile */
		try { new MeanShiftParameters().setAutoBandwidthEstimationQuantile(1.5).fitNewModel(data_); } catch(IllegalArgumentException i){ a= true; } finally{ assertTrue(a); a = false; }
	
		/*
		 * This is a hard test to replicate... that all points are too far from provided seeds
		 */
		try {
			StandardScaler scaler = new StandardScaler().fit(data_);
			new MeanShiftParameters(1.5)
				.setSeeds(new double[][]{
					new double[]{1500,1250,1300,1557},
					new double[]{150,175,250,189}
				}).fitNewModel(scaler.transform(data_)).fit();
		} catch(IllegalClusterStateException i) {
			a = true;
		} finally {
			assertTrue(a);
			a = false;
		}
	}
}
