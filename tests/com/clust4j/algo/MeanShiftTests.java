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

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.TestSuite;
import com.clust4j.algo.MeanShift.SerialCenterIntensity;
import com.clust4j.algo.MeanShift.MeanShiftPlanner;
import com.clust4j.algo.MeanShift.MeanShiftSeed;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.algo.RadiusNeighbors.RadiusNeighborsPlanner;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.except.NonUniformMatrixException;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class MeanShiftTests implements ClusterTest, ClassifierTest, ConvergeableTest, BaseModelTest {
	final static Array2DRowRealMatrix data_ = ExampleDataSets.loadIris().getData();
	final static Array2DRowRealMatrix wine = ExampleDataSets.loadWine().getData();

	@Test
	public void MeanShiftTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {5.0,  4.3,  19.0, 4.0},
			new double[] {9.06, 12.6, 3.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		MeanShift ms = new MeanShift(mat, new MeanShift
			.MeanShiftPlanner(0.5)
				.setVerbose(true)).fit();
		System.out.println();
		
		assertTrue(ms.getNumberOfIdentifiedClusters() == 3);
		assertTrue(ms.getNumberOfNoisePoints() == 0);
		assertTrue(ms.hasWarnings()); // will be because we don't standardize
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
		
		MeanShift ms = new MeanShift(mat, new MeanShift
			.MeanShiftPlanner(0.5)
				.setMaxIter(100)
				.setMinChange(0.0005)
				.setSeed(new Random(100))
				.setVerbose(true)).fit();
		assertTrue(ms.getNumberOfIdentifiedClusters() == 4);
		assertTrue(ms.getLabels()[2] == ms.getLabels()[3]);
		System.out.println();
		
		
		ms = new MeanShift(mat, new MeanShift
			.MeanShiftPlanner(0.05)
				.setVerbose(true)).fit();
		assertTrue(ms.getNumberOfIdentifiedClusters() == 5);
		assertTrue(ms.hasWarnings()); // will because not normalizing
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
		
		MeanShiftPlanner msp = new MeanShiftPlanner(0.5);
		MeanShift ms1 = msp.buildNewModelInstance(mat).fit();
		
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
	
	@Test(expected=ModelNotFitException.class)
	public void testMeanShiftMFE1() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(MatUtils.randomGaussian(50, 2));
		MeanShift ms = new MeanShift(mat, new MeanShiftPlanner());
		ms.getLabels();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMeanShiftMFE2() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(MatUtils.randomGaussian(50, 2));
		MeanShift ms = new MeanShift(mat, 0.5);
		ms.getCentroids();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMeanShiftIAEConst() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(MatUtils.randomGaussian(50, 2));
		new MeanShift(mat, 0.0);
	}

	
	@Test
	public void testChunkSizeMeanShift() {
		final int chunkSize = 500;
		assertTrue(MeanShift.getNumChunks(chunkSize, 500) == 1);
		assertTrue(MeanShift.getNumChunks(chunkSize, 501) == 2);
		assertTrue(MeanShift.getNumChunks(chunkSize, 23) == 1);
		assertTrue(MeanShift.getNumChunks(chunkSize, 10) == 1);
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
			new MeanShiftPlanner().setVerbose(true)).fit();
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
			new MeanShiftPlanner()
				.setAutoBandwidthEstimationQuantile(1.1));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMeanShiftAutoBwEstimateException2() {
		final double[][] x = TestSuite.bigMatrix;
		new MeanShift(new Array2DRowRealMatrix(x), 
			new MeanShiftPlanner()
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
		new MeanShift(data_, new MeanShiftPlanner(0.05));
	}

	@Test
	@Override
	public void testFit() {
		new MeanShift(data_,
			new MeanShiftPlanner()).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new MeanShiftPlanner()
			.buildNewModelInstance(data_);
	}

	@Test
	@Override
	public void testItersElapsed() {
		assertTrue(new MeanShift(data_, 
				new MeanShiftPlanner()).fit().itersElapsed() > 0);
	}

	@Test
	@Override
	public void testConverged() {
		assertTrue(new MeanShift(data_, 
				new MeanShiftPlanner()).fit().didConverge());
	}

	@Test
	@Override
	public void testScoring() {
		new MeanShift(data_,
			new MeanShiftPlanner()).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		MeanShift ms = new MeanShift(data_,
			new MeanShiftPlanner(0.5)
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
		
		// MS estimates bw at 1.202076812799869
		final double bandwidth = 1.202076812799869;
		assertTrue(MeanShift.autoEstimateBW(iris, 0.3, 
			Distance.EUCLIDEAN, GlobalState.DEFAULT_RANDOM_STATE, false) == bandwidth);
		
		// Asserting fit works without breaking things...
		RadiusNeighbors r = new RadiusNeighbors(iris,
			new RadiusNeighborsPlanner(bandwidth)).fit();
		
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
		
		MeanShift ms = new MeanShift(iris, 
			new MeanShift.MeanShiftPlanner()
				.setScale(true)).fit();
		
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
			new MeanShift.MeanShiftPlanner()
				.setScale(true)
				.setSeeds(iris.getData())).fit();
		System.out.println(Arrays.toString(ms.getLabels()));
	}
	*/
	
	@Test(expected=NonUniformMatrixException.class)
	public void testSeededNUME() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShift.MeanShiftPlanner()
				.setSeeds(new double[][]{
					new double[]{1,2,3,4},
					new double[]{0}
				}));
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testSeededDME() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShift.MeanShiftPlanner()
				.setSeeds(new double[][]{
					new double[]{1,2,3},
					new double[]{1,2,3}
				}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testSeededIAE1() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShift.MeanShiftPlanner()
				.setSeeds(new double[][]{
				}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testSeededIAE2() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShift.MeanShiftPlanner()
				.setSeeds(MatUtils.randomGaussian(iris.getRowDimension() + 1, 
					iris.getColumnDimension())));
	}
	
	@Test
	public void testSeededIrisFunctional() {
		Array2DRowRealMatrix iris = data_;
		
		new MeanShift(iris, 
			new MeanShift.MeanShiftPlanner()
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
		Array2DRowRealMatrix iris = (Array2DRowRealMatrix)FeatureNormalization
			.STANDARD_SCALE.operate(data_);
		final double[][] X = iris.getData();
		
		// MS estimates bw at 1.5971266273438018
		final double bandwidth = 1.5971266273438018;
		
		assertTrue(MeanShift.autoEstimateBW(iris, 0.3, 
			Distance.EUCLIDEAN, GlobalState.DEFAULT_RANDOM_STATE, false) == bandwidth);
		assertTrue(MeanShift.autoEstimateBW(iris, 0.3, 
			Distance.EUCLIDEAN, GlobalState.DEFAULT_RANDOM_STATE, true) == bandwidth);
		
		// Asserting fit works without breaking things...
		RadiusNeighbors r = new RadiusNeighbors(iris,
			new RadiusNeighborsPlanner(bandwidth)).fit();
				
		TreeSet<MeanShiftSeed> centers = new TreeSet<>();
		for(double[] seed: X)
			centers.add(MeanShift.singleSeed(seed, r, X, 300));
		
		assertTrue(centers.size() == 5);
		
		double[][] expected_dists = new double[][]{
			new double[]{ 0.50161528154395962, -0.31685274298813487,  0.65388162422893481, 0.65270450741975761},
			new double[]{ 0.4829041180399124,  -0.3184802762043775,   0.6434194172372906,  0.6471200248238047 },
			new double[]{ 0.52001211065400177, -0.29561728795619946,  0.67106269515983397, 0.67390853215763813},
			new double[]{ 0.54861244890482475, -0.25718786696105495,  0.68964559485632182, 0.69326664641211422},
			new double[]{-1.0595457115461515,   0.74408909010240054, -1.2995708885010491, -1.2545442961404225 }
		};
		
		int[] expected_centers= new int[]{
			82, 81, 80, 77, 45
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
		
		
		
		// Now test the actual method...
		SerialCenterIntensity intensity = new SerialCenterIntensity(
				iris, r, bandwidth, X,
				Distance.EUCLIDEAN, 300);
		
		ArrayList<EntryPair<double[], Integer>> center_intensity2 = intensity.pairs;
		assertTrue(center_intensity2.size() == center_intensity.size());
		
		
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
			new RadiusNeighborsPlanner(bandwidth)
				.setVerbose(false)).fit();
		
		
		// Iterate over sorted centers and query radii
		int[] indcs;
		double[] center;
		for(int i = 0; i < m_prime; i++) {
			if(unique[i]) {
				center = sorted_centers.getRow(i);
				indcs = nbrs.getNeighbors(
					new double[][]{center}, bandwidth)
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
		
		assertTrue(redundant_ct == 3);
		assertTrue(centroids.size() == 2);
		assertTrue(VecUtils.equalsExactly(centroids.get(0), new double[]{
			0.4999404345258693, -0.3217963110452594, 0.6517519610505076, 0.6504383581073984
		}));
		
		assertTrue(VecUtils.equalsExactly(centroids.get(1), new double[]{
			-1.05600798643927, 0.7555834087538411, -1.2954688594835102, -1.2498288991228386
		}));
		
		
		// also put the centroids into a matrix. We have to
		// wait to perform this op, because we have to know
		// the size of centroids first...
		Array2DRowRealMatrix clust_centers = new Array2DRowRealMatrix(centroids.size(), iris.getColumnDimension());
		for(int i = 0; i < clust_centers.getRowDimension(); i++)
			clust_centers.setRow(i, centroids.get(i));
		
		// The final nearest neighbors model -- if this works, we are in the clear...
		new NearestNeighbors(clust_centers,
			new NearestNeighborsPlanner(1)).fit();
	}
	
	@Test
	public void testOnWineDataNonScaled() {
		new MeanShift(wine,
			new MeanShiftPlanner()
			.setVerbose(true)).fit();
	}
	
	@Test
	public void testOnWineDataScaled() {
		new MeanShift(wine,
			new MeanShiftPlanner()
				.setScale(true)
				.setVerbose(true)).fit();
	}
	
	@Test
	public void testParallelSmall() {
		try {
			MeanShift iris_serial = new MeanShift(data_, new MeanShiftPlanner()).fit();

			GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = true;
			MeanShift iris_paral = new MeanShift(data_, new MeanShiftPlanner()).fit();

			assertTrue(iris_serial.silhouetteScore() == iris_paral.silhouetteScore());
			assertTrue(VecUtils.equalsExactly(iris_serial.getLabels(), iris_paral.getLabels()));
		} finally {
			GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = false;
		}
	}
	
	@Test
	public void testParallelLarger() {
		try {
			MeanShift wine_serial = new MeanShift(wine, 
				new MeanShiftPlanner().setVerbose(true)).fit();
			System.out.println();
			
			GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = true;
			MeanShift wine_paral = new MeanShift(wine, 
				new MeanShiftPlanner().setVerbose(true)).fit();
			System.out.println();
			
			assertTrue(wine_serial.getBandwidth() == wine_paral.getBandwidth());
			assertTrue(wine_serial.silhouetteScore() == wine_paral.silhouetteScore());
			assertTrue(VecUtils.equalsExactly(wine_serial.getLabels(), wine_paral.getLabels()));
		} finally {
			GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = false;
		}
	}
	
	@Test
	public void testParallelHuge() {
		GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = true;
		
		try {
			// Construct a large matrix of two separate gaussian seeds
			double[][] A = MatUtils.randomGaussian(1000, 20);
			double[][] B = MatUtils.randomGaussian(1000, 20, 25.0);
			double[][] C = MatUtils.rbind(A, B);
			
			new MeanShift(new Array2DRowRealMatrix(C, false),
				new MeanShiftPlanner()
					//.setScale(true)
					.setVerbose(true)).fit();
			
			// This should result in one cluster. We are testing that that works.
		} finally {
			GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = false;
		}
	}
}
