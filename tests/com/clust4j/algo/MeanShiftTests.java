package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.MeanShift.MeanShiftPlanner;
import com.clust4j.data.DataSet;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.utils.Distance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

public class MeanShiftTests implements ClusterTest, ClassifierTest, ConvergeableTest {
	final static Array2DRowRealMatrix data_ = ExampleDataSets.IRIS.getData();

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
		assertTrue(ms1.getConvergenceTolerance() == MeanShift.DEF_MIN_CHANGE);
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
	
	// Hard condition to force..
	@Test(expected=com.clust4j.utils.IllegalClusterStateException.class)
	public void MeanShiftTest4() {
		DataSet iris = ExampleDataSets.IRIS;
		final Array2DRowRealMatrix data = iris.getData();
		
		new MeanShift(data, 
			new MeanShift.MeanShiftPlanner()
				.setScale(true)
				.setVerbose(true)).fit();
		System.out.println();
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
		double bw = MeanShift.autoEstimateBW(new Array2DRowRealMatrix(x, false), 0.3, Distance.EUCLIDEAN, new Random());
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
		new MeanShift(ExampleDataSets.IRIS.getData());
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
		ms.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		MeanShift ms2 = (MeanShift)MeanShift.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(ms2.getNumberOfNoisePoints() == n);
		assertTrue(ms.equals(ms2));
		Files.delete(TestSuite.path);
	}
}
