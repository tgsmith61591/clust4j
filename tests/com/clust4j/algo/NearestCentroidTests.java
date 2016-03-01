package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

public class NearestCentroidTests implements ClassifierTest, ClusterTest {
	final Array2DRowRealMatrix data_ = ExampleDataSets.IRIS.getData();
	final int[] target_ = ExampleDataSets.IRIS.getLabels();

	@Test
	@Override
	public void testDefConst() {
		new NearestCentroid(data_, target_);
	}

	@Test
	@Override
	public void testArgConst() {
		// NA
		assertTrue(true);
		return;
	}

	@Test
	@Override
	public void testPlannerConst() {
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner());
	}

	@Test
	@Override
	public void testFit() {
		new NearestCentroid(data_, target_).fit();
		new NearestCentroid(data_, target_).fit().fit(); // Test fit again... ensure no exceptions
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner()).fit();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner().setShrinkage(0.5)).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new NearestCentroid.NearestCentroidPlanner().buildNewModelInstance(data_, target_);
	}

	@Test
	@Override
	public void testScoring() {
		new NearestCentroid(data_, target_).fit().score();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner()).fit().score();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner().setVerbose(true)).fit().score();
		new NearestCentroid(data_, target_, new NearestCentroid.NearestCentroidPlanner().setShrinkage(0.5)).fit().score();
	}

	@Test(expected=DimensionMismatchException.class)
	public void testDME() {
		new NearestCentroid(data_, new int[]{1,2,3});
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE() {
		new NearestCentroid(data_, VecUtils.repInt(1, data_.getRowDimension()));
	}
	
	@Test
	public void testWarn() {
		/*// We need to allow this behavior now that NC used in KMeans
		NearestCentroid nn =
			new NearestCentroid(data_, target_, 
				new NearestCentroid.NearestCentroidPlanner()
					.setSep(new GaussianKernel()));
		assertTrue(nn.hasWarnings());
		*/
	}
	
	@Test
	public void testMiscellany() {
		assertTrue(new NearestCentroid.NearestCentroidPlanner()
			.getNormalizer().equals(AbstractClusterer.DEF_NORMALIZER));
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE1() {
		new NearestCentroid(data_, target_).getCentroids();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE2() {
		new NearestCentroid(data_, target_).predict(data_);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE3() {
		new NearestCentroid(data_, target_).getLabels();
	}
	
	@Test
	public void testLabels() {
		final int[] copy = new NearestCentroid(data_, target_).getTrainingLabels();
		copy[0] = 9; // Testing immutability of copy
		assertFalse(target_[0] == 9);
	}
	
	@Test
	public void testGetters() {
		NearestCentroid nn = new NearestCentroid(data_, target_).fit();
		nn.getCentroids();
		nn.predict(data_);
		nn.getLabels();
	}
	
	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		NearestCentroid nn = new NearestCentroid(data_, target_,
			new NearestCentroid.NearestCentroidPlanner()
				.setVerbose(true)
				.setScale(true)).fit();
		
		final int[] c = nn.getLabels();
		nn.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		NearestCentroid nn2 = (NearestCentroid)NearestCentroid.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(VecUtils.equalsExactly(c, nn2.getLabels()));
		assertTrue(nn2.equals(nn));
		Files.delete(TestSuite.path);
	}
	
	@Test
	public void testCentroidViabilityKMeans() {
		final double[][] X = new double[][]{
			new double[]{0,0,0},
			new double[]{4,4,4},
			new double[]{8,8,8}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(X, false);
		NearestCentroid nn = new NearestCentroid(mat, new int[]{0,1,2},
			new NearestCentroid.NearestCentroidPlanner()
				.setVerbose(true)
				.setScale(false)).fit();
		
		Array2DRowRealMatrix Y = new Array2DRowRealMatrix(
			new double[][]{
				new double[]{0,0,0},
				new double[]{1,1,1},
				new double[]{4,4,4},
				new double[]{5,5,5},
				new double[]{8,8,8},
				new double[]{9,9,9}
			}, false);
		
		assertTrue(VecUtils.equalsExactly(nn.predict(Y), new int[]{0,0,1,1,2,2}));
	}
}
