package com.clust4j.data;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.metrics.scoring.SupervisedMetric;

public class TrainTestSplitTests {

	@Test
	public void testIris() {
		TrainTestSplit split = new TrainTestSplit(TestSuite.IRIS_DATASET, 0.7);
		
		DataSet train = split.getTrain();
		DataSet test = split.getTest();
		
		assertTrue(train.numRows() + test.numRows() == TestSuite.IRIS_DATASET.numRows());
	}

	@Test
	public void testLowerBoundWithLabels() {
		DataSet set = new DataSet(
			new Array2DRowRealMatrix(new double[][]{
				new double[]{0,0,0},
				new double[]{1,1,1}
			}, false),
			new int[]{0,0}
		);
		
		TrainTestSplit split = new TrainTestSplit(set, 0.8);
		assertTrue(split.getTrain().numRows() == 1);
		assertTrue(split.getTest().numRows() == 1);
		
		split = new TrainTestSplit(set, 0.1);
		assertTrue(split.getTrain().numRows() == 1);
		assertTrue(split.getTest().numRows() == 1);
	}
	
	@Test
	public void testLowerBoundWithNoLabels() {
		int[] labels = null;
		DataSet set = new DataSet(
			new Array2DRowRealMatrix(new double[][]{
				new double[]{0,0,0},
				new double[]{1,1,1}
			}, false),
			
			labels // null
		);
		
		TrainTestSplit split = new TrainTestSplit(set, 0.8);
		assertTrue(split.getTrain().numRows() == 1);
		assertTrue(split.getTest().numRows() == 1);
		
		split = new TrainTestSplit(set, 0.1);
		assertTrue(split.getTrain().numRows() == 1);
		assertTrue(split.getTest().numRows() == 1);
	}
	
	@Test
	public void testExceptions() {
		DataSet set = new DataSet(
			new Array2DRowRealMatrix(new double[][]{
				new double[]{0,0,0}
			}, false),
			new int[]{1}
		);
		
		/*
		 * Test one row fails
		 */
		boolean a= false;
		try {
			new TrainTestSplit(set, 0.5);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		
		// re-assign
		set = new DataSet(
			new Array2DRowRealMatrix(new double[][]{
				new double[]{0,0,0},
				new double[]{1,1,1}
			}, false),
			new int[]{1,1}
		);
		
		/*
		 * test fails on 1.0+
		 */
		a= false;
		try {
			new TrainTestSplit(set, 1.0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a= false;
		try {
			new TrainTestSplit(set, 1.1);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * test fails on 0.0-
		 */
		a= false;
		try {
			new TrainTestSplit(set, 0.0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a= false;
		try {
			new TrainTestSplit(set,-0.1);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testOnModel() {
		TrainTestSplit split = new TrainTestSplit(TestSuite.IRIS_DATASET, 0.75);
		DataSet train = split.getTrain();
		DataSet test  = split.getTest();
		
		KMeans model = new KMeansParameters(3).fitNewModel(train.getData());
		int[] predictions = model.predict(test.getData());
		
		// examine affinity:
		System.out.println("Affinity: " + SupervisedMetric.INDEX_AFFINITY.evaluate(test.getLabels(), predictions));
	}
}
