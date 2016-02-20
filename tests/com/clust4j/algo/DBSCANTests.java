package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import static com.clust4j.TestSuite.getRandom;

import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.DBSCAN.DBSCANPlanner;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.kernel.Kernel;
import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.utils.Distance;

public class DBSCANTests implements ClusterTest, ClassifierTest {
	final Array2DRowRealMatrix data = ExampleDataSets.IRIS.getData();


	@Test
	@Override
	public void testScoring() {
		new DBSCAN(data).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testDefConst() {
		new DBSCAN(data);
	}

	@Test
	@Override
	public void testArgConst() {
		new DBSCAN(data, 1.5);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new DBSCAN(data, new DBSCANPlanner(0.5));
	}

	@Test
	@Override
	public void testFit() {
		new DBSCAN.DBSCANPlanner().buildNewModelInstance(data).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new DBSCAN.DBSCANPlanner().buildNewModelInstance(data);
		
	}

	@Test
	public void DBSCANTest1() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		new DBSCAN(mat, new DBSCAN.DBSCANPlanner(0.05)
			.setScale(true)
			.setVerbose(true)).fit();
	}
	
	@Test
	public void DBSCANTest2() {
		final double[][] train_array = new double[][] {
				new double[] {0.00504, 	 0.0001,    0.08172},
				new double[] {3.65816,   2.9471,    3.12331},
				new double[] {4.12344,   3.0001,    2.89002}
			};
			
			final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
			
			
		assertTrue(Distance.EUCLIDEAN.getDistance(train_array[1], train_array[2]) > 0.5);
		DBSCAN db = new DBSCAN(mat, new DBSCAN.DBSCANPlanner(0.75)
			.setScale(true)
			.setMinPts(1)
			.setVerbose(true))
				.fit();
		
		assertTrue(db.getNumberOfIdentifiedClusters() > 0);
		assertTrue(db.getLabels()[1] == db.getLabels()[2]);
	}
	
	@Test
	public void DBSCANTest3() {
		final double[][] train_array = new double[][] {
			new double[] {0.00504, 	 0.0001,    0.08172},
			new double[] {3.65816,   2.9471,    3.12331},
			new double[] {4.12344,   3.0001,    2.89002},
			new double[] {0.00403, 	 0.0003,    0.08231}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
			
			
		assertTrue(Distance.EUCLIDEAN.getDistance(train_array[1], train_array[2]) > 0.5);
		DBSCAN db = new DBSCAN(mat, new DBSCAN.DBSCANPlanner(0.75)
			.setScale(true)
			.setMinPts(1)
			.setVerbose(true))
				.fit();
		
		assertTrue(db.getNumberOfIdentifiedClusters() == 2);
		assertTrue(db.getLabels()[1] == db.getLabels()[2]);
		assertTrue(db.getLabels()[0] == db.getLabels()[3]);
		assertFalse(db.hasWarnings());
		assertTrue(db.getMinPts() == 1);
	}
	
	@Test
	public void DBSCANLoadTest() {
		try {
			final Array2DRowRealMatrix mat = getRandom(5000, 10);
			new DBSCAN(mat, new DBSCAN
				.DBSCANPlanner()
					.setVerbose(true))
				.fit();
		} catch(OutOfMemoryError | StackOverflowError e) {
			return; // Not enough heap space..
		}
	}
	
	@Test
	public void DBSCANKernelTest1() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
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
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		DBSCAN db = new DBSCAN(data, 
			new DBSCAN.DBSCANPlanner(0.75)
				.setScale(true)
				.setMinPts(1)
				.setVerbose(true)).fit();
		
		int a = db.getNumberOfNoisePoints();
		db.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		DBSCAN db2 = (DBSCAN)DBSCAN.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(a == db2.getNumberOfNoisePoints());
		assertTrue(db.equals(db2));
		Files.delete(TestSuite.path);
	}
}
