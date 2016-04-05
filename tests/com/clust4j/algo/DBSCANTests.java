package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;

import static com.clust4j.TestSuite.getRandom;

import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.DBSCAN.DBSCANPlanner;
import com.clust4j.data.DataSet;
import com.clust4j.kernel.Kernel;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.HaversineDistance;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class DBSCANTests implements ClusterTest, ClassifierTest, BaseModelTest {
	final DataSet irisds = TestSuite.IRIS_DATASET;
	final Array2DRowRealMatrix data = irisds.getData();


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
		System.out.println();
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
		System.out.println();
		
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
		System.out.println();
		
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
			System.out.println();
		} catch(OutOfMemoryError | StackOverflowError e) { // swallow, move along
		}
	}
	
	@Test
	public void DBSCANKernelTest1() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		Kernel kernel = new RadialBasisKernel(0.05);
		DBSCAN db = new DBSCAN(mat, 
				new DBSCAN.DBSCANPlanner(0.05)
					.setMetric(kernel)
					.setScale(true)
					.setVerbose(true)).fit();
		System.out.println();
		assertTrue(db.hasWarnings());
		assertTrue(db.equals(db)); // ref equals
		assertFalse(db.equals(new DBSCAN(mat)));
		assertFalse(db.equals(new String("asdf"))); // test against different type
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		DBSCAN db = new DBSCAN(data, 
			new DBSCAN.DBSCANPlanner(0.75)
				.setScale(true)
				.setMinPts(1)
				.setVerbose(true)).fit();
		System.out.println();
		
		int a = db.getNumberOfNoisePoints();
		db.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		DBSCAN db2 = (DBSCAN)DBSCAN.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(a == db2.getNumberOfNoisePoints());
		assertTrue(db.equals(db2));
		Files.delete(TestSuite.path);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMinPtsIAE() {
		new DBSCAN(data, 
			new DBSCANPlanner().setMinPts(0));
	}
	
	@Test
	public void testResults() {
		DBSCAN db = new DBSCAN(data).fit();
		
		// sklearn's result
		final int[] expected = new int[]{
			0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
		    0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
		    1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,
		   -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
		    1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,  1,
		    1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
		    1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1, -1,
		    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
		};
		
		assertTrue(Precision.equals(0.95, db.indexAffinityScore(expected), 0.05));
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
		
		int[] labels = new DBSCAN(X, new DBSCANPlanner(1).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.VecIntSeries(labels, Inequality.EQUAL_TO, labels[0]).all()); // these might be noise in DBSCAN
		
		labels = new DBSCAN(X, new DBSCANPlanner().setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.VecIntSeries(labels, Inequality.EQUAL_TO, labels[0]).all());
	}
	
	@Test
	public void testNoSimilaritiesAllowed() {
		DBSCAN model;
		for(Kernel k: KernelTestCases.all_kernels) {
			model = new DBSCAN(data, new DBSCANPlanner().setMetric(k)).fit();
			assertTrue(model.hasWarnings());
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
		}
	}
	
	@Test
	public void testValidMetrics() {
		DBSCAN model;
		
		for(Distance d: Distance.values()) {
			model = new DBSCAN(data, new DBSCANPlanner().setMetric(d)).fit();
			assertTrue(model.dist_metric.equals(d)); // assert not internally changed.
		}
		
		DistanceMetric d= new MinkowskiDistance(1.5);
		model = new DBSCAN(data, new DBSCANPlanner().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert not internally changed.
		
		/*
		 * Now haversine...
		 */
		d = new HaversineDistance();
		
		model = new DBSCAN(TestSuite.IRIS_SMALL.getData(), new DBSCANPlanner().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert not internally changed.
	}
}
