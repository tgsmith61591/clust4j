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

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;

import static com.clust4j.TestSuite.getRandom;

import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.DBSCANParameters;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.data.DataSet;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.kernel.Kernel;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class DBSCANTests implements ClusterTest, ClassifierTest, BaseModelTest {
	final DataSet irisds = TestSuite.IRIS_DATASET.copy();
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
		new DBSCAN(data, new DBSCANParameters(0.5));
	}

	@Test
	@Override
	public void testFit() {
		new DBSCANParameters().fitNewModel(data).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new DBSCANParameters().fitNewModel(data);
		
	}

	@Test
	public void DBSCANTest1() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		StandardScaler scaler = new StandardScaler().fit(mat);
		
		new DBSCAN(scaler.transform(mat), 
			new DBSCANParameters(0.05)
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
		StandardScaler scaler = new StandardScaler().fit(mat);
			
			
		assertTrue(Distance.EUCLIDEAN.getDistance(train_array[1], train_array[2]) > 0.5);
		DBSCAN db = new DBSCAN(scaler.transform(mat), 
			new DBSCANParameters(0.75)
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
		StandardScaler scaler = new StandardScaler().fit(mat);
			
			
		assertTrue(Distance.EUCLIDEAN.getDistance(train_array[1], train_array[2]) > 0.5);
		DBSCAN db = new DBSCAN(scaler.transform(mat), 
			new DBSCANParameters(0.75)
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
			final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
			new DBSCAN(mat, new DBSCANParameters()
					.setVerbose(true))
				.fit();
			System.out.println();
		} catch(OutOfMemoryError | StackOverflowError e) { // swallow, move along
		}
	}
	
	@Test
	public void DBSCANKernelTest1() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		StandardScaler scaler = new StandardScaler().fit(mat);
		
		Kernel kernel = new RadialBasisKernel(0.05);
		DBSCAN db = new DBSCAN(scaler.transform(mat), 
				new DBSCANParameters(0.05)
					.setMetric(kernel)
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
			new DBSCANParameters(0.75)
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
			new DBSCANParameters().setMinPts(0));
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
		
		int[] labels = new DBSCAN(X, new DBSCANParameters(1).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, labels[0]).all()); // these might be noise in DBSCAN
		
		labels = new DBSCAN(X, new DBSCANParameters().setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, labels[0]).all());
	}
	
	@Test
	public void testNoSimilaritiesAllowed() {
		DBSCAN model;
		for(Kernel k: KernelTestCases.all_kernels) {
			model = new DBSCAN(data, new DBSCANParameters().setMetric(k)).fit();
			assertTrue(model.hasWarnings());
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
		}
	}
	
	@Test
	public void testValidMetrics() {
		DBSCAN model;
		
		for(Distance d: Distance.values()) {
			model = new DBSCAN(data, new DBSCANParameters().setMetric(d)).fit();
			assertTrue(model.dist_metric.equals(d)); // assert not internally changed.
		}
		
		DistanceMetric d= new MinkowskiDistance(1.5);
		model = new DBSCAN(data, new DBSCANParameters().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert not internally changed.
		
		/*
		 * Now haversine...
		 */
		final Array2DRowRealMatrix small = TestSuite.IRIS_SMALL.getData();
		
		d = Distance.HAVERSINE.MI;
		model = new DBSCAN(small, new DBSCANParameters().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert not internally changed.
		

		d = Distance.HAVERSINE.KM;
		model = new DBSCAN(small, new DBSCANParameters().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert not internally changed.
	}
	
	@Test
	public void testBadEps() {
		boolean a = false;
		try {
			new DBSCAN(data, new DBSCANParameters(0.0));
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testForceBadMetric() {
		DBSCAN d = new DBSCAN(data);
		assertFalse(d.isValidMetric(new GaussianKernel()));
	}
	
	@Test
	public void testGetter() {
		DBSCAN d = new DBSCAN(data, new DBSCANParameters(1.5));
		assertTrue(d.getEps() == 1.5);
		
		/*
		 * Test mnfe for labels
		 */
		boolean a = false;
		try {
			d.getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testEqualsAndMultiFit() {
		DBSCAN d = new DBSCAN(data).fit();
		assertTrue(d.equals(d.fit()));
		assertFalse(d.equals(new DBSCAN(data))); // second has been fit yet
		
		DBSCAN e = new DBSCAN(data).fit();
		
		// if the key is equal, pass, otherwise full equals compare
		assertTrue(d.getKey().equals(e.getKey()) || !d.equals(e));
		assertFalse(d.equals(new Object()));
	}
	
	@Test
	public void testPredict() {
		DBSCAN d = new DBSCANParameters(1.5).fitNewModel(data);
		
		/*
		 * Test on exact records and one noisey
		 */
		Array2DRowRealMatrix newData = new Array2DRowRealMatrix(new double[][]{
			d.data.getRow(0),
			d.data.getRow(149),
			new double[]{150,150,150,150}
		}, false);
		
		int[] predicted = d.predict(newData);
		assertTrue(VecUtils.equalsExactly(predicted, new int[]{0, 1,-1}));
		
		
		/*
		 * Test on all noisey
		 */
		newData = new Array2DRowRealMatrix(new double[][]{
			new double[]{150,150,150,150},
			new double[]{ 0, 0, 0, 0},
			new double[]{ 3, 3, 3, 3},
			new double[]{ 3, 1, 2, 2},
			new double[]{ 5, 5, 5, 5},
			new double[]{-5,-5,-5,-5}
		}, false);
		
		predicted = d.predict(newData);
		assertTrue(VecUtils.equalsExactly(predicted, new int[]{-1,-1,-1,-1,-1,-1}));
		
		/*
		 * Test for dim mismatch
		 */
		newData = new Array2DRowRealMatrix(new double[][]{
			new double[]{150,150,150,150,150}
		}, false);
		boolean a = false;
		try {
			d.predict(newData);
		} catch(DimensionMismatchException dim) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
}
