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
import static com.clust4j.TestSuite.getRandom;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.BaseClustererParameters;
import com.clust4j.algo.DBSCAN;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.except.NaNException;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.Similarity;
import com.clust4j.utils.MatrixFormatter;

public class ClustTests {
	
	private static boolean print = false;
	private static final MatrixFormatter formatter = new MatrixFormatter();
	

	@Test
	public void testFormatter() {
		final double[][] data = new double[][] {
			new double[] {0.0128275, 0.182751, 0.1284},
			new double[] {0.65816,   1.29518,  2.123316},
			new double[] {4.1234,    0.0001,   1.000002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		if(print) System.out.println(formatter.format(mat));
	}

	@Test
	public void mutabilityTest1() {
		final double eps = 0.3;
		final Array2DRowRealMatrix mat = getRandom(5,5);
		StandardScaler scaler = new StandardScaler().fit(mat);
		final double val11 = mat.getEntry(0, 0);
		
		DBSCAN db1 = new DBSCAN(mat, eps); // No scaling
		DBSCAN db2 = new DBSCAN(scaler.transform(mat), eps);
		
		// Testing mutability of scaling
		assertTrue(db1.getData().getEntry(0, 0) == val11);
		assertFalse(db2.getData().getEntry(0, 0) == val11);
	}
	
	
	@Test(expected=NaNException.class)
	public void testNanException() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  Double.NaN,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		new NearestNeighbors(mat, 1);
	}
	
	@Test
	public void testCallerConstructor() {
		KMeans k = new KMeans(TestSuite.IRIS_DATASET.getData());
		AbstractRealMatrix ref = k.data;
		
		/*
		 * Assert that this constructor retains the reference
		 */
		AbstractClusterer a = new AbstractClusterer(k) {
			private static final long serialVersionUID = 1L;
			
			@Override public Algo getLoggerTag() { return null; }
			@Override public String getName() { return null; }
			@Override public boolean isValidMetric(GeometricallySeparable geo) { return false; }
			@Override public AbstractClusterer fit() { return null; }
			@Override protected ModelSummary modelSummary() { return null; }
			@Override protected Object[] getModelFitSummaryHeaders() { return null; }
		};
		
		assertTrue(ref == a.data);
	}
	
	@Test(expected=NaNException.class)
	public void ensureNanException() {
		double[][] d = new double[][]{
			new double[]{1,2,3},
			new double[]{Double.NaN, 0, 1}
		};
		
		new KMeans(toMat(d), 2);
	}
	
	@Test
	public void testHash() {
		double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{-1,0,1}
		};
		
		double[][] b = new double[][]{
			new double[]{-1,29,43},
			new double[]{112,90,21}
		};
		
		KMeans k1 = new KMeans(toMat(a), 1);
		KMeans k2 = new KMeans(toMat(b), new KMeansParameters(1).setMetric(Similarity.COSINE));
		
		assertFalse(k1.hashCode() == k2.hashCode());
		assertFalse(k1.getVerbose());
		assertFalse(k1.hasWarnings());
		assertNull(k1.getWarnings());
		
		/*
		 * Just for coverage love...
		 */
		k1.trace("blah blah");
		k2.debug("blah blah");
	}
	
	static Array2DRowRealMatrix toMat(double[][] d) {
		return new Array2DRowRealMatrix(d, true);
	}
	
	@Test
	public void testAbstractClustererEquals() {
		BaseClustererParameters planner = new BaseClustererParameters(){
			private static final long serialVersionUID = 1L;
			@Override public BaseClustererParameters copy() {return this;}
			@Override public BaseClustererParameters setSeed(Random rand) { return this; }
			@Override public BaseClustererParameters setVerbose(boolean b) { return this; }
			@Override public BaseClustererParameters setMetric(GeometricallySeparable dist) { return this; }
			@Override public BaseClustererParameters setForceParallel(boolean b) { return this; }
		};
		
		AbstractClusterer a = new AbstractClusterer(TestSuite.IRIS_DATASET.getData(), planner){
			private static final long serialVersionUID = 1L;

			@Override public Algo getLoggerTag() {return Algo.CLUST4J;}
			@Override public String getName() {return "anonymous";}
			@Override public boolean isValidMetric(GeometricallySeparable geo) {return true;}
			@Override public AbstractClusterer fit() { return this;}
			@Override protected ModelSummary modelSummary() { return null; }
			@Override protected Object[] getModelFitSummaryHeaders() { return null; }
		};
		
		AbstractClusterer b = new AbstractClusterer(TestSuite.IRIS_DATASET.getData(), planner){
			private static final long serialVersionUID = 1L;

			@Override public Algo getLoggerTag() {return Algo.CLUST4J;}
			@Override public String getName() {return "anonymous";}
			@Override public boolean isValidMetric(GeometricallySeparable geo) {return true;}
			@Override public AbstractClusterer fit() { return this;}
			@Override protected ModelSummary modelSummary() { return null; }
			@Override protected Object[] getModelFitSummaryHeaders() { return null; }
		};
		
		assertTrue(a.equals(a));
		assertTrue(b.equals(b));
		assertFalse(a.equals(b));
		assertFalse(a.equals(new Object()));
	}
}
