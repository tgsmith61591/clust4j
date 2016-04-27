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
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.BaseNeighborsModel.NeighborsAlgorithm;
import com.clust4j.algo.Neighborhood;
import com.clust4j.algo.RadiusNeighborsParameters;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.metrics.pairwise.Similarity;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.Series.Inequality;

public class RadiusNeighborsTests implements ClusterTest, BaseModelTest {
	final static Array2DRowRealMatrix iris = TestSuite.IRIS_DATASET.getData();
	
	final static Array2DRowRealMatrix data=
		new Array2DRowRealMatrix(new double[][]{
			new double[]{0.0,0.1,0.2},
			new double[]{2.3,2.5,3.1},
			new double[]{2.0,2.6,3.0},
			new double[]{0.3,0.2,0.1}
		}, false);

	@Test
	public void testWithVerbose() {
		NeighborsAlgorithm[] algs = new NeighborsAlgorithm[]{NeighborsAlgorithm.BALL_TREE, NeighborsAlgorithm.KD_TREE};
		
		for(NeighborsAlgorithm alg: algs) {
			new RadiusNeighbors(data, 
				new RadiusNeighborsParameters(1.0)
						.setVerbose(true)
						.setAlgorithm(alg)
						.setLeafSize(3)
						.setSeed(new Random())
						.setMetric(Distance.RUSSELL_RAO) ).fit();
		}
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testDimMM1() {
		RadiusNeighbors n = new RadiusNeighbors(data, 1.0).fit();
		n.getNeighbors(new Array2DRowRealMatrix(new double[][]{
			new double[]{1,2,3,4},
			new double[]{5,6,7,8}
		}, false));
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testDimMM2() {
		RadiusNeighbors n = new RadiusNeighbors(data, 1.0).fit();
		n.getNeighbors(new Array2DRowRealMatrix(new double[][]{
			new double[]{1,2,3,4},
			new double[]{5,6,7,8}
		}, false), 2.0);
	}
	
	@Test
	public void testWarning() {
		BaseNeighborsModel n = new RadiusNeighbors(data, 
			new RadiusNeighborsParameters(1)
				.setMetric(new GaussianKernel()));
		assertTrue(n.hasWarnings());
	}

	@Test
	public void testFitResults() {
		NeighborsAlgorithm[] algos = new NeighborsAlgorithm[]{NeighborsAlgorithm.KD_TREE, NeighborsAlgorithm.BALL_TREE};
		
		for(NeighborsAlgorithm algo: algos) {
			double[][] expected = new double[][]{
				new double[]{1,2,3},
				new double[]{4,5,6},
				new double[]{7,8,9}
			};
			
			Array2DRowRealMatrix x = new Array2DRowRealMatrix(expected);
			
			double radius = 1.0;
			RadiusNeighbors nn = new RadiusNeighbors(x, 
				new RadiusNeighborsParameters(radius)
					.setAlgorithm(algo)).fit();
			
			assertTrue(MatUtils.equalsExactly(expected, nn.fit_X));
			
			Neighborhood n1 = nn.getNeighbors();
			
			double[][] d1 = new double[][]{
				new double[]{},
				new double[]{},
				new double[]{}
			};
			
			int[][] i1 = new int[][]{
				new int[]{},
				new int[]{},
				new int[]{}
			};
			assertTrue(MatUtils.equalsExactly(d1, n1.getDistances()));
			assertTrue(MatUtils.equalsExactly(i1, n1.getIndices()));
			
			Neighborhood n2 = nn.getNeighbors(x);
			double[][] d2 = new double[][]{
				new double[]{0.0},
				new double[]{0.0},
				new double[]{0.0}
			};
			
			// Test the toString() method for total coverage:
			String n2s = n2.toString();
			assertTrue(n2s.startsWith("Distances"));
			
			int[][] i2 = new int[][]{
				new int[]{0},
				new int[]{1},
				new int[]{2}
			};
			assertTrue(MatUtils.equalsExactly(d2, n2.getDistances()));
			assertTrue(MatUtils.equalsExactly(i2, n2.getIndices()));
			
			assertTrue(nn.getRadius() == 1.0);
			
			Neighborhood n3 = nn.getNeighbors(x, 11.0);
			double[][] d3 = new double[][]{
				new double[]{0.0,                5.196152422706632,  10.392304845413264},
				new double[]{5.196152422706632 , 0.0              ,  5.196152422706632 },
				new double[]{10.392304845413264, 5.196152422706632,  0.0               }
			};
			
			int[][] i3 = new int[][]{
				new int[]{0, 1, 2},
				new int[]{0, 1, 2},
				new int[]{0, 1, 2}
			};
			
			assertTrue(MatUtils.equalsExactly(d3, n3.getDistances()));
			assertTrue(MatUtils.equalsExactly(i3, n3.getIndices()));
		}
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit1() {
		RadiusNeighbors nn = new RadiusNeighbors(data, 1.0);
		nn.getNeighbors();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit2() {
		RadiusNeighbors nn = new RadiusNeighbors(data, 1.0);
		nn.getNeighbors(data);
	}
	
	@Test
	@Override
	public void testFromPlanner() {
		RadiusNeighbors nn = new RadiusNeighborsParameters(1.0)
			.setAlgorithm(BaseNeighborsModel.NeighborsAlgorithm.BALL_TREE)
			.setLeafSize(40)
			.setSeed(new Random())
			.setMetric(new GaussianKernel())
			.setVerbose(false).copy().fitNewModel(data);
		
		assertTrue(nn.hasWarnings()); // Sep method
		assertTrue(nn.getRadius() == 1.0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor1() {
		// Assert 0 is not permissible
		new RadiusNeighbors(data, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor2() {
		new RadiusNeighbors(data, -1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor3() {
		new RadiusNeighbors(data, 
			new RadiusNeighborsParameters(2.0)
				.setLeafSize(-1));
	}
	
	@Test(expected=NullPointerException.class)
	public void testNPEConstructor1() {
		new RadiusNeighbors(data, 
			new RadiusNeighborsParameters(2)
				.setAlgorithm(null));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod1() {
		RadiusNeighbors nn = new RadiusNeighbors(data, 2.0).fit();
		nn.getNeighbors(data, -1.0);
	}

	@Test
	@Override
	public void testDefConst() {
		new RadiusNeighbors(iris);
	}

	@Test
	@Override
	public void testArgConst() {
		new RadiusNeighbors(iris, 2.0);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new RadiusNeighbors(iris, new RadiusNeighborsParameters());
		new RadiusNeighbors(iris, new RadiusNeighborsParameters(6.0));
	}

	@Test
	@Override
	public void testFit() {
		new RadiusNeighbors(iris).fit();
		new RadiusNeighbors(iris).fit().fit(); // test for any other exceptions
		new RadiusNeighbors(iris, 2.0).fit();
		new RadiusNeighbors(iris, new RadiusNeighborsParameters()).fit();
		new RadiusNeighbors(iris, new RadiusNeighborsParameters(6.0)).fit();
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		RadiusNeighbors nn = new RadiusNeighbors(iris, 
			new RadiusNeighborsParameters(5.0)
				.setVerbose(true)).fit();
		
		final int[][] c = nn.getNeighbors().getIndices();
		nn.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		RadiusNeighbors nn2 = (RadiusNeighbors)RadiusNeighbors.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(MatUtils.equalsExactly(nn2.getNeighbors().getIndices(), c));
		assertTrue(nn2.equals(nn));
		assertTrue(nn.equals(nn)); // test the ref return
		assertFalse(nn.equals(new Object()));
		
		Files.delete(TestSuite.path);
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
		
		Neighborhood neighb = new RadiusNeighbors(X, new RadiusNeighborsParameters(3.0).setVerbose(true)).fit().getNeighbors();
		assertTrue(new MatUtils.MatSeries(neighb.getDistances(), Inequality.EQUAL_TO, 0).all());
		System.out.println();
		
		/*
		 * Test default constructor
		 */
		neighb = new RadiusNeighbors(X, new RadiusNeighborsParameters().setVerbose(true)).fit().getNeighbors();
		assertTrue(new MatUtils.MatSeries(neighb.getDistances(), Inequality.EQUAL_TO, 0).all());
		System.out.println();
		
		/*
		 * Test BallTree
		 */
		neighb = new RadiusNeighbors(X, new RadiusNeighborsParameters(3.0).setVerbose(true).setAlgorithm(NeighborsAlgorithm.BALL_TREE)).fit().getNeighbors();
		assertTrue(new MatUtils.MatSeries(neighb.getDistances(), Inequality.EQUAL_TO, 0).all());
		System.out.println();
	}
	
	@Test
	public void testValidMetrics() {
		RadiusNeighbors model;
		final double rad = 3.0;
		final RadiusNeighborsParameters planner = new RadiusNeighborsParameters(rad);
		Array2DRowRealMatrix small= TestSuite.IRIS_SMALL.getData();
		
		/*
		 * For each of AUTO, KD and BALL
		 */
		for(NeighborsAlgorithm na: NeighborsAlgorithm.values()) {
			planner.setAlgorithm(na);
			
			for(Distance d: Distance.values()) {
				planner.setMetric(d);
				model = planner.fitNewModel(data).fit();
				assertTrue(BallTree.VALID_METRICS.contains(model.dist_metric.getClass()));
			}
			
			// minkowski
			planner.setMetric(new MinkowskiDistance(1.5));
			model = planner.fitNewModel(data).fit();
			assertFalse(model.hasWarnings());
			
			// haversine
			planner.setMetric(Distance.HAVERSINE.MI);
			model = planner.fitNewModel(small).fit();
			
			if(na.equals(NeighborsAlgorithm.BALL_TREE)) // else it WILL
				assertFalse(model.hasWarnings());
			
			// try a sim metric...
			planner.setMetric(Similarity.COSINE);
			model = planner.fitNewModel(small).fit();
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
		}
	}
	
	@Test
	public void testRNParallel() {
		/*
		 * Travis CI only has 1 core, so we have to ensure this
		 * will work even on single core machines...
		 */
		RadiusNeighborsParameters planner = new RadiusNeighborsParameters(0.5).setForceParallel(true);
		RadiusNeighbors model = planner.fitNewModel(iris).fit();
		model.getNeighbors(iris.getData(), true);
		model.getNeighbors(iris.getData());
	}
}
