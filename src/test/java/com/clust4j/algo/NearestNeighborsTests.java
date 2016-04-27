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

import static com.clust4j.TestSuite.getRandom;
import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;
import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.TestSuite;
import com.clust4j.algo.BaseNeighborsModel.NeighborsAlgorithm;
import com.clust4j.algo.Neighborhood;
import com.clust4j.algo.NearestNeighborsParameters;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.metrics.pairwise.Similarity;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class NearestNeighborsTests implements ClusterTest, BaseModelTest {
	final Array2DRowRealMatrix data = TestSuite.IRIS_DATASET.getData();
	
	@Test
	@Override
	public void testDefConst() {
		new NearestNeighbors(data);
	}

	@Test
	@Override
	public void testArgConst() {
		new NearestNeighbors(data, 3);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new NearestNeighbors(data, new NearestNeighborsParameters());
	}

	@Test
	@Override
	public void testFit() {
		new NearestNeighbors(data).fit();
		new NearestNeighbors(data).fit().fit(); // test the extra fit for any exceptions
		new NearestNeighbors(data, 3).fit();
		new NearestNeighbors(data, new NearestNeighborsParameters()).fit();
		new NearestNeighbors(data, new NearestNeighborsParameters(5)).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new NearestNeighborsParameters().fitNewModel(data);
		new NearestNeighborsParameters(4).fitNewModel(data);
	}
	
	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		NearestNeighbors nn = new NearestNeighbors(data, 
			new NearestNeighborsParameters(5)
				.setVerbose(true)).fit();
		
		final int[][] c = nn.getNeighbors().getIndices();
		nn.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		NearestNeighbors nn2 = (NearestNeighbors)NearestNeighbors.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(MatUtils.equalsExactly(nn2.getNeighbors().getIndices(), c));
		assertTrue(nn2.equals(nn));
		assertTrue(nn.equals(nn)); // test the ref return
		assertFalse(nn.equals(new Object()));
		
		Files.delete(TestSuite.path);
	}

	@Test
	public void NN_KNEAREST_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		
		final int[] ks = new int[]{1, 5, 10};
		for(int k: ks) {
			new NearestNeighbors(mat, 
				new NearestNeighborsParameters(k)
					.setVerbose(true)).fit();
		}
	}
	
	@Test
	public void NN_RADIUS_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		
		final double[] radii = new double[]{0.5, 5.0, 10.0};
		for(double radius: radii) {
			new RadiusNeighbors(mat, 
				new RadiusNeighborsParameters(radius)
					.setVerbose(true)).fit();
			System.out.println();
		}
	}
	
	@Test
	public void NNTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  2.0,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		NearestNeighbors nn = new NearestNeighbors(mat, 
			new NearestNeighborsParameters(1)
				.setVerbose(true)
				.setMetric(new GaussianKernel())).fit();
		
		int[][] ne = nn.getNeighbors().getIndices();
		assertTrue(ne[0].length == 1);
		assertTrue(ne[0].length == 1);
		System.out.println();
		
		new RadiusNeighbors(mat, 
			new RadiusNeighborsParameters(3.0)
				.setVerbose(true)
				.setMetric(new GaussianKernel()) ).fit();
	}
	
	@Test
	public void NN_kernel_KNEAREST_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		
		final int[] ks = new int[]{1, 5, 10};
		for(int k: ks) {
			new NearestNeighbors(mat, 
				new NearestNeighborsParameters(k)
					.setVerbose(true)
					.setMetric(new GaussianKernel()) ).fit();
		}
	}
	
	@Test
	public void NN_kernel_RADIUS_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		
		final double[] radii = new double[]{0.5, 5.0, 10.0};
		for(double radius: radii) {
			new RadiusNeighbors(mat, 
				new RadiusNeighborsParameters(radius)
					.setVerbose(true)
					.setMetric(new GaussianKernel()) ).fit();
			System.out.println();
		}
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE1() {
		new NearestNeighbors(data, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE2() {
		new NearestNeighbors(data, 151);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE3() {
		new NearestNeighbors(data, new NearestNeighborsParameters(0));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE4() {
		new NearestNeighbors(data, new NearestNeighborsParameters(151));
	}
	
	@Test
	public void testK() {
		assertTrue(new NearestNeighbors(data, 3).getK() == 3);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFNeigb1() {
		new NearestNeighbors(data).getNeighbors(data);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFNeigb2() {
		new NearestNeighbors(data).getNeighbors(data, 2);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFNeigb3() {
		new NearestNeighbors(data).getNeighbors();
	}
	
	@Test
	public void testGetNeighb() {
		new NearestNeighbors(data).fit().getNeighbors().getDistances();
		new NearestNeighbors(data).fit().getNeighbors(data).getDistances();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testGetNeighbIAE1() {
		new NearestNeighbors(data).fit().getNeighbors(data, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testGetNeighbIAE2() {
		new NearestNeighbors(data).fit().getNeighbors(data, 151);
	}
	
	final static Array2DRowRealMatrix DATA=
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
			new NearestNeighbors(DATA, 
				new NearestNeighborsParameters(1)
					.setVerbose(true)
					.setAlgorithm(alg)
					.setLeafSize(3)
					.setSeed(new Random())
					.setMetric(Distance.RUSSELL_RAO) ).fit();
		}
	}
	
	@Test
	public void testWarning() {
		BaseNeighborsModel n = new NearestNeighbors(DATA, 
			new NearestNeighborsParameters(1)
				.setMetric(new GaussianKernel()));
		assertTrue(n.hasWarnings());
	}
	
	@Test
	public void testMasking() {
		int[][] indices = new int[][]{
			new int[]{1,2,3},
			new int[]{4,5,6},
			new int[]{7,8,9}
		};
		
		// Set up sample range
		int[] sampleRange = new int[]{1,2,8};
		int m = indices.length, ni = indices[0].length;
		
		
		boolean allInRow, bval;
		boolean[] dupGroups = new boolean[m];
		boolean[][] sampleMask= new boolean[m][ni];
		for(int i = 0; i < m; i++) {
			allInRow = true;
			
			for(int j = 0; j < ni; j++) {
				bval = indices[i][j] != sampleRange[i];
				sampleMask[i][j] = bval;
				allInRow &= bval;
			}
			
			dupGroups[i] = allInRow;
		}
		
		assertTrue(MatUtils.equalsExactly(sampleMask, new boolean[][]{
			new boolean[]{false, true,  true},
			new boolean[]{true,  true,  true},
			new boolean[]{true,  false, true},
		}));
		
		assertTrue(VecUtils.equalsExactly(dupGroups, 
			new boolean[]{false, true, false}));
		
		// Test de-dup cornercase
		for(int i = 0; i < m; i++)
			if(dupGroups[i])
				sampleMask[i][0] = false;
		
		assertTrue(MatUtils.equalsExactly(sampleMask, new boolean[][]{
			new boolean[]{false, true,  true},
			new boolean[]{false, true,  true},
			new boolean[]{true,  false, true},
		}));
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
			
			int k= 2;
			NearestNeighbors nn = new NearestNeighbors(x, new NearestNeighborsParameters(k).setAlgorithm(algo)).fit();
			
			assertTrue(MatUtils.equalsExactly(expected, nn.fit_X));
			
			Neighborhood n1 = nn.getNeighbors();
			double[][] d1 = new double[][]{
				new double[]{5.196152422706632, 10.392304845413264},
				new double[]{5.196152422706632, 5.196152422706632 },
				new double[]{5.196152422706632, 10.392304845413264}
			};
			
			int[][] i1 = new int[][]{
				new int[]{1,2},
				new int[]{2,0},
				new int[]{1,0}
			};
			assertTrue(MatUtils.equalsExactly(d1, n1.getDistances()));
			assertTrue(MatUtils.equalsExactly(i1, n1.getIndices()));
			
			Neighborhood n2 = nn.getNeighbors(x);
			double[][] d2 = new double[][]{
				new double[]{0.0, 5.196152422706632},
				new double[]{0.0, 5.196152422706632},
				new double[]{0.0, 5.196152422706632}
			};
			
			// Test the toString() method for total coverage:
			String n2s = n2.toString();
			assertTrue(n2s.startsWith("Distances"));
			
			int[][] i2 = new int[][]{
				new int[]{0,1},
				new int[]{1,0},
				new int[]{2,1}
			};
			assertTrue(MatUtils.equalsExactly(d2, n2.getDistances()));
			assertTrue(MatUtils.equalsExactly(i2, n2.getIndices()));
			
			assertTrue(nn.getK() == 2);
			
			Neighborhood n3 = nn.getNeighbors(x, 1);
			double[][] d3 = new double[][]{
				new double[]{0.0},
				new double[]{0.0},
				new double[]{0.0}
			};
			
			int[][] i3 = new int[][]{
				new int[]{0},
				new int[]{1},
				new int[]{2}
			};
			
			assertTrue(MatUtils.equalsExactly(d3, n3.getDistances()));
			assertTrue(MatUtils.equalsExactly(i3, n3.getIndices()));
		}
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit1() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 1);
		nn.getNeighbors();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit2() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 1);
		nn.getNeighbors(DATA);
	}
	
	@Test
	public void testFromPlanner2() {
		NearestNeighbors nn = new NearestNeighborsParameters(1)
			.setAlgorithm(BaseNeighborsModel.NeighborsAlgorithm.BALL_TREE)
			.setLeafSize(40)
			.setSeed(new Random())
			.setMetric(new GaussianKernel())
			.setVerbose(false).copy().fitNewModel(data);
		
		assertTrue(nn.hasWarnings()); // Sep method
		assertTrue(nn.getK() == 1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor1() {
		new NearestNeighbors(DATA, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor2() {
		new NearestNeighbors(DATA, 8);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor3() {
		new NearestNeighbors(DATA, -1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor4() {
		new NearestNeighbors(DATA, new NearestNeighborsParameters(2).setLeafSize(-1));
	}
	
	@Test(expected=NullPointerException.class)
	public void testNPEConstructor1() {
		new NearestNeighbors(DATA, new NearestNeighborsParameters(2).setAlgorithm(null));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod1() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 2).fit();
		nn.getNeighbors(DATA, 9);
		// test refit
		nn.fit();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod2() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 2).fit();
		nn.getNeighbors(DATA, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod3() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 2).fit();
		nn.getNeighbors(DATA, -1);
	}
	
	@Test
	public void NNTest1_ap() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  2.0,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		BaseNeighborsModel nn = new NearestNeighbors(mat, 
			new NearestNeighborsParameters(1)
				.setVerbose(false)).fit();
		
		int[][] ne = nn.getNeighbors().getIndices();
		assertTrue(ne[0].length == 1);
		assertTrue(ne[0].length == 1);
		assertTrue(ne[0][0] == 1);
		assertTrue(ne[1][0] == 0);
		assertTrue(ne[2][0] == 1);
		System.out.println();
		
		nn = new RadiusNeighbors(mat, 
			new RadiusNeighborsParameters(3.0)
				.setVerbose(false)).fit();
		
		ne = nn.getNeighbors().getIndices();
		assertTrue(ne[0].length == 1);
		assertTrue(ne[1].length == 1);
		assertTrue(ne[2].length == 0);
	}
	
	@Test
	public void testBigWithParallelQuery() {
		final int k= 3;
		final Array2DRowRealMatrix big = TestSuite.getRandom(500, k); // need to reduce size for travis CI
		NearestNeighbors nn;
		try {
			nn = new NearestNeighbors(big, 
				new NearestNeighborsParameters(3)
					.setVerbose(true)
					.setForceParallel(true)).fit();
		} catch(OutOfMemoryError | RejectedExecutionException e) {
			// don't propagate these...
			return;
		}
		
		// test query now...
		//final Neighborhood res = nn.tree.query(big.getData(), k, BaseNeighborsModel.DUAL_TREE_SEARCH, BaseNeighborsModel.SORT);
		//final Neighborhood par = 
		nn.getNeighbors(big.getDataRef(), true);
		
		//assertTrue(res.equals(par));
	}
	
	@Test
	public void testSorted() {
		final Array2DRowRealMatrix big = TestSuite.getRandom(50, 3);
		Neighborhood n = new NearestNeighbors(big, new NearestNeighborsParameters(3)
			.setVerbose(true)).fit().getNeighbors();
		
		for(double[] d : n.getDistances()) {
			assertTrue(VecUtils.max(d) == d[d.length - 1]);
		}
	}
	
	/**
	 * Assert that when all of the matrix entries are exactly the same,
	 * the algorithm will still converge, yet produce one label: 0
	 */
	@Override
	@Test
	public void testAllSame() {
		final double[][] x = MatUtils.rep(-1, 6, 6);
		final Array2DRowRealMatrix X = new Array2DRowRealMatrix(x, false);
		
		Neighborhood neighb = new NearestNeighbors(X, new NearestNeighborsParameters(3).setVerbose(true)).fit().getNeighbors();
		assertTrue(new MatUtils.MatSeries(neighb.getDistances(), Inequality.EQUAL_TO, 0).all());
		System.out.println();
		
		/*
		 * Test def constructor
		 */
		neighb = new NearestNeighbors(X, new NearestNeighborsParameters().setVerbose(true)).fit().getNeighbors();
		assertTrue(new MatUtils.MatSeries(neighb.getDistances(), Inequality.EQUAL_TO, 0).all());
		System.out.println();
		
		/*
		 * Test BallTree
		 */
		neighb = new NearestNeighbors(X, new NearestNeighborsParameters().setVerbose(true).setAlgorithm(NeighborsAlgorithm.BALL_TREE)).fit().getNeighbors();
		assertTrue(new MatUtils.MatSeries(neighb.getDistances(), Inequality.EQUAL_TO, 0).all());
		System.out.println();
	}
	
	@Test
	public void testValidMetrics() {
		NearestNeighbors model;
		final int nn = 3;
		final NearestNeighborsParameters planner = new NearestNeighborsParameters(nn);
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
			assertFalse(model.isValidMetric(new GaussianKernel()));
		}
	}
	
	@Test
	public void testUnsupportedOperation() {
		boolean a = false;
		try {
			BaseNeighborsModel.NeighborsAlgorithm.AUTO.isValidMetric(null);
		} catch(UnsupportedOperationException u) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test //travis may not be able to handle this...
	public void testSmallParallelJob() {
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
			Neighborhood n1 = new NearestNeighbors(a, new NearestNeighborsParameters(2).setForceParallel(true)).fit().getNeighbors();
			Neighborhood n2 = new NearestNeighbors(a, new NearestNeighborsParameters(2).setForceParallel(false)).fit().getNeighbors();
			assertTrue(n1.equals(n2));
		} finally {
			/*
			 * Reset
			 */
			GlobalState.ParallelismConf.PARALLELISM_ALLOWED = orig;
		}
	}
	
	@Test
	public void testCallerConstructors() {
		KMeans km = new KMeansParameters(3).fitNewModel(data);
		new NearestNeighbors(km).fit();
		new NearestNeighbors(km, new NearestNeighborsParameters(2)).fit();
	}
}
