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
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.KMedoidsParameters;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.data.DataSet;
import com.clust4j.kernel.HyperbolicTangentKernel;
import com.clust4j.kernel.Kernel;
//import com.clust4j.kernel.KernelTestCases;
import com.clust4j.kernel.LaplacianKernel;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class KMedoidsTests implements ClusterTest, ClassifierTest, ConvergeableTest, BaseModelTest {
	final Array2DRowRealMatrix irisdata = TestSuite.IRIS_DATASET.getData();
	final Array2DRowRealMatrix winedata = TestSuite.WINE_DATASET.getData();
	final Array2DRowRealMatrix bcdata = TestSuite.BC_DATASET.getData();
	
	/**
	 * This is the method as it is used in the KMedoids class,
	 * except that the distance matrix is passed in
	 * @param indices
	 * @param med_idx
	 * @return
	 */
	protected static double getCost(ArrayList<Integer> indices, final int med_idx, final double[][] dist_mat) {
		double cost = 0;
		for(Integer idx: indices)
			cost += dist_mat[FastMath.min(idx, med_idx)][FastMath.max(idx, med_idx)];
		return cost;
	}
	
	
	@Test
	public void test() {
		final double[][] distanceMatrix = new double[][] {
			new double[]{0,1,2,3},
			new double[]{0,0,1,2},
			new double[]{0,0,0,1},
			new double[]{0,0,0,0}
		};
		
		final int med_idx = 2;
		
		final ArrayList<Integer> belonging = new ArrayList<Integer>();
		belonging.add(0); belonging.add(1); belonging.add(2); belonging.add(3);
		assertTrue(getCost(belonging, med_idx, distanceMatrix) == 4);
	}

	@Test
	@Override
	public void testItersElapsed() {
		assertTrue(new KMedoids(irisdata).fit().itersElapsed() > 0);
	}


	@Test
	@Override
	public void testConverged() {
		assertTrue(new KMedoids(irisdata).fit().didConverge());
	}


	@Test
	@Override
	public void testScoring() {
		new KMedoids(irisdata).fit().silhouetteScore();
	}


	@Test
	@Override
	public void testDefConst() {
		new KMedoids(irisdata);
	}


	@Test
	@Override
	public void testArgConst() {
		new KMedoids(irisdata, 3);
	}


	@Test
	@Override
	public void testPlannerConst() {
		new KMedoids(irisdata, new KMedoidsParameters());
		new KMedoids(irisdata, new KMedoidsParameters(3));
	}


	@Test
	@Override
	public void testFit() {
		new KMedoids(irisdata, new KMedoidsParameters()).fit();
		new KMedoids(irisdata, new KMedoidsParameters(3)).fit();
	}


	@Test
	@Override
	public void testFromPlanner() {
		new KMedoidsParameters().fitNewModel(irisdata);
		new KMedoidsParameters(3).fitNewModel(irisdata);
	}

	/** Scale = false */
	@Test
	public void KMedoidsTest1() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMedoids km = new KMedoidsParameters(2).setVerbose(true).fitNewModel(mat);
		assertTrue(km.getSeparabilityMetric().equals(Distance.MANHATTAN));
		
		km.fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
		//km.info("testing the k-medoids logger");
	}
	
	/** Scale = true */
	@Test
	public void KMedoidsTest2() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002},
			new double[] {0.015, 	 0.161352,  0.1173},
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		StandardScaler scaler = new StandardScaler().fit(mat);
		RealMatrix X = scaler.transform(mat);
		
		KMedoids km = new KMedoids(X, 
				new KMedoidsParameters(2)
					.setVerbose(true));
		km.fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1 && km.getLabels()[3] == 0);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		
		// test re-fit
		km.fit();
		
		assertTrue(km.getLabels()[0] == km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = false and multiclass */
	@Test
	public void KMedoidsTest3() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMedoids km = new KMedoids(mat, 
				new KMedoidsParameters(3)
					.setVerbose(true));
		km.fit();

		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1 && km.getLabels()[3] == 2);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = true and multiclass */
	@Test
	public void KMedoidsTest4() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.2801,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		StandardScaler scaler = new StandardScaler().fit(mat);
		RealMatrix X = scaler.transform(mat);
		
		KMedoids km = new KMedoids(X, new KMedoidsParameters(3));
		km.fit();
		
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
	}
	
	/** Now scale = true and multiclass */
	@Test
	public void KMedoidsTest4_5() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.2801,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		KMedoids km = new KMedoids(mat, new KMedoidsParameters(3));
		km.fit();
		
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.getLabels()[0] != km.getLabels()[3]);
		assertTrue(km.didConverge());
		
		System.out.println(Arrays.toString(km.getCentroids().get(1)));
		assertTrue( VecUtils.equalsExactly(km.getCentroids().get(0), data[0]) );
		assertTrue( VecUtils.equalsExactly(km.getCentroids().get(1), data[1]) );
		assertTrue( VecUtils.equalsExactly(km.getCentroids().get(2), data[3]) );
	}
	
	// What if k = 1??
	@Test
	public void KMedoidsTest5() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.0001,    1.8900002},
			new double[] {100,       200,       100}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		StandardScaler scaler = new StandardScaler().fit(mat);
		RealMatrix X = scaler.transform(mat);
		final boolean[] scale = new boolean[]{false, true};
		
		KMedoids km = null;
		for(boolean b : scale) {
			km = new KMedoids(b ? X : mat, new KMedoidsParameters(1));
			km.fit();
			assertTrue(km.didConverge());
		}
	}
	
	@Test
	public void KMedoidsLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		StandardScaler scaler = new StandardScaler().fit(mat);
		RealMatrix X = scaler.transform(mat);
		
		final boolean[] scale = new boolean[] {false, true};
		final int[] ks = new int[] {1,3,5};
		
		KMedoids km = null;
		for(boolean b : scale) {
			for(int k : ks) {
				km = new KMedoids(b ? X : mat, 
						new KMedoidsParameters(k)
							.setVerbose(true));
				km.fit();
			}
		}
	}

	@Test
	public void KMedoidsLoadTest2FullLogger() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		KMedoids km = new KMedoids(mat, 
				new KMedoidsParameters(5)
					.setVerbose(true)
				);
		km.fit();
	}
	
	@Test
	public void KernelKMedoidsLoadTest1() {
		final Array2DRowRealMatrix mat = getRandom(500, 10); // need to reduce size for travis CI
		final int[] ks = new int[] {1,3,5,7};
		Kernel kernel = new LaplacianKernel(0.05);
		
		KMedoids km = null;
		for(int k : ks) {
			km = new KMedoids(mat, 
					new KMedoidsParameters(k)
						.setMetric(kernel)
						.setVerbose(true));
			km.fit();
		}
		System.out.println();
	}
	
	@Test
	public void KernelKMedoidsLoadTest2() {
		final Array2DRowRealMatrix mat = getRandom(500, 10); // need to reduce size for travis CI
		final int[] ks = new int[] {12};
		Kernel kernel = new HyperbolicTangentKernel(); //SplineKernel();
		
		for(int k : ks) {
			new KMedoids(mat, 
				new KMedoidsParameters(k)
					.setMetric(kernel)
					.setVerbose(true)).fit();
		}
		System.out.println();
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		KMedoids km = new KMedoids(irisdata,
			new KMedoidsParameters(3)
				.setVerbose(true)).fit();
		
		final double c = km.getTSS();
		km.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		KMedoids km2 = (KMedoids)KMedoids.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(km2.getTSS() == c);
		assertTrue(km2.equals(km));
		Files.delete(TestSuite.path);
	}
	
	@Test
	public void findBestDistMetric() {
		DataSet ds = TestSuite.IRIS_DATASET.shuffle();
		final Array2DRowRealMatrix d = ds.getData();
		final int[] actual = ds.getLabels();
		GeometricallySeparable best = null;
		double ia = 0;
		KMedoids kmed;
		
		// it's not linearly separable, so most won't perform incredibly well...
		for(DistanceMetric dist: Distance.values()) {
			if(KMedoids.UNSUPPORTED_METRICS.contains(dist.getClass()))
				continue;
			
			KMedoidsParameters km = new KMedoidsParameters(3).setMetric(dist);
			double i = -1;
			
			kmed = km.fitNewModel(d);
			i = kmed.indexAffinityScore(actual);
			
			System.out.println(kmed.dist_metric.getName() + ", " + i);
			if(i > ia) {
				ia = i;
				best = dist;
			}
		}
		
		
		// ALWAYS converges
		System.out.println(best);
	}
	
	@Test
	public void findBestKernelMetric() {
		DataSet ds = TestSuite.IRIS_DATASET.shuffle();
		final Array2DRowRealMatrix d = ds.getData();
		final int[] actual = ds.getLabels();
		GeometricallySeparable best = null;
		double ia = 0;
		
		// it's not linearly separable, so most won't perform incredibly well...
		KMedoids model;
		for(Kernel dist: com.clust4j.kernel.KernelTestCases.all_kernels) {
			if(KMedoids.UNSUPPORTED_METRICS.contains(dist.getClass()))
				continue;
			
			System.out.println(dist);
			KMedoidsParameters km = new KMedoidsParameters(3).setMetric(dist);
			double i = -1;
			
			model = km.fitNewModel(d).fit();
			if(model.getK() != 3) // gets modified if totally equal
				continue;
			
			i = model.indexAffinityScore(actual);
			

			System.out.println(model.getSeparabilityMetric().getName() + ", " + i);
			if(i > ia) {
				ia = i;
				best = model.getSeparabilityMetric();
			}
		}
		
		
		System.out.println("BEST: " + best.getName() + ", " + ia);
	}
	
	/**
	 * Assert that when all of the matrix entries are exactly the same,
	 * the algorithm will still converge, yet produce one label: 0
	 */
	@Override
	@Test
	public void testAllSame() {
		final double[][] x = MatUtils.rep(-1, 3, 3);
		final Array2DRowRealMatrix X = new Array2DRowRealMatrix(x, false);
		
		int[] labels = new KMedoids(X, new KMedoidsParameters(3).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
	}
}
