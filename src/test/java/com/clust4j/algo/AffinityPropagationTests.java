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

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.TestSuite;
import com.clust4j.algo.AffinityPropagationParameters;
import com.clust4j.data.DataSet;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.kernel.Kernel;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.metrics.pairwise.Similarity;
import com.clust4j.metrics.pairwise.SimilarityMetric;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class AffinityPropagationTests implements ClusterTest, ClassifierTest, ConvergeableTest, BaseModelTest {
	final DataSet irisds = TestSuite.IRIS_DATASET.copy();
	final Array2DRowRealMatrix data = irisds.getData();
	
	@Test
	@Override
	public void testItersElapsed() {
		assertTrue(new AffinityPropagation(data).fit().itersElapsed() > 0);
	}

	@Test
	@Override
	public void testConverged() {
		assertTrue(new AffinityPropagation(data).fit().didConverge());
	}

	@Test
	@Override
	public void testScoring() {
		new AffinityPropagation(data).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testDefConst() {
		new AffinityPropagation(data);
	}

	@Test
	@Override
	public void testArgConst() {
		// Pass -- no such constructor
		return;
	}

	@Test
	@Override
	public void testPlannerConst() {
		new AffinityPropagation(data, new AffinityPropagationParameters());
	}

	@Test
	@Override
	public void testFit() {
		new AffinityPropagation(data).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new AffinityPropagationParameters().fitNewModel(data);
	}

	@Test
	public void AffinityPropTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		final Random seed = new Random(5);
		final boolean[] b = new boolean[]{true, false};
		
		for(boolean bool: b) {
			AffinityPropagation a = 
					new AffinityPropagation(mat, new AffinityPropagationParameters()
							.useGaussianSmoothing(bool)
							.setVerbose(true)
							.setSeed(seed)).fit();
					
					final int[] labels = a.getLabels();
					assertTrue(labels.length == 5);
					assertTrue(labels[0] == labels[1]);
					assertTrue(labels[2] == labels[3]);
					if(bool) assertTrue(a.getNumberOfIdentifiedClusters() == 3);
					assertTrue(a.didConverge());
					assertTrue(labels[0] == 0);
					assertTrue(labels[2] == 1);
					assertTrue(labels[4] == 2);
		}
	}
	
	@Test
	public void AffinityPropLoadTest() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		new AffinityPropagation(mat, new AffinityPropagationParameters()
				.setVerbose(true)).fit();
	}
	
	@Test
	public void AffinityPropKernelTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		final Random seed = new Random(5);
		final boolean[] b = new boolean[]{true, false};
		
		for(boolean bool: b) {
			AffinityPropagation a = 
					new AffinityPropagation(mat, new AffinityPropagationParameters()
							.useGaussianSmoothing(bool)
							.setVerbose(true)
							.setMetric(new GaussianKernel())
							.setSeed(seed)).fit();
					
					final int[] labels = a.getLabels();
					assertTrue(labels.length == 5);
					assertTrue(labels[0] == labels[1]);
					assertTrue(labels[2] == labels[3]);
		}
	}
	
	@Test
	public void AffinityPropKernelLoadTest() {
		final Array2DRowRealMatrix mat = getRandom(400, 10); // need to reduce size for travis CI
		new AffinityPropagation(mat, new AffinityPropagationParameters()
				.setMetric(new GaussianKernel())
				.setVerbose(true)).fit();
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		AffinityPropagation ap = new AffinityPropagation(data, 
			new AffinityPropagationParameters()
					.setVerbose(true)).fit();
		
		double[][] a = ap.getAvailabilityMatrix();
		ap.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		AffinityPropagation ap2 = (AffinityPropagation)AffinityPropagation.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(MatUtils.equalsExactly(a, ap2.getAvailabilityMatrix()));
		assertTrue(ap2.equals(ap));
		Files.delete(TestSuite.path);
	}
	
	@Test
	public void testOnIris() {
		Array2DRowRealMatrix iris = data;
		AffinityPropagation ap = new AffinityPropagation(iris, 
			new AffinityPropagationParameters()
					.setVerbose(true)).fit();
		
		final int[] expected = new LabelEncoder(new int[]{
			1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
			1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0,
			1, 0, 1, 0, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 4, 3, 2,
			3, 4, 3, 4, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 3, 2, 2, 2, 3, 3, 3, 2,
			3, 3, 3, 3, 3, 2, 3, 3, 6, 4, 6, 6, 6, 5, 3, 5, 6, 5, 6, 4, 6, 4, 4,
			6, 6, 5, 5, 4, 6, 4, 5, 4, 6, 6, 4, 4, 6, 6, 5, 5, 6, 4, 4, 5, 6, 6,
			4, 6, 6, 6, 4, 6, 6, 6, 4, 6, 6, 4
		}).fit().getEncodedLabels();
		
		// Assert that the predicted labels are at least 90% in sync with sklearn
		// expected labels (give leeway for random state...)
		assertTrue(Precision.equals(ap.indexAffinityScore(expected), 1.0, 0.1));
	}
	
	@Test
	public void testSimMatFormulation() {
		double[][] X = MatUtils.reshape(VecUtils.asDouble(VecUtils.arange(9)), 3, 3);
		double[][] S = AffinityPropagation
			.computeSmoothedSimilarity(X, Distance.EUCLIDEAN, 
					GlobalState.DEFAULT_RANDOM_STATE, false);
		
		assertTrue(MatUtils.equalsExactly(S, new double[][]{
			new double[]{ -27, -27, -108},
			new double[]{ -27, -27,  -27},
			new double[]{-108, -27,  -27}
		}));
		
		double[][] S_noise = AffinityPropagation
			.computeSmoothedSimilarity(X, Distance.EUCLIDEAN, 
					GlobalState.DEFAULT_RANDOM_STATE, true);
		
		assertTrue(MatUtils.equalsWithTolerance(S_noise, new double[][]{
			new double[]{ -27, -27, -108},
			new double[]{ -27, -27,  -27},
			new double[]{-108, -27,  -27}
		}, 1e-12));
		
		
		
		// ==== DIFF EXAMPLE ====
		
		X = new double[][]{
			new double[]{0.1,0.2,0.3,0.4},
			new double[]{0.2,0.2,0.3,0.1},
			new double[]{12.1,18.1,34,12},
			new double[]{15,23.2,32.1,14}
		};
		
		S_noise = AffinityPropagation
			.computeSmoothedSimilarity(X, Distance.EUCLIDEAN, 
				GlobalState.DEFAULT_RANDOM_STATE, true);
		
		assertTrue(MatUtils.equalsWithTolerance(S_noise, 
			new double[][]{
				new double[]{-8.88345000e+02, -1.00000000e-01, -1.73466000e+03, -1.94721000e+03},
				new double[]{-1.00000000e-01, -8.88345000e+02, -1.73932000e+03, -1.95249000e+03},
				new double[]{-1.73466000e+03, -1.73932000e+03, -8.88345000e+02, -4.20300000e+01},
				new double[]{-1.94721000e+03, -1.95249000e+03, -4.20300000e+01, -8.88345000e+02}
			}, 1e-8));
		
		
		final int m = S_noise.length;
		double[][] A = new double[m][m];
		double[][] R = new double[m][m];
		double[][] tmp = new double[m][m];
		int[] I = new int[m];
		double[] Y = new double[m];
		double[] Y2 = new double[m];
		
		// Performs the work IN PLACE
		AffinityPropagation.affinityPiece1(A, S_noise, tmp, I, Y, Y2);
		
		assertTrue(MatUtils.equalsExactly(A, MatUtils.rep(0.0, m, m)));
		assertTrue(MatUtils.equalsExactly(R, MatUtils.rep(0.0, m, m)));
		assertTrue(VecUtils.equalsExactly(I, new int[]{1,0,3,2}));
		assertTrue(VecUtils.equalsWithTolerance(Y, new double[]{-0.1, -0.1 , -42.03, -42.03}, 1e-12));
		assertTrue(VecUtils.equalsWithTolerance(Y2, new double[]{-888.345, -888.345, -888.345, -888.345}, 1e-12));

		assertTrue(MatUtils.equalsWithTolerance(tmp, 
			new double[][]{
				new double[]{-8.88345000e+02, Double.NEGATIVE_INFINITY, -1.73466000e+03, -1.94721000e+03},
				new double[]{Double.NEGATIVE_INFINITY, -8.88345000e+02, -1.73932000e+03, -1.95249000e+03},
				new double[]{-1.73466000e+03, -1.73932000e+03, -8.88345000e+02, Double.NEGATIVE_INFINITY},
				new double[]{-1.94721000e+03, -1.95249000e+03, Double.NEGATIVE_INFINITY, -8.88345000e+02}
			}, 1e-8));
		
		
		// Performs the work IN PLACE
		double[] colSums = new double[m];
		AffinityPropagation.affinityPiece2(colSums, tmp, I, S_noise, R, Y, Y2, 0.5);
		
		assertTrue(MatUtils.equalsWithTolerance(R, new double[][]{
			new double[]{-444.1225,  444.1225, -867.28  , -973.555 },
			new double[]{ 444.1225, -444.1225, -869.61  , -976.195 },
			new double[]{-846.315 , -848.645 , -423.1575,  423.1575},
			new double[]{-952.59  , -955.23  ,  423.1575, -423.1575}
		}, 1e-12));
		
		assertTrue(MatUtils.equalsWithTolerance(tmp, new double[][]{
			new double[]{-444.1225,  444.1225,    0.    ,    0.    },
			new double[]{ 444.1225, -444.1225,    0.    ,    0.    },
			new double[]{   0.    ,    0.    , -423.1575,  423.1575},
			new double[]{   0.    ,    0.    ,  423.1575, -423.1575}
		}, 1e-12));
		
		
		// Performs the work IN PLACE
		double[] mask = new double[m];
		AffinityPropagation.affinityPiece3(tmp, colSums, A, R, mask, 0.5);
		
		assertTrue(MatUtils.equalsWithTolerance(R, new double[][]{
			new double[]{-444.1225,  444.1225, -867.28  , -973.555 },
			new double[]{ 444.1225, -444.1225, -869.61  , -976.195 },
			new double[]{-846.315 , -848.645 , -423.1575,  423.1575},
			new double[]{-952.59  , -955.23  ,  423.1575, -423.1575}
		}, 1e-12));
		
		assertTrue(MatUtils.equalsWithTolerance(A, new double[][]{
			new double[]{ 2.22061250e+02,  -2.22061250e+02,  -2.84217094e-14,  0.00000000e+00},
			new double[]{-2.22061250e+02,   2.22061250e+02,  -2.84217094e-14,  0.00000000e+00},
			new double[]{-8.52651283e-14,   0.00000000e+00,   2.11578750e+02, -2.11578750e+02},
			new double[]{-8.52651283e-14,   0.00000000e+00,  -2.11578750e+02,  2.11578750e+02}
		}, 1e-12));
		
		assertTrue(MatUtils.equalsWithTolerance(tmp, new double[][]{
			new double[]{-2.22061250e+02,   2.22061250e+02,   2.84217094e-14,  0.00000000e+00},
			new double[]{ 2.22061250e+02,  -2.22061250e+02,   2.84217094e-14,  0.00000000e+00},
			new double[]{ 8.52651283e-14,   0.00000000e+00,  -2.11578750e+02,  2.11578750e+02},
			new double[]{ 8.52651283e-14,   0.00000000e+00,   2.11578750e+02, -2.11578750e+02}
		}, 1e-12));
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
		
		int[] labels = new AffinityPropagation(X, new AffinityPropagationParameters().setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
		System.out.println();
	}
	
	@Test
	public void testValidMetrics() {
		AffinityPropagation model;
		
		for(Distance d: Distance.values()) {
			model = new AffinityPropagation(data, new AffinityPropagationParameters().setMetric(d)).fit();
			assertTrue(model.dist_metric.equals(d)); // assert didn't change
		}
		
		// what about minkowski?
		DistanceMetric d = new MinkowskiDistance(1.5);
		model = new AffinityPropagation(data, new AffinityPropagationParameters().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert didn't change
		
		// Haversine?
		d = Distance.HAVERSINE.KM;
		model = new AffinityPropagation(TestSuite.IRIS_SMALL.getData(), new AffinityPropagationParameters().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert didn't change
		
		d = Distance.HAVERSINE.MI;
		model = new AffinityPropagation(TestSuite.IRIS_SMALL.getData(), new AffinityPropagationParameters().setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(d)); // assert didn't change
		
		
		// Affinity should be able to support basically anything you throw at it, including similarity metrics:
		for(Kernel k: KernelTestCases.all_kernels) {
			model = new AffinityPropagation(data, new AffinityPropagationParameters().setMetric(k)).fit();
			assertTrue(model.dist_metric.equals(k));
		}
		
		// What about cosine similarity?
		SimilarityMetric sim = Similarity.COSINE;
		model = new AffinityPropagation(data, new AffinityPropagationParameters().setMetric(sim)).fit();
		assertTrue(model.dist_metric.equals(sim));
	}
	
	@Test
	public void testPredict() {
		AffinityPropagation a = new AffinityPropagation(data).fit();
		System.out.println("AffinityProp prediction affinity: " + a.indexAffinityScore(a.predict(data)));
	}
	
	@Test
	public void testDamping() {
		/*
		 * Damping must be: 0.5 <= damping < 1
		 */
		boolean a = false, b =  false;
		try {
			new AffinityPropagation(data, new AffinityPropagationParameters().setDampingFactor(0.49));
		} catch(IllegalArgumentException i) {
			a = true;
		}
		
		try {
			new AffinityPropagation(data, new AffinityPropagationParameters().setDampingFactor(1.00));
		} catch(IllegalArgumentException i) {
			b = true;
		}
		
		assertTrue(a && b);
	}
	
	@Test
	public void testMNFE() {
		boolean a = false;
		try {
			new AffinityPropagation(data).getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testNPE1() {
		boolean a = false;
		try {
			new AffinityPropagation(data).getAvailabilityMatrix();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testNPE2() {
		boolean a = false;
		try {
			new AffinityPropagation(data).getResponsibilityMatrix();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testDefaultParamsAndEquals() {
		AffinityPropagation ap = new AffinityPropagation(data).fit();
		assertTrue(ap.getMaxIter() == AffinityPropagation.DEF_MAX_ITER);
		assertTrue(ap.didConverge());
		assertTrue(ap.getConvergenceTolerance() == AffinityPropagation.DEF_TOL);
		
		/*
		 * test double fit!
		 */
		assertTrue(ap.equals(ap.fit()));
		assertFalse(ap.equals(new AffinityPropagation(data)));
		assertTrue(ap.equals(ap));
		assertFalse(ap.equals(new Object()));
		
		AffinityPropagation newer = new AffinityPropagation(data).fit();
		assertTrue(newer.getKey().equals(ap.getKey()) || !ap.equals(newer));
		//assertTrue(new AffinityPropagation(data).equals(new AffinityPropagation(data)));
	}
	
	@Test
	public void testCentroids() {
		AffinityPropagation ap = new AffinityPropagation(data);
		
		/*
		 * Test not fit exception
		 */
		boolean a = false;
		try {
			ap.getCentroids();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * fit
		 */
		ap.fit();
		ap.getCentroids(); // should pass
	}
}
