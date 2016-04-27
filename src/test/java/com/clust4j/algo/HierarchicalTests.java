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
import java.util.ArrayList;
import java.util.Arrays;

import static com.clust4j.TestSuite.getRandom;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.utils.SimpleHeap;
import com.clust4j.algo.HierarchicalAgglomerativeParameters;
import com.clust4j.algo.HierarchicalAgglomerative.EfficientDistanceMatrix;
import com.clust4j.algo.HierarchicalAgglomerative.Linkage;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.kernel.Kernel;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;

public class HierarchicalTests implements ClusterTest, ClassifierTest, BaseModelTest {
	final Array2DRowRealMatrix data_ = TestSuite.IRIS_DATASET.getData();
	private static Array2DRowRealMatrix matrix = getRandom(250, 10);
	static final MatrixFormatter formatter = new MatrixFormatter();
	
	@Test
	public void testCondensedIdx() {
		assertTrue(EfficientDistanceMatrix.getIndexFromFlattenedVec(10, 3, 4) == 24);
	}
	
	@Test
	public void testCut() {
		final double[][] children = new double[][] {
			new double[]{0,2},
			new double[]{1,3}
		};
		
		final int n_clusters = 2, n_leaves = 3;
		int[] l = HierarchicalAgglomerative.hcCut(n_clusters, children, n_leaves);
		assertTrue(l[0]==0 && l[1]==1 && l[2]==0);
	}
	
	@Test
	public void testRandom() {
		HierarchicalAgglomerative hac = 
			new HierarchicalAgglomerative(matrix,
				new HierarchicalAgglomerativeParameters().setVerbose(false));
		hac.fit();
	}

	@Test
	public void testMore() {
		final double[][] data = new double[][] {
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		int[] labels;
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		for(Linkage linkage: HierarchicalAgglomerative.Linkage.values()) {
			HierarchicalAgglomerative hac = 
				new HierarchicalAgglomerative(mat,
					new HierarchicalAgglomerativeParameters()
							.setLinkage(linkage)
							.setVerbose(false)).fit();
			
			labels = hac.getLabels();
			assertTrue(labels[0] == labels[2]);
		}
	}
	
	@Test
	public void testKernel() {
		final double[][] data = new double[][] {
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		int[] labels;
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		for(Linkage linkage: HierarchicalAgglomerative.Linkage.values()) {
			HierarchicalAgglomerative hac = 
				new HierarchicalAgglomerative(mat,
					new HierarchicalAgglomerativeParameters()
							.setLinkage(linkage)
							.setMetric(new GaussianKernel())
							.setVerbose(false)).fit();
			
			labels = hac.getLabels();
			assertTrue(labels[0] == labels[2]);
		}
	}
	
	@Test
	public void testTreeLinkage() {
		final double[][] data = new double[][] {
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		HierarchicalAgglomerative hac = 
			new HierarchicalAgglomerative(mat,
				new HierarchicalAgglomerativeParameters()
						.setLinkage(Linkage.AVERAGE)
						.setVerbose(false));
		
		//final double[][] dist_mat = com.clust4j.utils.ClustUtils.distanceUpperTriangMatrix(mat, hac.getSeparabilityMetric());
		//System.out.println(formatter.format(new Array2DRowRealMatrix(dist_mat,false)));
		
		HierarchicalAgglomerative.AverageLinkageTree a = hac.new AverageLinkageTree();
		final double[][] Z = a.linkage();
		assertTrue(Z[0][0] == 0 && Z[0][1] == 2 && Z[1][0] == 1 && Z[1][1]==3);
	}
	
	@Test
	public void loadTest() {
		Array2DRowRealMatrix mat = getRandom(250, 10); // need to reduce size for travis CI
		new HierarchicalAgglomerative(mat,
			new HierarchicalAgglomerativeParameters()
					.setLinkage(Linkage.AVERAGE)
					.setVerbose(true)).fit().getLabels();
	}
	
	@Test
	public void loadTestKernel() {
		Array2DRowRealMatrix mat = getRandom(250, 10); // need to reduce size for travis CI
		new HierarchicalAgglomerative(mat,
			new HierarchicalAgglomerativeParameters()
					.setLinkage(Linkage.AVERAGE)
					.setMetric(new GaussianKernel())
					.setVerbose(true)).fit().getLabels();
	}

	@Test
	@Override
	public void testScoring() {
		new HierarchicalAgglomerative(data_).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testDefConst() {
		new HierarchicalAgglomerative(data_);
	}

	@Test
	@Override
	public void testArgConst() {
		// NA
		return;
	}

	@Test
	@Override
	public void testPlannerConst() {
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters());
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters(Linkage.AVERAGE));
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters(Linkage.COMPLETE));
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters(Linkage.WARD));
	}

	@Test
	@Override
	public void testFit() {
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters()).fit();
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters(Linkage.AVERAGE)).fit();
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters(Linkage.COMPLETE)).fit();
		new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters(Linkage.WARD)).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new HierarchicalAgglomerativeParameters().fitNewModel(data_);
		new HierarchicalAgglomerativeParameters(Linkage.AVERAGE).fitNewModel(data_);
		new HierarchicalAgglomerativeParameters(Linkage.COMPLETE).fitNewModel(data_);
		new HierarchicalAgglomerativeParameters(Linkage.WARD).fitNewModel(data_);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit1() {
		new HierarchicalAgglomerative(data_).getLabels();
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		HierarchicalAgglomerative agglom = 
			new HierarchicalAgglomerative(matrix, 
				new HierarchicalAgglomerativeParameters()
					.setVerbose(true)).fit();
		
		final int[] l = agglom.getLabels();
		agglom.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		HierarchicalAgglomerative agglom2 = (HierarchicalAgglomerative)HierarchicalAgglomerative
			.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		
		// test re-fit:
		agglom2 = agglom2.fit();
		
		assertTrue(VecUtils.equalsExactly(l, agglom2.getLabels()));
		assertTrue(agglom2.equals(agglom));
		Files.delete(TestSuite.path);
	}
	
	@Test(expected=IllegalStateException.class)
	public void testHeapUtils() {
		final ArrayList<Integer> a= new ArrayList<>();
		final SimpleHeap<Integer> b = new SimpleHeap<>(a);
		
		assertNotNull(b); // Just to make sure we get here...
		b.popInPlace(); // thrown here
	}
	
	@Test
	public void testHeapifier() {
		// Test heapify initial
		final SimpleHeap<Integer> x = new SimpleHeap<>(new ArrayList<>(Arrays.asList(new Integer[]{19, 56, 1, 52, 7, 2, 23})));
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{1, 7, 2, 52, 56, 19, 23}))));
		
		// Test push pop
		Integer i = x.pushPop(2);
		assertTrue(i.equals(1));
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{2, 7, 2, 52, 56, 19, 23}))));
		
		// Test pop
		i = x.pop();
		assertTrue(i.equals(2));
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{2, 7, 19, 52, 56, 23}))));
		
		// Test push
		x.push(9);
		assertTrue(x.equals(new ArrayList<Integer>(Arrays.asList(new Integer[]{2, 7, 9, 52, 56, 23, 19}))));
		
		while(!x.isEmpty())
			x.popInPlace();
		
		assertTrue(x.size() == 0);
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
		
		int[] labels = new HierarchicalAgglomerative(X, new HierarchicalAgglomerativeParameters(Linkage.AVERAGE).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
		
		labels = new HierarchicalAgglomerative(X, new HierarchicalAgglomerativeParameters(Linkage.COMPLETE).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
		
		labels = new HierarchicalAgglomerative(X, new HierarchicalAgglomerativeParameters(Linkage.WARD).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, 0).all());
	}
	
	@Test
	public void testValidMetrics() {
		HierarchicalAgglomerative model;
		Linkage link;
		
		// small dataset for haversine
		Array2DRowRealMatrix small = TestSuite.IRIS_SMALL.getData();
		
		/*
		 * First try Complete and Average -- should allow anything...
		 */
		for(Linkage l: new Linkage[]{Linkage.COMPLETE, Linkage.AVERAGE}) {
			link = l;
			for(Distance d: Distance.values()) {
				model = new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(d)).fit();
				assertTrue(model.dist_metric.equals(d)); // assert didn't change...
			}
			
			// minkowski
			DistanceMetric d = new MinkowskiDistance(1.5);
			model = new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(d)).fit();
			assertTrue(model.dist_metric.equals(d)); // assert didn't change...
			
			// haversine
			d = Distance.HAVERSINE.MI;
			model = new HierarchicalAgglomerative(small, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(d)).fit();
			assertTrue(model.dist_metric.equals(d)); // assert didn't change...
			
			// similarity?
			for(Kernel k: KernelTestCases.all_kernels) {
				model = new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(k)).fit();
			}
		}
		
		
		/*
		 * Ward. Should only allow Euclidean distance
		 */
		link = Linkage.WARD;
		for(Distance d: Distance.values()) {
			model = new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(d)).fit();
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN)); // assert didn't change...
		}
		
		// minkowski
		DistanceMetric d = new MinkowskiDistance(1.5);
		model = new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN)); // assert didn't change...
		
		// haversine
		d = Distance.HAVERSINE.MI;
		model = new HierarchicalAgglomerative(small, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(d)).fit();
		assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN)); // assert didn't change...
		
		// similarity?
		for(Kernel k: KernelTestCases.all_kernels) {
			model = new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(k)).fit();
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN)); // assert didn't change...
		}
		
		model = new HierarchicalAgglomerative(data_, new HierarchicalAgglomerativeParameters().setLinkage(link).setMetric(Distance.HAVERSINE.KM));
		assertTrue(model.getLinkage().equals(link));
		assertTrue(model.getSeparabilityMetric().equals(Distance.EUCLIDEAN));
	}
	
	@Test
	public void testBadEfficientDistMatTest() {
		boolean a = false;
		try {
			HierarchicalAgglomerative.EfficientDistanceMatrix.getIndexFromFlattenedVec(0, 0, 0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testPredict() {
		HierarchicalAgglomerative h = new HierarchicalAgglomerativeParameters(3).fitNewModel(data_);
		
		/*
		 * Test on actual rows
		 */
		int[] predicted = h.predict(new Array2DRowRealMatrix(new double[][]{
			h.data.getRow(0),
			h.data.getRow(148),
			h.data.getRow(149)
		}, false));
		
		assertTrue(VecUtils.equalsExactly(predicted, new int[]{0,2,1}));
		
		
		/*
		 * Test on random rows...
		 */
		predicted = h.predict(new Array2DRowRealMatrix(new double[][]{
			new double[]{150,150,150,150}
		}, false));
		
		assertTrue(VecUtils.equalsExactly(predicted, new int[]{2}));
		
		
		/*
		 * Test on k = 1
		 */
		h = new HierarchicalAgglomerativeParameters(1).fitNewModel(data_);
		predicted = h.predict(new Array2DRowRealMatrix(new double[][]{
			h.data.getRow(0),
			h.data.getRow(148),
			h.data.getRow(149),
			new double[]{150,150,150,150}
		}, false));
		assertTrue(VecUtils.equalsExactly(predicted, new int[]{0,0,0,0}));
		
		/*
		 * Test for dim mismatch
		 */
		Array2DRowRealMatrix newData = new Array2DRowRealMatrix(new double[][]{
			new double[]{150,150,150,150,150}
		}, false);
		boolean a = false;
		try {
			h.predict(newData);
		} catch(DimensionMismatchException dim) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
}
