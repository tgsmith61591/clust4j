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
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.HDBSCAN.HDBSCAN_Algorithm;
import com.clust4j.algo.HDBSCAN.CompQuadTup;
import com.clust4j.algo.HDBSCANParameters;
import com.clust4j.algo.HDBSCAN.LinkageTreeUtils;
import com.clust4j.algo.HDBSCAN.TreeUnionFind;
import com.clust4j.algo.HDBSCAN.UnionFind;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.algo.Neighborhood;
import com.clust4j.data.DataSet;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.kernel.Kernel;
import com.clust4j.kernel.KernelTestCases;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.MinkowskiDistance;
import com.clust4j.metrics.pairwise.Pairwise;
import com.clust4j.metrics.pairwise.Similarity;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.Series.Inequality;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.MatUtils.MatSeries;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.QuadTup;

public class HDBSCANTests implements ClusterTest, ClassifierTest, BaseModelTest {
	final Array2DRowRealMatrix DATA = TestSuite.IRIS_DATASET.getData();
	final Array2DRowRealMatrix iris = DATA;
	final static MatrixFormatter formatter = TestSuite.formatter;
	final static double[][] dist_mat = new double[][]{
		new double[]{1,2,3},
		new double[]{4,5,6},
		new double[]{7,8,9}
	};
	
	final int[] expected_iris_labs  = new LabelEncoder(new int[]{
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	}).fit().getEncodedLabels();

	
	@Test
	public void testHDBSCANGenericMutualReachability() {
		final int m = dist_mat.length, minPts = 3;
		
		final int min_points = FastMath.min(m - 1, minPts);
		final double[] core_distances = MatUtils
			.sortColsAsc(dist_mat)[min_points];
		
		final MatSeries ser1 = new MatSeries(core_distances, Inequality.GREATER_THAN, dist_mat);
		double[][] stage1 = MatUtils.where(ser1, core_distances, dist_mat);
		
		stage1 = MatUtils.transpose(stage1);
		final MatSeries ser2 = new MatSeries(core_distances, Inequality.GREATER_THAN, stage1);
		final double[][] result = MatUtils.where(ser2, core_distances, stage1);
		
		final double[][] res = MatUtils.transpose(result);
		assertTrue(MatUtils.equalsExactly(res, new double[][]{
			new double[]{7,8,9},
			new double[]{8,8,9},
			new double[]{9,9,9}
		}));
	}
	
	@Test
	public void testHDBSCANGenericMstLinkageCore() {
		double[][] X = new double[][]{
			new double[]{ 0.1,   0.6,   0.3},
			new double[]{ 0.6,   0.6,   0.6},
			new double[]{12.1,  13.1,  11.8}
		};
		
		final int m = X.length;
		
		double[][] result = HDBSCAN.LinkageTreeUtils.minSpanTreeLinkageCore(X, m);
		
		final double[][] expected = new double[][]{
			new double[]{0.0, 2.0, 0.3},
			new double[]{2.0, 1.0, 0.6}
		};
		
		assertTrue(MatUtils.equalsExactly(result, expected));
	}
	
	@Test
	public void testArgSorting() {
		double[][] m = new double[][]{
			new double[]{0.0 ,  2.0 ,  0.3},
			new double[]{2.0 ,  1.0 ,  0.6}
		};
		
		double[][] exp = MatUtils.copy(m);
		int[] sortedArgs = VecUtils.argSort(MatUtils.transpose(m)[2]);
		m = MatUtils.reorder(m, sortedArgs);
		
		assertTrue(MatUtils.equalsExactly(exp, m));
	}
	
	@Test
	public void testLabeling() {
		double[][] x = new double[][]{
			new double[]{0.0, 2.0, 0.3},
			new double[]{2.0, 1.0, 0.6},
		};
		
		double[][] y = new double[][]{
			new double[]{0.0, 2.0, 0.3, 2.0},
			new double[]{3.0, 1.0, 0.6, 3.0}
		};
		
		assertTrue(MatUtils.equalsExactly(HDBSCAN.label(x), y));
		
		// More complex
		assertTrue(MatUtils.equalsExactly(HDBSCAN.label(
				new double[][]{
					new double[]{0.0,2.0,0.3,1.9}, 
					new double[]{2.0,1.0,0.6,6.7},
					new double[]{1.0,4.3,0.9,0.1}
				}), 
				new double[][]{
			new double[]{0.0,2.0,0.3,2.0,0.0},
		    new double[]{4.0,1.0,0.6,3.0,0.0},
		    new double[]{5.0,5.0,0.9,6.0,0.}
		}));
	}
	
	@Test
	public void testCondenseAndComputeStability() {
		double[][] slt = new double[][]{
			new double[]{0.0, 2.0, 0.3, 2.0},
			new double[]{3.0, 1.0, 0.6, 3.0}
		};
		
		ArrayList<CompQuadTup<Integer, Integer, Double, Integer>> h = HDBSCAN.LinkageTreeUtils.condenseTree(slt, 5);
		QuadTup<Integer, Integer, Double, Integer> q = h.get(0);
		assertTrue(q.getFirst() == 3);
		assertTrue(q.getSecond() == 0);
		// Three is a repeating decimal...
		assertTrue(q.getFourth() == 1);
		
		TreeMap<Integer, Double> computedStability = HDBSCAN.LinkageTreeUtils.computeStability(h);
		assertTrue(computedStability.size() == 1);
		assertTrue(computedStability.get(3) == 5);
		
		int[] labels = HDBSCAN.getLabels(h, computedStability);
		assertTrue(labels.length == 3);
		assertTrue(labels[0] == -1);
		assertTrue(labels[1] == -1);
		assertTrue(labels[2] == -1);
	}
	
	@Test
	public void testUnionFindClass() {
		UnionFind u = new UnionFind(5);
		int val = 2;
		
		// Test finds
		assertTrue(VecUtils.equalsExactly(new int[]{1,1,1,1,1,0,0,0,0}, u.size));
		assertTrue(u.fastFind(val) == val);
		assertTrue(u.parent[u.parent.length - 1] == val);
		assertFalse(u.parent[val] == val); // Should look like [-1, -1, -1, -1, -1, -1, -1, -1,  2]
		assertTrue(u.fastFind(-1) == val);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1, -1, -1, -1, -1, -1, 2}));
		assertTrue(u.fastFind(3) == 3);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1, -1, -1, -1, -1, -1, 3}));
		
		// Test unions
		u.union(3, 4);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1,  5,  5, -1, -1, -1, 3}));
		assertTrue(VecUtils.equalsExactly(u.size, new int[]{1, 1, 1, 1, 1, 2, 0, 0, 0}));
		
		u.union(-1, -2);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1,  5,  5, -1, -1,  6,  6}));
		
		assertTrue(u.find(6) == 6);
		assertTrue(VecUtils.equalsExactly(u.parent, new int[]{-1, -1, -1,  5,  5, -1, -1,  6,  6}));
	}
	
	@Test
	public void testTreeUnionFindClass() {
		TreeUnionFind t = new TreeUnionFind(5);
		int[][] parent = new int[][]{
				new int[]{0,0},
				new int[]{1,0},
				new int[]{2,0},
				new int[]{3,0},
				new int[]{4,0}
			};
		
		assertTrue(MatUtils.equalsExactly(t.dataArr, parent));
		assertTrue(t.find(3) == 3);
		assertTrue(MatUtils.equalsExactly(t.dataArr, parent));
		
		assertTrue(t.find(-1) == 4);
		assertTrue(MatUtils.equalsExactly(t.dataArr, parent));
		
		// Test union
		t.union(3, -1);
		assertTrue(MatUtils.equalsExactly(t.dataArr, new int[][]{
			new int[]{0,0},
			new int[]{1,0},
			new int[]{2,0},
			new int[]{3,1},
			new int[]{3,0}
		}));
		
		t.union(-3, 4);
		assertTrue(MatUtils.equalsExactly(t.dataArr, new int[][]{
			new int[]{0,0},
			new int[]{1,0},
			new int[]{3,0},
			new int[]{3,1},
			new int[]{3,0}
		}));
		
		assertTrue(VecUtils.equalsExactly(t.components(), new int[]{0,1,3}));
	}

	@Test
	public void testGenericRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANParameters(1)
				.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testFindNodeSplitDim() {
		final double[][] a = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		assertTrue(NearestNeighborHeapSearch.findNodeSplitDim(a, new int[]{0,1,2}) == 2);
	}
	
	
	@Test
	public void testPrimKD() {
		double m = dist_mat.length;
		int min_points = (int)FastMath.min(m - 1, 5);
		Array2DRowRealMatrix X = new Array2DRowRealMatrix(dist_mat);
		
		KDTree tree = new KDTree(X, HDBSCAN.DEF_LEAF_SIZE, Distance.EUCLIDEAN);
		Neighborhood query = tree.query(dist_mat, min_points, true, true);
		double[][] dists = query.getDistances();
		double[] coreDistances = MatUtils.getColumn(dists, dists[0].length - 1);
		
		// Needs to equal this:
		// [ 5.19615242,  5.19615242,  5.19615242]
		assertTrue(VecUtils.equalsExactly(coreDistances, new double[]{
			5.196152422706632, 5.196152422706632, 5.196152422706632
		}));
		
		assertTrue(MatUtils.equalsExactly(query.getValue(), new int[][]{
			new int[]{0,1},
			new int[]{1,0},
			new int[]{2,1}
		}));
		
		
		double[][] minSpanningTree = LinkageTreeUtils
				.minSpanTreeLinkageCore_cdist(dist_mat, coreDistances, 
						Distance.EUCLIDEAN, HDBSCAN.DEF_ALPHA);
		
		double[][] expected = new double[][]{
			new double[]{0.0, 1.0, 5.196152422706632},
			new double[]{1.0, 2.0, 5.196152422706632}
		};
		
		assertTrue(MatUtils.equalsExactly(minSpanningTree, expected));
	}
	
	@Test
	public void testPrimKDRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANParameters(1)
					.setAlgo(HDBSCAN_Algorithm.PRIMS_KDTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testPrimBall() {
		double m = dist_mat.length;
		int min_points = (int)FastMath.min(m - 1, 5);
		Array2DRowRealMatrix X = new Array2DRowRealMatrix(dist_mat);
		
		BallTree tree = new BallTree(X, HDBSCAN.DEF_LEAF_SIZE, Distance.EUCLIDEAN);
		Neighborhood query = tree.query(dist_mat, min_points, true, true);
		double[][] dists = query.getDistances();
		double[] coreDistances = MatUtils.getColumn(dists, dists[0].length - 1);
		
		// Needs to equal this:
		// [ 5.19615242,  5.19615242,  5.19615242]
		assertTrue(VecUtils.equalsExactly(coreDistances, new double[]{
			5.196152422706632, 5.196152422706632, 5.196152422706632
		}));
		
		assertTrue(MatUtils.equalsExactly(query.getValue(), new int[][]{
			new int[]{0,1},
			new int[]{1,0},
			new int[]{2,1}
		}));
		
		
		double[][] minSpanningTree = LinkageTreeUtils
				.minSpanTreeLinkageCore_cdist(dist_mat, coreDistances, 
						Distance.EUCLIDEAN, HDBSCAN.DEF_ALPHA);
		
		double[][] expected = new double[][]{
			new double[]{0.0, 1.0, 5.196152422706632},
			new double[]{1.0, 2.0, 5.196152422706632}
		};
		
		assertTrue(MatUtils.equalsExactly(minSpanningTree, expected));
	}
	
	@Test
	public void testMoreBFS() {
		double[][] x = new double[][]{
			new double[]{0,1,1.414,2},
			new double[]{3,2,10.05,3}
		};
		
		ArrayList<Integer> result;
		int root;
		
		// Test with root == 0
		root = 0;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
		
		// Test with root == 1
		root = 1;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);

		// Test with root == 2
		root = 2;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
		
		// Test with root == -1
		root = -1;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
		
		// Test with root == -2
		root = -2;
		result = HDBSCAN.LinkageTreeUtils.breadthFirstSearch(x, root);
		assertTrue(result.size() == 1);
		assertTrue(result.get(0) == root);
	}
	
	@Test
	public void testPrimBallRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANParameters(1)
					.setAlgo(HDBSCAN_Algorithm.PRIMS_BALLTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testDescKeySet() {
		TreeMap<Integer, Double> stability = new TreeMap<>();
		stability.put(1, 456.0);
		stability.put(9, 23.0);
		stability.put(-5, 89.0);
			
		ArrayList<Integer> nodes = HDBSCAN.GetLabelUtils.descSortedKeySet(stability);
		assertTrue(nodes.size() == 2);
		assertTrue(nodes.get(0) == 9);
		assertTrue(nodes.get(1) == 1);
		// It should trim the last one
	}
	
	@Test
	public void testSizeOverOne() {
		ArrayList<CompQuadTup<Integer, Integer, Double, Integer>> tup = new ArrayList<>();
		tup.add(new CompQuadTup<Integer, Integer, Double, Integer>(1,2,1.0,1));
		tup.add(new CompQuadTup<Integer, Integer, Double, Integer>(1,1,1.0,2));
		tup.add(new CompQuadTup<Integer, Integer, Double, Integer>(1,1,1.0,2));
		tup.add(new CompQuadTup<Integer, Integer, Double, Integer>(1,1,1.0,2));
		
		EntryPair<ArrayList<double[]>, Integer> entry = 
			HDBSCAN.GetLabelUtils.childSizeGtOneAndMaxChild(tup);
		
		assertTrue(entry.getKey().size() == 3);
		assertTrue(entry.getValue() == 3);
	}
	
	@Test
	public void testDataSet() { // See if the iris dataset works...
		new HDBSCAN(iris, 
				new HDBSCANParameters(1)
					.setVerbose(true)).fit();
		
	}

	@Test
	@Override
	public void testScoring() {
		new HDBSCAN(DATA).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testDefConst() {
		new HDBSCAN(DATA);
	}

	@Test
	@Override
	public void testArgConst() {
		new HDBSCAN(DATA, 3);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new HDBSCAN(DATA, new HDBSCANParameters());
	}

	@Test
	@Override
	public void testFit() {
		new HDBSCAN(DATA, 1).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new HDBSCANParameters().fitNewModel(DATA);
		new HDBSCANParameters(3).fitNewModel(DATA);
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		HDBSCAN hd = new HDBSCAN(DATA, 
			new HDBSCANParameters(1)
				.setVerbose(true)).fit();
		System.out.println();

		final int[] labels = hd.getLabels();
		hd.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		HDBSCAN hd2 = (HDBSCAN)HDBSCAN.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(VecUtils.equalsExactly(hd2.getLabels(), labels));
		assertTrue(hd.equals(hd2));
		Files.delete(TestSuite.path);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testAlphaIAE() {
		new HDBSCAN(TestSuite.getRandom(5, 5), new HDBSCANParameters().setAlpha(0.0));
	}
	
	@Test
	public void testSepWarn() {
		HDBSCAN h = new HDBSCAN(TestSuite.getRandom(5, 5), 
			new HDBSCANParameters()
				.setAlgo(HDBSCAN_Algorithm.PRIMS_KDTREE)
				.setMetric(new GaussianKernel()));
		assertTrue(h.hasWarnings());
	}
	
	@Test
	public void compareToPckg() {
		Array2DRowRealMatrix X= new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				0.58459246,  0.2411591 ,  0.54266953,  0.80298748,  0.7108317 ,
		        0.77419375,  0.19460038,  0.51769224,  0.80581355,  0.06109043,
		        0.57755264,  0.48690635,  0.4698578 ,  0.68256655,  0.35583625,
		        0.33956817,  0.46084149,  0.2266772 ,  0.78013553,  0.84169725,
		        0.45929076,  0.5763663 ,  0.85034392,  0.42344478,  0.08823549
			}, 5, 5), false);
		
		HDBSCAN h = new HDBSCAN(X).fit();
		assertTrue(VecUtils.equalsExactly(h.getLabels(), VecUtils.repInt(-1, 5)));
		
		h = new HDBSCAN(X, new HDBSCANParameters().setAlgo(HDBSCAN_Algorithm.PRIMS_KDTREE)).fit();
		assertTrue(VecUtils.equalsExactly(h.getLabels(), VecUtils.repInt(-1, 5)));
		
		h = new HDBSCAN(X, new HDBSCANParameters().setAlgo(HDBSCAN_Algorithm.PRIMS_BALLTREE)).fit();
		assertTrue(VecUtils.equalsExactly(h.getLabels(), VecUtils.repInt(-1, 5)));
		
		// Test on IRIS
		h = new HDBSCAN(iris, new HDBSCANParameters().setAlgo(HDBSCAN_Algorithm.GENERIC)).fit();
		
		int[] expectedLabels = new NoiseyLabelEncoder(expected_iris_labs)
			.fit().getEncodedLabels();
		
		assertTrue(VecUtils.equalsExactly(expectedLabels, h.getLabels()));
		
		h = new HDBSCAN(X, new HDBSCANParameters().setAlgo(HDBSCAN_Algorithm.PRIMS_KDTREE)).fit();
		System.out.println(Arrays.toString(h.getLabels()));
	}
	
	@Test
	public void testMutualReachability() {
		Array2DRowRealMatrix X= new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3), false);
		
		final double[][] dist = Pairwise.getDistance(X, Distance.EUCLIDEAN, false, false);
		
		
		// first test the partition by row
		final int minPts = FastMath.min(X.getRowDimension() - 1, 5);
		final double[] core_distances = MatUtils
				.sortColsAsc(dist)[minPts];
		assertTrue(VecUtils.equalsExactly(core_distances, new double[]{
			10.392304845413264,  5.196152422706632, 10.392304845413264
		}));
		
		final double[][] expected = new double[][]{
			new double[]{10.392304845413264, 10.392304845413264, 10.392304845413264},
			new double[]{10.392304845413264,  5.196152422706632, 10.392304845413264},
			new double[]{10.392304845413264, 10.392304845413264, 10.392304845413264},
		};
		
		double[][] mr = HDBSCAN.LinkageTreeUtils.mutualReachability(dist, minPts, 1.0);
		assertTrue(MatUtils.equalsExactly(mr, expected));
	}
	
	@Test
	public void testGenericAlgo() {
		HDBSCAN h = new HDBSCAN(iris,
			new HDBSCANParameters()
				.setAlgo(HDBSCAN_Algorithm.GENERIC)).fit();
		
		assertTrue(Precision.equals(h.indexAffinityScore(expected_iris_labs), 1.0, 0.05));
		assertTrue(h.getNumberOfIdentifiedClusters() == 2);
	}
	
	@Test
	public void testPrimsKD() {
		HDBSCAN h = new HDBSCAN(iris,
			new HDBSCANParameters()
				.setAlgo(HDBSCAN_Algorithm.PRIMS_KDTREE)).fit();

		assertTrue(Precision.equals(h.indexAffinityScore(expected_iris_labs), 1.0, 0.05));
	}
	
	@Test
	public void testPrimsBall() {
		HDBSCAN h = new HDBSCAN(iris,
			new HDBSCANParameters()
				.setAlgo(HDBSCAN_Algorithm.PRIMS_BALLTREE)).fit();

		assertTrue(Precision.equals(h.indexAffinityScore(expected_iris_labs), 1.0, 0.05));
	}
	
	@Test
	public void testBoruvkaKDTree() {
		HDBSCAN h = new HDBSCAN(iris,
			new HDBSCANParameters()
				.setAlgo(HDBSCAN_Algorithm.BORUVKA_KDTREE)).fit();
		
		assertTrue(Precision.equals(h.indexAffinityScore(expected_iris_labs), 1.0, 0.05));
	}
	
	@Test
	public void testBoruvkaBallTree() {
		HDBSCAN h = new HDBSCAN(iris,
			new HDBSCANParameters()
				.setAlgo(HDBSCAN_Algorithm.BORUVKA_BALLTREE)).fit();
		
		assertTrue(Precision.equals(h.indexAffinityScore(expected_iris_labs), 1.0, 0.05));
	}
	
	@Test
	public void testPrimLinkage() {
		KDTree k = new KDTree(iris);
		double[] core_dists = MatUtils.getColumn(
			k.query(iris.getDataRef(), 5, true, true).getDistances(),
			4
		);
		
		double[][] X = iris.getData();
		DistanceMetric metric = Distance.EUCLIDEAN;
		double alpha = 1.0;
		
		double[][] cdist = HDBSCAN.LinkageTreeUtils
			.minSpanTreeLinkageCore_cdist(X, 
				core_dists, metric, alpha);
		
		double[][] expected_cdists = new double[][]{
			new double[]{  0.00000000e+00,   3.90000000e+01,   1.41421356e-01},
			new double[]{  3.90000000e+01,   1.70000000e+01,   1.73205081e-01},
			new double[]{  1.70000000e+01,   2.70000000e+01,   1.73205081e-01},
			new double[]{  2.70000000e+01,   2.80000000e+01,   1.73205081e-01},
			new double[]{  2.80000000e+01,   7.00000000e+00,   2.00000000e-01},
			new double[]{  7.00000000e+00,   4.90000000e+01,   2.23606798e-01},
			new double[]{  4.90000000e+01,   4.00000000e+00,   2.23606798e-01},
			new double[]{  4.00000000e+00,   2.60000000e+01,   2.44948974e-01},
			new double[]{  2.60000000e+01,   4.00000000e+01,   2.44948974e-01},
			new double[]{  4.00000000e+01,   4.80000000e+01,   2.44948974e-01},
			new double[]{  4.80000000e+01,   9.00000000e+00,   2.64575131e-01},
			new double[]{  9.00000000e+00,   1.00000000e+00,   1.73205081e-01},
			new double[]{  1.00000000e+00,   3.00000000e+01,   1.73205081e-01},
			new double[]{  3.00000000e+01,   3.40000000e+01,   1.73205081e-01},
			new double[]{  3.40000000e+01,   3.70000000e+01,   1.73205081e-01},
			new double[]{  3.70000000e+01,   1.20000000e+01,   1.73205081e-01},
			new double[]{  1.20000000e+01,   2.50000000e+01,   2.23606798e-01},
			new double[]{  2.50000000e+01,   2.90000000e+01,   2.23606798e-01},
			new double[]{  2.90000000e+01,   4.70000000e+01,   2.23606798e-01},
			new double[]{  4.70000000e+01,   3.00000000e+00,   2.44948974e-01},
			new double[]{  3.00000000e+00,   1.90000000e+01,   2.64575131e-01},
			new double[]{  1.90000000e+01,   2.00000000e+00,   2.64575131e-01},
			new double[]{  2.00000000e+00,   2.10000000e+01,   2.64575131e-01},
			new double[]{  2.10000000e+01,   4.50000000e+01,   2.64575131e-01},
			new double[]{  4.50000000e+01,   3.80000000e+01,   3.00000000e-01},
			new double[]{  3.80000000e+01,   1.10000000e+01,   3.00000000e-01},
			new double[]{  1.10000000e+01,   4.20000000e+01,   3.00000000e-01},
			new double[]{  4.20000000e+01,   4.60000000e+01,   3.00000000e-01},
			new double[]{  4.60000000e+01,   6.00000000e+00,   3.16227766e-01},
			new double[]{  6.00000000e+00,   3.10000000e+01,   3.16227766e-01},
			new double[]{  3.10000000e+01,   3.50000000e+01,   3.31662479e-01},
			new double[]{  3.50000000e+01,   1.00000000e+01,   3.31662479e-01},
			new double[]{  1.00000000e+01,   8.00000000e+00,   3.46410162e-01},
			new double[]{  8.00000000e+00,   3.60000000e+01,   3.46410162e-01},
			new double[]{  3.60000000e+01,   2.00000000e+01,   3.60555128e-01},
			new double[]{  2.00000000e+01,   4.30000000e+01,   3.74165739e-01},
			new double[]{  4.30000000e+01,   5.00000000e+00,   3.74165739e-01},
			new double[]{  5.00000000e+00,   2.30000000e+01,   3.87298335e-01},
			new double[]{  2.30000000e+01,   1.60000000e+01,   3.87298335e-01},
			new double[]{  1.60000000e+01,   4.40000000e+01,   4.12310563e-01},
			new double[]{  4.40000000e+01,   3.30000000e+01,   4.12310563e-01},
			new double[]{  3.30000000e+01,   3.20000000e+01,   4.24264069e-01},
			new double[]{  3.20000000e+01,   2.40000000e+01,   4.24264069e-01},
			new double[]{  2.40000000e+01,   1.30000000e+01,   4.79583152e-01},
			new double[]{  1.30000000e+01,   1.80000000e+01,   5.09901951e-01},
			new double[]{  1.80000000e+01,   2.20000000e+01,   5.38516481e-01},
			new double[]{  2.20000000e+01,   1.40000000e+01,   5.56776436e-01},
			new double[]{  1.40000000e+01,   1.50000000e+01,   6.16441400e-01},
			new double[]{  1.50000000e+01,   4.10000000e+01,   7.81024968e-01},
			new double[]{  4.10000000e+01,   9.80000000e+01,   1.64012195e+00},
			new double[]{  9.80000000e+01,   5.70000000e+01,   7.93725393e-01},
			new double[]{  5.70000000e+01,   6.00000000e+01,   7.21110255e-01},
			new double[]{  6.00000000e+01,   8.00000000e+01,   7.14142843e-01},
			new double[]{  8.00000000e+01,   6.90000000e+01,   3.00000000e-01},
			new double[]{  6.90000000e+01,   9.20000000e+01,   2.64575131e-01},
			new double[]{  9.20000000e+01,   9.90000000e+01,   2.64575131e-01},
			new double[]{  9.90000000e+01,   9.60000000e+01,   2.44948974e-01},
			new double[]{  9.60000000e+01,   8.20000000e+01,   3.00000000e-01},
			new double[]{  8.20000000e+01,   8.90000000e+01,   3.00000000e-01},
			new double[]{  8.90000000e+01,   9.40000000e+01,   3.00000000e-01},
			new double[]{  9.40000000e+01,   8.80000000e+01,   3.16227766e-01},
			new double[]{  8.80000000e+01,   5.50000000e+01,   3.31662479e-01},
			new double[]{  5.50000000e+01,   9.50000000e+01,   3.31662479e-01},
			new double[]{  9.50000000e+01,   6.70000000e+01,   3.60555128e-01},
			new double[]{  6.70000000e+01,   6.10000000e+01,   3.60555128e-01},
			new double[]{  6.10000000e+01,   7.80000000e+01,   3.60555128e-01},
			new double[]{  7.80000000e+01,   9.10000000e+01,   3.46410162e-01},
			new double[]{  9.10000000e+01,   9.70000000e+01,   3.46410162e-01},
			new double[]{  9.70000000e+01,   7.40000000e+01,   3.87298335e-01},
			new double[]{  7.40000000e+01,   5.40000000e+01,   3.87298335e-01},
			new double[]{  5.40000000e+01,   5.80000000e+01,   3.74165739e-01},
			new double[]{  5.80000000e+01,   7.50000000e+01,   3.16227766e-01},
			new double[]{  7.50000000e+01,   8.60000000e+01,   3.31662479e-01},
			new double[]{  8.60000000e+01,   6.50000000e+01,   3.46410162e-01},
			new double[]{  6.50000000e+01,   5.20000000e+01,   3.46410162e-01},
			new double[]{  5.20000000e+01,   5.10000000e+01,   3.74165739e-01},
			new double[]{  5.10000000e+01,   7.60000000e+01,   3.74165739e-01},
			new double[]{  7.60000000e+01,   7.10000000e+01,   4.00000000e-01},
			new double[]{  7.10000000e+01,   6.60000000e+01,   4.12310563e-01},
			new double[]{  6.60000000e+01,   7.70000000e+01,   4.24264069e-01},
			new double[]{  7.70000000e+01,   1.47000000e+02,   4.24264069e-01},
			new double[]{  1.47000000e+02,   1.45000000e+02,   3.74165739e-01},
			new double[]{  1.45000000e+02,   1.12000000e+02,   3.74165739e-01},
			new double[]{  1.12000000e+02,   1.20000000e+02,   3.74165739e-01},
			new double[]{  1.20000000e+02,   1.40000000e+02,   3.46410162e-01},
			new double[]{  1.40000000e+02,   1.43000000e+02,   3.46410162e-01},
			new double[]{  1.43000000e+02,   1.39000000e+02,   3.74165739e-01},
			new double[]{  1.39000000e+02,   1.24000000e+02,   3.74165739e-01},
			new double[]{  1.24000000e+02,   1.15000000e+02,   3.87298335e-01},
			new double[]{  1.15000000e+02,   1.11000000e+02,   3.87298335e-01},
			new double[]{  1.11000000e+02,   1.03000000e+02,   3.87298335e-01},
			new double[]{  1.03000000e+02,   1.16000000e+02,   3.87298335e-01},
			new double[]{  1.16000000e+02,   1.28000000e+02,   3.87298335e-01},
			new double[]{  1.28000000e+02,   1.04000000e+02,   3.87298335e-01},
			new double[]{  1.04000000e+02,   1.44000000e+02,   4.00000000e-01},
			new double[]{  1.44000000e+02,   1.46000000e+02,   4.12310563e-01},
			new double[]{  1.46000000e+02,   1.23000000e+02,   4.12310563e-01},
			new double[]{  1.23000000e+02,   1.27000000e+02,   3.60555128e-01},
			new double[]{  1.27000000e+02,   1.38000000e+02,   3.16227766e-01},
			new double[]{  1.38000000e+02,   1.49000000e+02,   3.31662479e-01},
			new double[]{  1.49000000e+02,   1.01000000e+02,   3.31662479e-01},
			new double[]{  1.01000000e+02,   1.42000000e+02,   3.31662479e-01},
			new double[]{  1.42000000e+02,   8.30000000e+01,   3.74165739e-01},
			new double[]{  8.30000000e+01,   1.26000000e+02,   3.87298335e-01},
			new double[]{  1.26000000e+02,   7.00000000e+01,   4.24264069e-01},
			new double[]{  7.00000000e+01,   7.20000000e+01,   4.24264069e-01},
			new double[]{  7.20000000e+01,   1.10000000e+02,   4.24264069e-01},
			new double[]{  1.10000000e+02,   9.00000000e+01,   4.24264069e-01},
			new double[]{  9.00000000e+01,   6.30000000e+01,   4.24264069e-01},
			new double[]{  6.30000000e+01,   1.33000000e+02,   4.35889894e-01},
			new double[]{  1.33000000e+02,   5.30000000e+01,   4.35889894e-01},
			new double[]{  5.30000000e+01,   8.10000000e+01,   4.35889894e-01},
			new double[]{  8.10000000e+01,   1.32000000e+02,   4.35889894e-01},
			new double[]{  1.32000000e+02,   1.37000000e+02,   4.35889894e-01},
			new double[]{  1.37000000e+02,   7.30000000e+01,   4.35889894e-01},
			new double[]{  7.30000000e+01,   1.36000000e+02,   4.35889894e-01},
			new double[]{  1.36000000e+02,   7.90000000e+01,   4.47213595e-01},
			new double[]{  7.90000000e+01,   5.60000000e+01,   4.58257569e-01},
			new double[]{  5.60000000e+01,   1.21000000e+02,   4.58257569e-01},
			new double[]{  1.21000000e+02,   5.00000000e+01,   4.58257569e-01},
			new double[]{  5.00000000e+01,   1.02000000e+02,   4.58257569e-01},
			new double[]{  1.02000000e+02,   8.50000000e+01,   4.69041576e-01},
			new double[]{  8.50000000e+01,   1.25000000e+02,   4.69041576e-01},
			new double[]{  1.25000000e+02,   8.40000000e+01,   4.89897949e-01},
			new double[]{  8.40000000e+01,   1.30000000e+02,   5.09901951e-01},
			new double[]{  1.30000000e+02,   1.41000000e+02,   5.09901951e-01},
			new double[]{  1.41000000e+02,   6.40000000e+01,   5.19615242e-01},
			new double[]{  6.40000000e+01,   1.13000000e+02,   5.19615242e-01},
			new double[]{  1.13000000e+02,   1.14000000e+02,   5.19615242e-01},
			new double[]{  1.14000000e+02,   5.90000000e+01,   5.29150262e-01},
			new double[]{  5.90000000e+01,   1.07000000e+02,   5.47722558e-01},
			new double[]{  1.07000000e+02,   1.05000000e+02,   5.47722558e-01},
			new double[]{  1.05000000e+02,   1.29000000e+02,   5.56776436e-01},
			new double[]{  1.29000000e+02,   1.00000000e+02,   5.56776436e-01},
			new double[]{  1.00000000e+02,   1.19000000e+02,   5.83095189e-01},
			new double[]{  1.19000000e+02,   6.20000000e+01,   5.83095189e-01},
			new double[]{  6.20000000e+01,   8.70000000e+01,   6.08276253e-01},
			new double[]{  8.70000000e+01,   1.48000000e+02,   6.16441400e-01},
			new double[]{  1.48000000e+02,   1.08000000e+02,   6.16441400e-01},
			new double[]{  1.08000000e+02,   9.30000000e+01,   6.48074070e-01},
			new double[]{  9.30000000e+01,   1.34000000e+02,   6.63324958e-01},
			new double[]{  1.34000000e+02,   6.80000000e+01,   6.78232998e-01},
			new double[]{  6.80000000e+01,   1.35000000e+02,   6.78232998e-01},
			new double[]{  1.35000000e+02,   1.22000000e+02,   6.78232998e-01},
			new double[]{  1.22000000e+02,   1.09000000e+02,   7.54983444e-01},
			new double[]{  1.09000000e+02,   1.06000000e+02,   8.77496439e-01},
			new double[]{  1.06000000e+02,   1.18000000e+02,   9.27361850e-01},
			new double[]{  1.18000000e+02,   1.31000000e+02,   9.32737905e-01},
			new double[]{  1.31000000e+02,   1.17000000e+02,   1.00498756e+00}
		};
		
		assertTrue(MatUtils.equalsWithTolerance(cdist, expected_cdists, 1e-8));
		
		double[][] srtd_cdists = MatUtils.sortAscByCol(cdist, 2);
		double[][] expected_srted = new double[][]{
			new double[]{  0.00000000e+00,   3.90000000e+01,   1.41421356e-01},
			new double[]{  3.90000000e+01,   1.70000000e+01,   1.73205081e-01},
			new double[]{  1.70000000e+01,   2.70000000e+01,   1.73205081e-01},
			new double[]{  2.70000000e+01,   2.80000000e+01,   1.73205081e-01},
			new double[]{  3.40000000e+01,   3.70000000e+01,   1.73205081e-01},
			new double[]{  3.00000000e+01,   3.40000000e+01,   1.73205081e-01},
			new double[]{  9.00000000e+00,   1.00000000e+00,   1.73205081e-01},
			new double[]{  1.00000000e+00,   3.00000000e+01,   1.73205081e-01},
			new double[]{  3.70000000e+01,   1.20000000e+01,   1.73205081e-01},
			new double[]{  2.80000000e+01,   7.00000000e+00,   2.00000000e-01},
			new double[]{  1.20000000e+01,   2.50000000e+01,   2.23606798e-01},
			new double[]{  7.00000000e+00,   4.90000000e+01,   2.23606798e-01},
			new double[]{  4.90000000e+01,   4.00000000e+00,   2.23606798e-01},
			new double[]{  2.90000000e+01,   4.70000000e+01,   2.23606798e-01},
			new double[]{  2.50000000e+01,   2.90000000e+01,   2.23606798e-01},
			new double[]{  4.00000000e+00,   2.60000000e+01,   2.44948974e-01},
			new double[]{  2.60000000e+01,   4.00000000e+01,   2.44948974e-01},
			new double[]{  4.00000000e+01,   4.80000000e+01,   2.44948974e-01},
			new double[]{  4.70000000e+01,   3.00000000e+00,   2.44948974e-01},
			new double[]{  9.90000000e+01,   9.60000000e+01,   2.44948974e-01},
			new double[]{  4.80000000e+01,   9.00000000e+00,   2.64575131e-01},
			new double[]{  3.00000000e+00,   1.90000000e+01,   2.64575131e-01},
			new double[]{  1.90000000e+01,   2.00000000e+00,   2.64575131e-01},
			new double[]{  2.00000000e+00,   2.10000000e+01,   2.64575131e-01},
			new double[]{  2.10000000e+01,   4.50000000e+01,   2.64575131e-01},
			new double[]{  9.20000000e+01,   9.90000000e+01,   2.64575131e-01},
			new double[]{  6.90000000e+01,   9.20000000e+01,   2.64575131e-01},
			new double[]{  4.50000000e+01,   3.80000000e+01,   3.00000000e-01},
			new double[]{  4.20000000e+01,   4.60000000e+01,   3.00000000e-01},
			new double[]{  3.80000000e+01,   1.10000000e+01,   3.00000000e-01},
			new double[]{  1.10000000e+01,   4.20000000e+01,   3.00000000e-01},
			new double[]{  8.00000000e+01,   6.90000000e+01,   3.00000000e-01},
			new double[]{  8.90000000e+01,   9.40000000e+01,   3.00000000e-01},
			new double[]{  8.20000000e+01,   8.90000000e+01,   3.00000000e-01},
			new double[]{  9.60000000e+01,   8.20000000e+01,   3.00000000e-01},
			new double[]{  4.60000000e+01,   6.00000000e+00,   3.16227766e-01},
			new double[]{  1.27000000e+02,   1.38000000e+02,   3.16227766e-01},
			new double[]{  6.00000000e+00,   3.10000000e+01,   3.16227766e-01},
			new double[]{  9.40000000e+01,   8.80000000e+01,   3.16227766e-01},
			new double[]{  5.80000000e+01,   7.50000000e+01,   3.16227766e-01},
			new double[]{  8.80000000e+01,   5.50000000e+01,   3.31662479e-01},
			new double[]{  3.10000000e+01,   3.50000000e+01,   3.31662479e-01},
			new double[]{  7.50000000e+01,   8.60000000e+01,   3.31662479e-01},
			new double[]{  1.38000000e+02,   1.49000000e+02,   3.31662479e-01},
			new double[]{  1.49000000e+02,   1.01000000e+02,   3.31662479e-01},
			new double[]{  1.01000000e+02,   1.42000000e+02,   3.31662479e-01},
			new double[]{  5.50000000e+01,   9.50000000e+01,   3.31662479e-01},
			new double[]{  3.50000000e+01,   1.00000000e+01,   3.31662479e-01},
			new double[]{  8.60000000e+01,   6.50000000e+01,   3.46410162e-01},
			new double[]{  1.00000000e+01,   8.00000000e+00,   3.46410162e-01},
			new double[]{  8.00000000e+00,   3.60000000e+01,   3.46410162e-01},
			new double[]{  7.80000000e+01,   9.10000000e+01,   3.46410162e-01},
			new double[]{  9.10000000e+01,   9.70000000e+01,   3.46410162e-01},
			new double[]{  1.20000000e+02,   1.40000000e+02,   3.46410162e-01},
			new double[]{  1.40000000e+02,   1.43000000e+02,   3.46410162e-01},
			new double[]{  6.50000000e+01,   5.20000000e+01,   3.46410162e-01},
			new double[]{  9.50000000e+01,   6.70000000e+01,   3.60555128e-01},
			new double[]{  1.23000000e+02,   1.27000000e+02,   3.60555128e-01},
			new double[]{  3.60000000e+01,   2.00000000e+01,   3.60555128e-01},
			new double[]{  6.10000000e+01,   7.80000000e+01,   3.60555128e-01},
			new double[]{  6.70000000e+01,   6.10000000e+01,   3.60555128e-01},
			new double[]{  1.12000000e+02,   1.20000000e+02,   3.74165739e-01},
			new double[]{  1.45000000e+02,   1.12000000e+02,   3.74165739e-01},
			new double[]{  1.47000000e+02,   1.45000000e+02,   3.74165739e-01},
			new double[]{  1.42000000e+02,   8.30000000e+01,   3.74165739e-01},
			new double[]{  1.43000000e+02,   1.39000000e+02,   3.74165739e-01},
			new double[]{  1.39000000e+02,   1.24000000e+02,   3.74165739e-01},
			new double[]{  5.20000000e+01,   5.10000000e+01,   3.74165739e-01},
			new double[]{  5.10000000e+01,   7.60000000e+01,   3.74165739e-01},
			new double[]{  2.00000000e+01,   4.30000000e+01,   3.74165739e-01},
			new double[]{  5.40000000e+01,   5.80000000e+01,   3.74165739e-01},
			new double[]{  4.30000000e+01,   5.00000000e+00,   3.74165739e-01},
			new double[]{  7.40000000e+01,   5.40000000e+01,   3.87298335e-01},
			new double[]{  9.70000000e+01,   7.40000000e+01,   3.87298335e-01},
			new double[]{  1.24000000e+02,   1.15000000e+02,   3.87298335e-01},
			new double[]{  1.16000000e+02,   1.28000000e+02,   3.87298335e-01},
			new double[]{  1.03000000e+02,   1.16000000e+02,   3.87298335e-01},
			new double[]{  1.11000000e+02,   1.03000000e+02,   3.87298335e-01},
			new double[]{  1.15000000e+02,   1.11000000e+02,   3.87298335e-01},
			new double[]{  8.30000000e+01,   1.26000000e+02,   3.87298335e-01},
			new double[]{  1.28000000e+02,   1.04000000e+02,   3.87298335e-01},
			new double[]{  5.00000000e+00,   2.30000000e+01,   3.87298335e-01},
			new double[]{  2.30000000e+01,   1.60000000e+01,   3.87298335e-01},
			new double[]{  7.60000000e+01,   7.10000000e+01,   4.00000000e-01},
			new double[]{  1.04000000e+02,   1.44000000e+02,   4.00000000e-01},
			new double[]{  1.46000000e+02,   1.23000000e+02,   4.12310563e-01},
			new double[]{  1.44000000e+02,   1.46000000e+02,   4.12310563e-01},
			new double[]{  1.60000000e+01,   4.40000000e+01,   4.12310563e-01},
			new double[]{  4.40000000e+01,   3.30000000e+01,   4.12310563e-01},
			new double[]{  7.10000000e+01,   6.60000000e+01,   4.12310563e-01},
			new double[]{  3.30000000e+01,   3.20000000e+01,   4.24264069e-01},
			new double[]{  1.26000000e+02,   7.00000000e+01,   4.24264069e-01},
			new double[]{  7.00000000e+01,   7.20000000e+01,   4.24264069e-01},
			new double[]{  3.20000000e+01,   2.40000000e+01,   4.24264069e-01},
			new double[]{  7.70000000e+01,   1.47000000e+02,   4.24264069e-01},
			new double[]{  6.60000000e+01,   7.70000000e+01,   4.24264069e-01},
			new double[]{  7.20000000e+01,   1.10000000e+02,   4.24264069e-01},
			new double[]{  1.10000000e+02,   9.00000000e+01,   4.24264069e-01},
			new double[]{  9.00000000e+01,   6.30000000e+01,   4.24264069e-01},
			new double[]{  6.30000000e+01,   1.33000000e+02,   4.35889894e-01},
			new double[]{  5.30000000e+01,   8.10000000e+01,   4.35889894e-01},
			new double[]{  1.33000000e+02,   5.30000000e+01,   4.35889894e-01},
			new double[]{  8.10000000e+01,   1.32000000e+02,   4.35889894e-01},
			new double[]{  1.32000000e+02,   1.37000000e+02,   4.35889894e-01},
			new double[]{  7.30000000e+01,   1.36000000e+02,   4.35889894e-01},
			new double[]{  1.37000000e+02,   7.30000000e+01,   4.35889894e-01},
			new double[]{  1.36000000e+02,   7.90000000e+01,   4.47213595e-01},
			new double[]{  7.90000000e+01,   5.60000000e+01,   4.58257569e-01},
			new double[]{  5.60000000e+01,   1.21000000e+02,   4.58257569e-01},
			new double[]{  1.21000000e+02,   5.00000000e+01,   4.58257569e-01},
			new double[]{  5.00000000e+01,   1.02000000e+02,   4.58257569e-01},
			new double[]{  1.02000000e+02,   8.50000000e+01,   4.69041576e-01},
			new double[]{  8.50000000e+01,   1.25000000e+02,   4.69041576e-01},
			new double[]{  2.40000000e+01,   1.30000000e+01,   4.79583152e-01},
			new double[]{  1.25000000e+02,   8.40000000e+01,   4.89897949e-01},
			new double[]{  1.30000000e+01,   1.80000000e+01,   5.09901951e-01},
			new double[]{  8.40000000e+01,   1.30000000e+02,   5.09901951e-01},
			new double[]{  1.30000000e+02,   1.41000000e+02,   5.09901951e-01},
			new double[]{  1.41000000e+02,   6.40000000e+01,   5.19615242e-01},
			new double[]{  6.40000000e+01,   1.13000000e+02,   5.19615242e-01},
			new double[]{  1.13000000e+02,   1.14000000e+02,   5.19615242e-01},
			new double[]{  1.14000000e+02,   5.90000000e+01,   5.29150262e-01},
			new double[]{  1.80000000e+01,   2.20000000e+01,   5.38516481e-01},
			new double[]{  5.90000000e+01,   1.07000000e+02,   5.47722558e-01},
			new double[]{  1.07000000e+02,   1.05000000e+02,   5.47722558e-01},
			new double[]{  1.05000000e+02,   1.29000000e+02,   5.56776436e-01},
			new double[]{  2.20000000e+01,   1.40000000e+01,   5.56776436e-01},
			new double[]{  1.29000000e+02,   1.00000000e+02,   5.56776436e-01},
			new double[]{  1.00000000e+02,   1.19000000e+02,   5.83095189e-01},
			new double[]{  1.19000000e+02,   6.20000000e+01,   5.83095189e-01},
			new double[]{  6.20000000e+01,   8.70000000e+01,   6.08276253e-01},
			new double[]{  8.70000000e+01,   1.48000000e+02,   6.16441400e-01},
			new double[]{  1.48000000e+02,   1.08000000e+02,   6.16441400e-01},
			new double[]{  1.40000000e+01,   1.50000000e+01,   6.16441400e-01},
			new double[]{  1.08000000e+02,   9.30000000e+01,   6.48074070e-01},
			new double[]{  9.30000000e+01,   1.34000000e+02,   6.63324958e-01},
			new double[]{  1.34000000e+02,   6.80000000e+01,   6.78232998e-01},
			new double[]{  6.80000000e+01,   1.35000000e+02,   6.78232998e-01},
			new double[]{  1.35000000e+02,   1.22000000e+02,   6.78232998e-01},
			new double[]{  6.00000000e+01,   8.00000000e+01,   7.14142843e-01},
			new double[]{  5.70000000e+01,   6.00000000e+01,   7.21110255e-01},
			new double[]{  1.22000000e+02,   1.09000000e+02,   7.54983444e-01},
			new double[]{  1.50000000e+01,   4.10000000e+01,   7.81024968e-01},
			new double[]{  9.80000000e+01,   5.70000000e+01,   7.93725393e-01},
			new double[]{  1.09000000e+02,   1.06000000e+02,   8.77496439e-01},
			new double[]{  1.06000000e+02,   1.18000000e+02,   9.27361850e-01},
			new double[]{  1.18000000e+02,   1.31000000e+02,   9.32737905e-01},
			new double[]{  1.31000000e+02,   1.17000000e+02,   1.00498756e+00},
			new double[]{  4.10000000e+01,   9.80000000e+01,   1.64012195e+00}
		};
		
		//System.out.println(Arrays.toString(VecUtils.argSort(MatUtils.getColumn(cdist, 2))));
		//fail();
		
		
		/*
		 * In comparison to sklearn, this can get off by 1 or 2 due to
		 * rounding errors in the sort. So lets just make sure there is a
		 * discrepancy less than one or two.
		 */
		int wrong_ct = 0;
		for(int i = 0; i < srtd_cdists.length; i++) {
			if(!VecUtils.equalsWithTolerance(srtd_cdists[i], expected_srted[i], 1e-8)) {
				if(!Precision.equals(srtd_cdists[i][2], expected_srted[i][2], 1e-8))
					wrong_ct++;
			}
		}
		
		assertTrue(wrong_ct < 2);
		
		
		// Do labeling
		double[][] labMat = HDBSCAN.label(srtd_cdists);
		
		double[][] expected_labMat = new double[][]{
			new double[]{  0.00000000e+00,   3.90000000e+01,   1.41421356e-01, 2.00000000e+00 },
			new double[]{  1.50000000e+02,   1.70000000e+01,   1.73205081e-01, 3.00000000e+00 },
			new double[]{  1.51000000e+02,   2.70000000e+01,   1.73205081e-01, 4.00000000e+00 },
			new double[]{  1.52000000e+02,   2.80000000e+01,   1.73205081e-01, 5.00000000e+00 },
			new double[]{  3.40000000e+01,   3.70000000e+01,   1.73205081e-01, 2.00000000e+00 },
			new double[]{  3.00000000e+01,   1.54000000e+02,   1.73205081e-01, 3.00000000e+00 },
			new double[]{  9.00000000e+00,   1.00000000e+00,   1.73205081e-01, 2.00000000e+00 },
			new double[]{  1.56000000e+02,   1.55000000e+02,   1.73205081e-01, 5.00000000e+00 },
			new double[]{  1.57000000e+02,   1.20000000e+01,   1.73205081e-01, 6.00000000e+00 },
			new double[]{  1.53000000e+02,   7.00000000e+00,   2.00000000e-01, 6.00000000e+00 },
			new double[]{  1.58000000e+02,   2.50000000e+01,   2.23606798e-01, 7.00000000e+00 },
			new double[]{  1.59000000e+02,   4.90000000e+01,   2.23606798e-01, 7.00000000e+00 },
			new double[]{  1.61000000e+02,   4.00000000e+00,   2.23606798e-01, 8.00000000e+00 },
			new double[]{  2.90000000e+01,   4.70000000e+01,   2.23606798e-01, 2.00000000e+00 },
			new double[]{  1.60000000e+02,   1.63000000e+02,   2.23606798e-01, 9.00000000e+00 },
			new double[]{  1.62000000e+02,   2.60000000e+01,   2.44948974e-01, 9.00000000e+00 },
			new double[]{  1.65000000e+02,   4.00000000e+01,   2.44948974e-01, 1.00000000e+01 },
			new double[]{  1.66000000e+02,   4.80000000e+01,   2.44948974e-01, 1.10000000e+01 },
			new double[]{  1.64000000e+02,   3.00000000e+00,   2.44948974e-01, 1.00000000e+01 },
			new double[]{  9.90000000e+01,   9.60000000e+01,   2.44948974e-01, 2.00000000e+00 },
			new double[]{  1.67000000e+02,   1.68000000e+02,   2.64575131e-01, 2.10000000e+01 },
			new double[]{  1.70000000e+02,   1.90000000e+01,   2.64575131e-01, 2.20000000e+01 },
			new double[]{  1.71000000e+02,   2.00000000e+00,   2.64575131e-01, 2.30000000e+01 },
			new double[]{  1.72000000e+02,   2.10000000e+01,   2.64575131e-01, 2.40000000e+01 },
			new double[]{  1.73000000e+02,   4.50000000e+01,   2.64575131e-01, 2.50000000e+01 },
			new double[]{  9.20000000e+01,   1.69000000e+02,   2.64575131e-01, 3.00000000e+00 },
			new double[]{  6.90000000e+01,   1.75000000e+02,   2.64575131e-01, 4.00000000e+00 },
			new double[]{  1.74000000e+02,   3.80000000e+01,   3.00000000e-01, 2.60000000e+01 },
			new double[]{  4.20000000e+01,   4.60000000e+01,   3.00000000e-01, 2.00000000e+00 },
			new double[]{  1.77000000e+02,   1.10000000e+01,   3.00000000e-01, 2.70000000e+01 },
			new double[]{  1.79000000e+02,   1.78000000e+02,   3.00000000e-01, 2.90000000e+01 },
			new double[]{  8.00000000e+01,   1.76000000e+02,   3.00000000e-01, 5.00000000e+00 },
			new double[]{  8.90000000e+01,   9.40000000e+01,   3.00000000e-01, 2.00000000e+00 },
			new double[]{  8.20000000e+01,   1.82000000e+02,   3.00000000e-01, 3.00000000e+00 },
			new double[]{  1.81000000e+02,   1.83000000e+02,   3.00000000e-01, 8.00000000e+00 },
			new double[]{  1.80000000e+02,   6.00000000e+00,   3.16227766e-01, 3.00000000e+01 },
			new double[]{  1.27000000e+02,   1.38000000e+02,   3.16227766e-01, 2.00000000e+00 },
			new double[]{  1.85000000e+02,   3.10000000e+01,   3.16227766e-01, 3.10000000e+01 },
			new double[]{  1.84000000e+02,   8.80000000e+01,   3.16227766e-01, 9.00000000e+00 },
			new double[]{  5.80000000e+01,   7.50000000e+01,   3.16227766e-01, 2.00000000e+00 },
			new double[]{  1.88000000e+02,   5.50000000e+01,   3.31662479e-01, 1.00000000e+01 },
			new double[]{  1.87000000e+02,   3.50000000e+01,   3.31662479e-01, 3.20000000e+01 },
			new double[]{  1.89000000e+02,   8.60000000e+01,   3.31662479e-01, 3.00000000e+00 },
			new double[]{  1.86000000e+02,   1.49000000e+02,   3.31662479e-01, 3.00000000e+00 },
			new double[]{  1.93000000e+02,   1.01000000e+02,   3.31662479e-01, 4.00000000e+00 },
			new double[]{  1.94000000e+02,   1.42000000e+02,   3.31662479e-01, 5.00000000e+00 },
			new double[]{  1.90000000e+02,   9.50000000e+01,   3.31662479e-01, 1.10000000e+01 },
			new double[]{  1.91000000e+02,   1.00000000e+01,   3.31662479e-01, 3.30000000e+01 },
			new double[]{  1.92000000e+02,   6.50000000e+01,   3.46410162e-01, 4.00000000e+00 },
			new double[]{  1.97000000e+02,   8.00000000e+00,   3.46410162e-01, 3.40000000e+01 },
			new double[]{  1.99000000e+02,   3.60000000e+01,   3.46410162e-01, 3.50000000e+01 },
			new double[]{  7.80000000e+01,   9.10000000e+01,   3.46410162e-01, 2.00000000e+00 },
			new double[]{  2.01000000e+02,   9.70000000e+01,   3.46410162e-01, 3.00000000e+00 },
			new double[]{  1.20000000e+02,   1.40000000e+02,   3.46410162e-01, 2.00000000e+00 },
			new double[]{  2.03000000e+02,   1.43000000e+02,   3.46410162e-01, 3.00000000e+00 },
			new double[]{  1.98000000e+02,   5.20000000e+01,   3.46410162e-01, 5.00000000e+00 },
			new double[]{  1.96000000e+02,   6.70000000e+01,   3.60555128e-01, 1.20000000e+01 },
			new double[]{  1.23000000e+02,   1.95000000e+02,   3.60555128e-01, 6.00000000e+00 },
			new double[]{  2.00000000e+02,   2.00000000e+01,   3.60555128e-01, 3.60000000e+01 },
			new double[]{  6.10000000e+01,   2.02000000e+02,   3.60555128e-01, 4.00000000e+00 },
			new double[]{  2.06000000e+02,   2.09000000e+02,   3.60555128e-01, 1.60000000e+01 },
			new double[]{  1.12000000e+02,   2.04000000e+02,   3.74165739e-01, 4.00000000e+00 },
			new double[]{  1.45000000e+02,   2.11000000e+02,   3.74165739e-01, 5.00000000e+00 },
			new double[]{  1.47000000e+02,   2.12000000e+02,   3.74165739e-01, 6.00000000e+00 },
			new double[]{  2.07000000e+02,   8.30000000e+01,   3.74165739e-01, 7.00000000e+00 },
			new double[]{  2.13000000e+02,   1.39000000e+02,   3.74165739e-01, 7.00000000e+00 },
			new double[]{  2.15000000e+02,   1.24000000e+02,   3.74165739e-01, 8.00000000e+00 },
			new double[]{  2.05000000e+02,   5.10000000e+01,   3.74165739e-01, 6.00000000e+00 },
			new double[]{  2.17000000e+02,   7.60000000e+01,   3.74165739e-01, 7.00000000e+00 },
			new double[]{  2.08000000e+02,   4.30000000e+01,   3.74165739e-01, 3.70000000e+01 },
			new double[]{  5.40000000e+01,   2.18000000e+02,   3.74165739e-01, 8.00000000e+00 },
			new double[]{  2.19000000e+02,   5.00000000e+00,   3.74165739e-01, 3.80000000e+01 },
			new double[]{  7.40000000e+01,   2.20000000e+02,   3.87298335e-01, 9.00000000e+00 },
			new double[]{  2.10000000e+02,   2.22000000e+02,   3.87298335e-01, 2.50000000e+01 },
			new double[]{  2.16000000e+02,   1.15000000e+02,   3.87298335e-01, 9.00000000e+00 },
			new double[]{  1.16000000e+02,   1.28000000e+02,   3.87298335e-01, 2.00000000e+00 },
			new double[]{  1.03000000e+02,   2.25000000e+02,   3.87298335e-01, 3.00000000e+00 },
			new double[]{  1.11000000e+02,   2.26000000e+02,   3.87298335e-01, 4.00000000e+00 },
			new double[]{  2.24000000e+02,   2.27000000e+02,   3.87298335e-01, 1.30000000e+01 },
			new double[]{  2.14000000e+02,   1.26000000e+02,   3.87298335e-01, 8.00000000e+00 },
			new double[]{  2.28000000e+02,   1.04000000e+02,   3.87298335e-01, 1.40000000e+01 },
			new double[]{  2.21000000e+02,   2.30000000e+01,   3.87298335e-01, 3.90000000e+01 },
			new double[]{  2.31000000e+02,   1.60000000e+01,   3.87298335e-01, 4.00000000e+01 },
			new double[]{  2.23000000e+02,   7.10000000e+01,   4.00000000e-01, 2.60000000e+01 },
			new double[]{  2.30000000e+02,   1.44000000e+02,   4.00000000e-01, 1.50000000e+01 },
			new double[]{  1.46000000e+02,   2.29000000e+02,   4.12310563e-01, 9.00000000e+00 },
			new double[]{  2.34000000e+02,   2.35000000e+02,   4.12310563e-01, 2.40000000e+01 },
			new double[]{  2.32000000e+02,   4.40000000e+01,   4.12310563e-01, 4.10000000e+01 },
			new double[]{  2.37000000e+02,   3.30000000e+01,   4.12310563e-01, 4.20000000e+01 },
			new double[]{  2.33000000e+02,   6.60000000e+01,   4.12310563e-01, 2.70000000e+01 },
			new double[]{  2.38000000e+02,   3.20000000e+01,   4.24264069e-01, 4.30000000e+01 },
			new double[]{  2.36000000e+02,   7.00000000e+01,   4.24264069e-01, 2.50000000e+01 },
			new double[]{  2.41000000e+02,   7.20000000e+01,   4.24264069e-01, 2.60000000e+01 },
			new double[]{  2.40000000e+02,   2.40000000e+01,   4.24264069e-01, 4.40000000e+01 },
			new double[]{  7.70000000e+01,   2.42000000e+02,   4.24264069e-01, 2.70000000e+01 },
			new double[]{  2.39000000e+02,   2.44000000e+02,   4.24264069e-01, 5.40000000e+01 },
			new double[]{  2.45000000e+02,   1.10000000e+02,   4.24264069e-01, 5.50000000e+01 },
			new double[]{  2.46000000e+02,   9.00000000e+01,   4.24264069e-01, 5.60000000e+01 },
			new double[]{  2.47000000e+02,   6.30000000e+01,   4.24264069e-01, 5.70000000e+01 },
			new double[]{  2.48000000e+02,   1.33000000e+02,   4.35889894e-01, 5.80000000e+01 },
			new double[]{  5.30000000e+01,   8.10000000e+01,   4.35889894e-01, 2.00000000e+00 },
			new double[]{  2.49000000e+02,   2.50000000e+02,   4.35889894e-01, 6.00000000e+01 },
			new double[]{  2.51000000e+02,   1.32000000e+02,   4.35889894e-01, 6.10000000e+01 },
			new double[]{  2.52000000e+02,   1.37000000e+02,   4.35889894e-01, 6.20000000e+01 },
			new double[]{  7.30000000e+01,   1.36000000e+02,   4.35889894e-01, 2.00000000e+00 },
			new double[]{  2.53000000e+02,   2.54000000e+02,   4.35889894e-01, 6.40000000e+01 },
			new double[]{  2.55000000e+02,   7.90000000e+01,   4.47213595e-01, 6.50000000e+01 },
			new double[]{  2.56000000e+02,   5.60000000e+01,   4.58257569e-01, 6.60000000e+01 },
			new double[]{  2.57000000e+02,   1.21000000e+02,   4.58257569e-01, 6.70000000e+01 },
			new double[]{  2.58000000e+02,   5.00000000e+01,   4.58257569e-01, 6.80000000e+01 },
			new double[]{  2.59000000e+02,   1.02000000e+02,   4.58257569e-01, 6.90000000e+01 },
			new double[]{  2.60000000e+02,   8.50000000e+01,   4.69041576e-01, 7.00000000e+01 },
			new double[]{  2.61000000e+02,   1.25000000e+02,   4.69041576e-01, 7.10000000e+01 },
			new double[]{  2.43000000e+02,   1.30000000e+01,   4.79583152e-01, 4.50000000e+01 },
			new double[]{  2.62000000e+02,   8.40000000e+01,   4.89897949e-01, 7.20000000e+01 },
			new double[]{  2.63000000e+02,   1.80000000e+01,   5.09901951e-01, 4.60000000e+01 },
			new double[]{  2.64000000e+02,   1.30000000e+02,   5.09901951e-01, 7.30000000e+01 },
			new double[]{  2.66000000e+02,   1.41000000e+02,   5.09901951e-01, 7.40000000e+01 },
			new double[]{  2.67000000e+02,   6.40000000e+01,   5.19615242e-01, 7.50000000e+01 },
			new double[]{  2.68000000e+02,   1.13000000e+02,   5.19615242e-01, 7.60000000e+01 },
			new double[]{  2.69000000e+02,   1.14000000e+02,   5.19615242e-01, 7.70000000e+01 },
			new double[]{  2.70000000e+02,   5.90000000e+01,   5.29150262e-01, 7.80000000e+01 },
			new double[]{  2.65000000e+02,   2.20000000e+01,   5.38516481e-01, 4.70000000e+01 },
			new double[]{  2.71000000e+02,   1.07000000e+02,   5.47722558e-01, 7.90000000e+01 },
			new double[]{  2.73000000e+02,   1.05000000e+02,   5.47722558e-01, 8.00000000e+01 },
			new double[]{  2.74000000e+02,   1.29000000e+02,   5.56776436e-01, 8.10000000e+01 },
			new double[]{  2.72000000e+02,   1.40000000e+01,   5.56776436e-01, 4.80000000e+01 },
			new double[]{  2.75000000e+02,   1.00000000e+02,   5.56776436e-01, 8.20000000e+01 },
			new double[]{  2.77000000e+02,   1.19000000e+02,   5.83095189e-01, 8.30000000e+01 },
			new double[]{  2.78000000e+02,   6.20000000e+01,   5.83095189e-01, 8.40000000e+01 },
			new double[]{  2.79000000e+02,   8.70000000e+01,   6.08276253e-01, 8.50000000e+01 },
			new double[]{  2.80000000e+02,   1.48000000e+02,   6.16441400e-01, 8.60000000e+01 },
			new double[]{  2.81000000e+02,   1.08000000e+02,   6.16441400e-01, 8.70000000e+01 },
			new double[]{  2.76000000e+02,   1.50000000e+01,   6.16441400e-01, 4.90000000e+01 },
			new double[]{  2.82000000e+02,   9.30000000e+01,   6.48074070e-01, 8.80000000e+01 },
			new double[]{  2.84000000e+02,   1.34000000e+02,   6.63324958e-01, 8.90000000e+01 },
			new double[]{  2.85000000e+02,   6.80000000e+01,   6.78232998e-01, 9.00000000e+01 },
			new double[]{  2.86000000e+02,   1.35000000e+02,   6.78232998e-01, 9.10000000e+01 },
			new double[]{  2.87000000e+02,   1.22000000e+02,   6.78232998e-01, 9.20000000e+01 },
			new double[]{  6.00000000e+01,   2.88000000e+02,   7.14142843e-01, 9.30000000e+01 },
			new double[]{  5.70000000e+01,   2.89000000e+02,   7.21110255e-01, 9.40000000e+01 },
			new double[]{  2.90000000e+02,   1.09000000e+02,   7.54983444e-01, 9.50000000e+01 },
			new double[]{  2.83000000e+02,   4.10000000e+01,   7.81024968e-01, 5.00000000e+01 },
			new double[]{  9.80000000e+01,   2.91000000e+02,   7.93725393e-01, 9.60000000e+01 },
			new double[]{  2.93000000e+02,   1.06000000e+02,   8.77496439e-01, 9.70000000e+01 },
			new double[]{  2.94000000e+02,   1.18000000e+02,   9.27361850e-01, 9.80000000e+01 },
			new double[]{  2.95000000e+02,   1.31000000e+02,   9.32737905e-01, 9.90000000e+01 },
			new double[]{  2.96000000e+02,   1.17000000e+02,   1.00498756e+00, 1.00000000e+02 },
			new double[]{  2.92000000e+02,   2.97000000e+02,   1.64012195e+00, 1.50000000e+02 }
		};

		// ensure the labeling method works and gets what the sklearn method would...
		assertTrue(MatUtils.equalsExactly(expected_labMat, HDBSCAN.label(expected_srted)));
		
		
		// expected sorted...
		ArrayList<CompQuadTup<Integer, Integer, Double, Integer>> expected_hlist = new ArrayList<>();
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(150, 151, 0.6097107608496923, 50));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(150, 152, 0.6097107608496923, 100));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 2, 3.7796447300922726, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 5, 2.6726124191242397, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 6, 3.1622776601683857, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 8, 2.886751345948128, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 10, 3.015113445777631, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 11, 3.3333333333333353, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 13, 2.0851441405707485, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 14, 1.796053020267749, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 15, 1.6222142113076248, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 16, 2.5819888974716076, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 18, 1.9611613513818411, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 19, 3.7796447300922766, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 20, 2.773500981126144, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 21, 3.7796447300922726, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 22, 1.85695338177052, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 23, 2.581988897471612, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 24, 2.35702260395516, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 31, 3.1622776601683804, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 32, 2.3570226039551616, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 33, 2.42535625036333, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 35, 3.0151134457776374, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 36, 2.8867513459481273, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 38, 3.3333333333333384, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 41, 1.2803687993289594, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 42, 3.3333333333333353, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 43, 2.6726124191242437, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 44, 2.4253562503633304, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 45, 3.7796447300922726, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 46, 3.3333333333333353, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 153, 3.7796447300922766, 11));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(151, 154, 3.7796447300922766, 10));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 50, 2.1821789023599227, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 53, 2.294157338705618, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 56, 2.1821789023599236, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 57, 1.386750490563073, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 59, 1.8898223650461363, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 60, 1.4002800840280099, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 62, 1.7149858514250882, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 63, 2.357022603955156, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 64, 1.9245008972987536, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 68, 1.4744195615489724, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 73, 2.2941573387056153, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 79, 2.2360679774997894, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 81, 2.294157338705618, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 84, 2.041241452319315, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 85, 2.132007163556105, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 87, 1.6439898730535734, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 90, 2.3570226039551567, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 93, 1.5430334996209187, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 98, 1.2598815766974234, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 100, 1.796053020267749, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 102, 2.1821789023599227, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 105, 1.8257418583505527, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 106, 1.1396057645963797, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 107, 1.8257418583505547, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 108, 1.6222142113076254, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 109, 1.3245323570650438, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 110, 2.3570226039551576, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 113, 1.9245008972987536, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 114, 1.9245008972987536, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 117, 0.995037190209989, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 118, 1.0783277320343838, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 119, 1.7149858514250889, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 121, 2.1821789023599227, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 122, 1.4744195615489704, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 125, 2.132007163556103, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 129, 1.7960530202677494, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 130, 1.9611613513818407, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 131, 1.0721125348377945, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 132, 2.294157338705618, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 133, 2.2941573387056184, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 134, 1.5075567228888174, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 135, 1.474419561548971, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 136, 2.2941573387056153, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 137, 2.294157338705617, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 141, 1.9611613513818398, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 148, 1.6222142113076257, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 155, 2.3570226039551576, 27));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(152, 156, 2.3570226039551576, 27));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 0, 5.773502691896247, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 4, 4.472135954999576, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 7, 5.000000000000003, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 17, 5.773502691896247, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 26, 4.082482904638632, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 27, 5.773502691896247, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 28, 5.773502691896247, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 39, 5.773502691896247, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 40, 4.08248290463863, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 48, 4.0824829046386295, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(153, 49, 4.47213595499958, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 1, 5.773502691896246, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 3, 4.082482904638627, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 9, 5.773502691896246, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 12, 5.773502691896245, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 25, 4.47213595499958, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 29, 4.472135954999572, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 30, 5.773502691896246, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 34, 5.773502691896246, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 37, 5.773502691896246, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(154, 47, 4.472135954999572, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(155, 66, 2.4253562503633277, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(155, 71, 2.5000000000000013, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(155, 157, 2.5819888974716125, 16));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(155, 158, 2.5819888974716125, 9));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(156, 70, 2.3570226039551603, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(156, 72, 2.3570226039551603, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(156, 77, 2.3570226039551576, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(156, 159, 2.425356250363331, 15));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(156, 160, 2.425356250363331, 9));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 55, 3.0151134457776374, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 61, 2.7735009811261433, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 67, 2.773500981126145, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 69, 3.333333333333332, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 78, 2.7735009811261433, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 80, 3.333333333333332, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 82, 3.3333333333333317, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 88, 3.162277660168379, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 89, 3.3333333333333317, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 91, 2.7735009811261433, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 92, 3.333333333333332, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 94, 3.3333333333333317, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 95, 3.015113445777636, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 96, 3.333333333333332, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 97, 2.7735009811261433, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(157, 99, 3.333333333333332, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 51, 2.672612419124244, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 52, 2.886751345948124, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 54, 2.6726124191242406, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 58, 2.886751345948124, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 65, 2.886751345948124, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 74, 2.5819888974716125, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 75, 2.886751345948124, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 76, 2.672612419124244, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(158, 86, 2.886751345948124, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 103, 2.581988897471612, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 104, 2.581988897471612, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 111, 2.581988897471612, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 112, 2.6726124191242464, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 115, 2.5819888974716125, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 116, 2.581988897471612, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 120, 2.6726124191242464, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 124, 2.6726124191242446, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 128, 2.581988897471612, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 139, 2.6726124191242455, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 140, 2.6726124191242464, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 143, 2.6726124191242464, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 144, 2.5000000000000004, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 145, 2.6726124191242464, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(159, 147, 2.6726124191242464, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 83, 2.672612419124246, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 101, 3.0151134457776365, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 123, 2.7735009811261446, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 126, 2.581988897471612, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 127, 3.0151134457776365, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 138, 3.0151134457776365, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 142, 3.0151134457776365, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 146, 2.425356250363331, 1));
		expected_hlist.add(new CompQuadTup<Integer, Integer, Double, Integer>(160, 149, 3.0151134457776365, 1));
		
		
		// test the condense tree label
		ArrayList<CompQuadTup<Integer, Integer, Double, Integer>> condensed = 
			HDBSCAN.LinkageTreeUtils.condenseTree(expected_labMat, 5);
		// Now sort it for the sake of comparing to the sklearn res...
		Collections.sort(condensed, new Comparator<QuadTup<Integer, Integer, Double, Integer>>(){
			@Override
			public int compare(QuadTup<Integer, Integer, Double, Integer> q1, 
					QuadTup<Integer, Integer, Double, Integer> q2) {
				int cmp = q1.getFirst().compareTo(q2.getFirst());
				
				if(cmp == 0) {
					cmp = q1.getSecond().compareTo(q2.getSecond());
					
					if(cmp == 0) {
						cmp = q1.getThird().compareTo(q2.getThird());
						
						if(cmp == 0) {
							return q1.getFourth().compareTo(q2.getFourth());
						}
						
						return cmp;
					}
					
					return cmp;
				}
				
				return cmp;
			}
		});
		
		for(int i = 0; i < condensed.size(); i++) {
			if(!condensed.get(i).almostEquals(expected_hlist.get(i))) {
				System.out.println(condensed.get(i));
				System.out.println(expected_hlist.get(i));
				fail();
			}
		}
		
		
		// If we get here, the condensed labels works!!
		TreeMap<Integer, Double> stability = HDBSCAN.LinkageTreeUtils.computeStability(condensed);
		TreeMap<Integer, Double> exp_stab  = new TreeMap<>();
		exp_stab.put(150, Double.NaN);
		exp_stab.put(151, 128.9165546745262);
		exp_stab.put(152, 150.98635723043549);
		exp_stab.put(153, 13.48314205238124);
		exp_stab.put(154, 14.343459620092055);
		exp_stab.put(155, 5.8354683803643868);
		exp_stab.put(156, 1.6400075137961618);
		exp_stab.put(157, 8.4148537644752253);
		exp_stab.put(158, 1.7956828073404498);
		exp_stab.put(159, 2.99248898237368);
		exp_stab.put(160, Double.NaN);
		
		/*
		 * Assert near equality...
		 */
		for(Map.Entry<Integer, Double> entry: exp_stab.entrySet()) {
			int key = entry.getKey();
			double stab = entry.getValue();
			
			if(Double.isNaN(stab) && Double.isNaN(stability.get(key)))
				continue;
			if(!Precision.equals(stab, stability.get(key), 1e-6)) {
				System.out.println(key + ", " + stab);
				System.out.println(key + ", " + stability.get(key));
				fail();
			}
		}
		
		
		// test the treeToLabels method...
		final int[] labs = new NoiseyLabelEncoder(HDBSCAN.treeToLabels(iris.getData(), labMat, 5)).fit().getEncodedLabels();
		assertTrue(VecUtils.equalsExactly(labs, expected_iris_labs));
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
		
		int[] labels = new HDBSCAN(X, new HDBSCANParameters(1).setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, labels[0]).all()); // could be noise...
		
		labels = new HDBSCAN(X, new HDBSCANParameters().setVerbose(true)).fit().getLabels();
		assertTrue(new VecUtils.IntSeries(labels, Inequality.EQUAL_TO, labels[0]).all()); // could be noise...
	}
	
	@Test
	public void testValidMetrics() {
		HDBSCAN model = null;
		HDBSCAN_Algorithm algo;
		StandardScaler scaler = new StandardScaler().fit(iris);
		RealMatrix X = scaler.transform(iris);
		
		/*
		 * Generic first... should theoretically allow similarity metrics as well...
		 */
		algo = HDBSCAN_Algorithm.GENERIC;
		for(DistanceMetric d: Distance.values()) {
			model = new HDBSCAN(X, new HDBSCANParameters().setAlgo(algo).setMetric(d)).fit();
			
			if(!model.isValidMetric(d)) {
				assertTrue(model.hasWarnings());
				assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
			}
		}
		
		for(Kernel k: KernelTestCases.all_kernels) {
			model = new HDBSCAN(iris, new HDBSCANParameters().setAlgo(algo).setMetric(k)).fit();
			
			if(!model.isValidMetric(k)) {
				assertTrue(model.hasWarnings());
				assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
			}
		}
		
		
		
		/*
		 * Prims/Boruvka KD tree now -- first assert no warnings for KD valid and then assert warnings for others.
		 */
		for(HDBSCAN_Algorithm al: new HDBSCAN_Algorithm[]{HDBSCAN_Algorithm.PRIMS_KDTREE, HDBSCAN_Algorithm.BORUVKA_KDTREE}) {
			algo = al;
			boolean warnings_thrown = false;
			for(DistanceMetric d: new DistanceMetric[]{Distance.EUCLIDEAN, 
					Distance.MANHATTAN, Distance.CHEBYSHEV, new MinkowskiDistance(2.0)}) {
				model = new HDBSCAN(iris, new HDBSCANParameters().setAlgo(algo).setMetric(d)).fit();
				
				if(model.hasWarnings()) {
					warnings_thrown= true;
					System.out.println(d + ", " + model.getWarnings());
				}
			}
			
			assertFalse(warnings_thrown);
			model = new HDBSCAN(iris, new HDBSCANParameters().setAlgo(algo).setMetric(Distance.CANBERRA)).fit();
			assertTrue(model.hasWarnings());
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
			
			// try a few sim metrics to assert the same
			model = new HDBSCAN(iris, new HDBSCANParameters().setAlgo(algo).setMetric(Similarity.COSINE)).fit();
			assertTrue(model.hasWarnings());
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
		}
		
		
		
		/*
		 * Prims/Boruvka ball tree
		 */
		for(HDBSCAN_Algorithm al: new HDBSCAN_Algorithm[]{HDBSCAN_Algorithm.PRIMS_BALLTREE, HDBSCAN_Algorithm.BORUVKA_BALLTREE}) {
			algo = al;
			// need to use a smaller dataset here because haversine is an option...
			DataSet irisSmall = TestSuite.IRIS_DATASET.copy();
			irisSmall.dropCol(3);
			irisSmall.dropCol(2);
			final Array2DRowRealMatrix small = irisSmall.getData();
			
			for(Distance d: Distance.values()) {
				model = new HDBSCAN(small, new HDBSCANParameters().setAlgo(algo).setMetric(d)).fit();
				
				if(model.hasWarnings()) {
					assertTrue(!model.isValidMetric(d));
				}
			}
			
			// Try minkowski and haversine...
			model = new HDBSCAN(small, new HDBSCANParameters().setAlgo(algo).setMetric(new MinkowskiDistance(1.5))).fit();
			assertFalse(model.hasWarnings());
			model = new HDBSCAN(small, new HDBSCANParameters().setAlgo(algo).setMetric(Distance.HAVERSINE.MI)).fit();
			assertFalse(model.hasWarnings());
			
			// assert sim doesn't fly for ball tree...
			model = new HDBSCAN(small, new HDBSCANParameters().setAlgo(algo).setMetric(Similarity.COSINE)).fit();
			assertTrue(model.dist_metric.equals(Distance.EUCLIDEAN));
		}
	}
	
	@Test
	public void testAutoGeneric() {
		HDBSCAN h = new HDBSCAN(DATA, new HDBSCANParameters().setMetric(Distance.YULE)).fit();
		assertTrue(h.algo.equals(HDBSCAN.HDBSCAN_Algorithm.GENERIC));
		
		/*
		 * ensure unsupported operation exception here
		 */
		boolean a = false;
		try {
			HDBSCAN.HDBSCAN_Algorithm.AUTO.isValidMetric(null);
		} catch(UnsupportedOperationException u) {
			a =true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testBadLeafSize() {
		boolean a = false;
		try{
			new HDBSCAN(DATA, new HDBSCANParameters().setLeafSize(0));
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testAlpha() {
		/*
		 * Ensuring it actually fits it for all algos...
		 */
		for(HDBSCAN_Algorithm h: HDBSCAN.HDBSCAN_Algorithm.values())
			new HDBSCAN(DATA, new HDBSCANParameters().setAlgo(h).setAlpha(1.5)).fit();
	}
	
	@Test
	public void testWrapAroundWorks() {
		boolean b = false;
		try {
			HDBSCAN.LinkageTreeUtils.wraparoundIdxGet(4, 6);
		} catch(ArrayIndexOutOfBoundsException i) {
			b = true;
		} finally {
			assertTrue(b);
		}
	}
	
	@Test
	public void testUnionFindToStringNotNull() {
		assertNotNull(new HDBSCAN.UnionFind(4).toString());
	}
	
	@Test
	public void testDoubleFit() {
		HDBSCAN h = new HDBSCAN(DATA);
		
		/*
		 * First catch the MNFE
		 */
		boolean a = false;
		try {
			h.getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		assertTrue(h.getNumberOfIdentifiedClusters() == -1);
		assertTrue(h.getNumberOfNoisePoints() == -1);
		
		h.fit();
		assertTrue(h.equals(h.fit()));
		assertFalse(h.equals(new Object()));
		
		HDBSCAN i = new HDBSCAN(DATA).fit();
		assertTrue(h.getKey().equals(i.getKey()) || !h.equals(i));
		assertFalse(h.equals(new HDBSCAN(DATA)));
	}
	
	@Test
	public void testModelNotFit() {
		boolean a = false;
		try {
			new HDBSCAN(DATA).getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void testPredict() {
		HDBSCAN d = new HDBSCANParameters().fitNewModel(iris);
		
		/*
		 * Test for dim mismatch
		 */
		Array2DRowRealMatrix newData = new Array2DRowRealMatrix(new double[][]{
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
		
		/*
		 * Ensure unsupportedOperation
		 */
		newData = new Array2DRowRealMatrix(new double[][]{
			new double[]{150,150,150,150}
		}, false);
		a = false;
		try {
			d.predict(newData);
		} catch(UnsupportedOperationException u) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
}
