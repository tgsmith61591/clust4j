package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.TreeMap;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.HDBSCAN.Algorithm;
import com.clust4j.algo.HDBSCAN.HDBSCANPlanner;
import com.clust4j.algo.HDBSCAN.HList;
import com.clust4j.algo.HDBSCAN.LinkageTreeUtils;
import com.clust4j.algo.HDBSCAN.TreeUnionFind;
import com.clust4j.algo.HDBSCAN.UnionFind;
import com.clust4j.utils.BallTree;
import com.clust4j.utils.Distance;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.Inequality;
import com.clust4j.utils.KDTree;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.NearestNeighborHeapSearch;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.MatUtils.MatSeries;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.QuadTup;

public class HDBSCANTests {
	final static MatrixFormatter formatter = TestSuite.formatter;
	final static double[][] dist_mat = new double[][]{
		new double[]{1,2,3},
		new double[]{4,5,6},
		new double[]{7,8,9}
	};

	
	@Test
	public void testHDBSCANGenericMutualReachability() {
		final int m = dist_mat.length, minPts = 3;
		
		final int min_points = FastMath.min(m - 1, minPts);
		final double[] core_distances = MatUtils
			.partitionByRow(dist_mat, min_points)[min_points];
		
		final MatSeries ser1 = new MatSeries(core_distances, Inequality.GT, dist_mat);
		double[][] stage1 = MatUtils.where(ser1, core_distances, dist_mat);
		
		stage1 = MatUtils.transpose(stage1);
		final MatSeries ser2 = new MatSeries(core_distances, Inequality.GT, stage1);
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
		
		HList<QuadTup<Integer, Integer, Double, Integer>> h = HDBSCAN.LinkageTreeUtils.condenseTree(slt, 5);
		QuadTup<Integer, Integer, Double, Integer> q = h.get(0);
		assertTrue(q.one == 3);
		assertTrue(q.two == 0);
		// Three is a repeating decimal...
		assertTrue(q.four == 1);
		
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
				new HDBSCANPlanner(1)
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
		EntryPair<double[][], int[][]> query = tree.query(dist_mat, min_points, true, true, true);
		double[][] dists = query.getKey();
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
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.PRIMS_KDTREE)
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
		EntryPair<double[][], int[][]> query = tree.query(dist_mat, min_points, true, true, true);
		double[][] dists = query.getKey();
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
		
		HList<Integer> result;
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
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.PRIMS_BALLTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	/*@Test
	public void testBoruvkaKDRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.BORUVKA_KDTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}
	
	@Test
	public void testBoruvkaBallRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 
				new HDBSCANPlanner(1)
					.setAlgo(Algorithm.BORUVKA_BALLTREE)
					.setVerbose(true)).fit();
		int[] labels = model.getLabels();
		assertTrue(VecUtils.equalsExactly(labels, new int[]{-1,-1,-1}));
		System.out.println();
	}*/
}
