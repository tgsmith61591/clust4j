package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.TreeMap;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;

import com.clust4j.algo.HDBSCAN.HList;
import com.clust4j.algo.HDBSCAN.UnifyFind;
import com.clust4j.utils.Inequality;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.NearestNeighborHeapSearch;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.MatUtils.MatSeries;
import com.clust4j.utils.QuadTup;

public class HDBSCANTests {

	
	@Test
	public void testHDBSCANGenericMutualReachability() {
		final double[][] dist_mat = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9}
		};
		
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
		
		double[][] result = HDBSCAN.LinkageTreeUtils.mstLinkageCore(X, m);
		
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
	public void testUnionClass() {
		UnifyFind uni = new UnifyFind(10);
		
		uni.parentArr[0] = 0;
		assertTrue(uni.parent[0] == 0);
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
	public void testGenericRun() {
		final double[][] x = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		HDBSCAN model = new HDBSCAN(new Array2DRowRealMatrix(x), 1).fit();
		System.out.println(Arrays.toString(model.getLabels()));
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
	public void testPartition() {
		final double[][] a = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		int idx_end = a.length, idx_start = 0;
		int n = a[0].length, n_points = idx_end - idx_start;
		int n_mid = n_points / 2;
		int[] indcs = new int[]{0,1,2};
		
		int i_max = NearestNeighborHeapSearch.findNodeSplitDim(a, indcs);
		//HDBSCAN.BinaryTree.partitionNodeIndices(a, indcs, i_max, n_mid, n, n_points);
	}
}
