package com.clust4j.utils;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.utils.NearestNeighborHeapSearch.NodeData;

public class NNHSTests {

	@Test
	public void testKD1() {
		final double[][] a = new double[][]{
			new double[]{0,1,0,2},
			new double[]{0,0,1,2},
			new double[]{5,6,7,4}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(a, false);
		KDTree kd = new KDTree(mat);
		
		QuadTup<double[][], int[], NodeData[], double[][][]> arrays = kd.getArrays();
		
		assertTrue(MatUtils.equalsExactly(arrays.one, a));
		assertTrue(VecUtils.equalsExactly(new int[]{0,1,2}, arrays.two));
		
		TriTup<Integer, Integer, Integer> stats = kd.getTreeStats();
		assertTrue(stats.one == 0);
		assertTrue(stats.two == 0);
		assertTrue(stats.three==0);
		
		NodeData data = arrays.three[0];
		assertTrue(data.idx_start == 0);
		assertTrue(data.idx_end == 3);
		assertTrue(data.is_leaf);
		assertTrue(data.radius == 1);
	}

}
