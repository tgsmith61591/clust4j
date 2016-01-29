package com.clust4j.utils;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.utils.NearestNeighborHeapSearch.PartialKernelDensity;
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

	
	@Test
	public void testKernelDensities() {
		// Test where dist > h first
		double dist = 5.0, h = 1.3;
		assertTrue(PartialKernelDensity.LOG_GAUSSIAN.getDensity(dist, h) == -7.396449704142011);
		assertTrue(PartialKernelDensity.LOG_TOPHAT.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_EPANECHNIKOV.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_EXPONENTIAL.getDensity(dist, h) == -3.846153846153846);
		assertTrue(PartialKernelDensity.LOG_LINEAR.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_COSINE.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		
		// Test where dist < h second
		dist = 1.3; 
		h = 5.0;
		
		assertTrue(PartialKernelDensity.LOG_GAUSSIAN.getDensity(dist, h) == -0.033800000000000004);
		assertTrue(PartialKernelDensity.LOG_TOPHAT.getDensity(dist, h) == 0.0);
		assertTrue(PartialKernelDensity.LOG_EPANECHNIKOV.getDensity(dist, h) == -0.06999337182053497);
		assertTrue(PartialKernelDensity.LOG_EXPONENTIAL.getDensity(dist, h) == -0.26);
		assertTrue(PartialKernelDensity.LOG_LINEAR.getDensity(dist, h) == -0.3011050927839216);
		assertTrue(PartialKernelDensity.LOG_COSINE.getDensity(dist, h) == -0.08582521637384073);
	}
}
