package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.HierarchicalAgglomerative.Linkage;

public class HierTests {
	private static Array2DRowRealMatrix matrix = ClustTests.getRandom(250, 10);
	
	@Test
	public void testRandom() {
		HierarchicalAgglomerative hac = 
			new HierarchicalAgglomerative(matrix,
				new HierarchicalAgglomerative
					.HierarchicalPlanner().setVerbose(true));
		hac.fit();
	}

	@Test
	public void testMore() {
		final double[][] data = new double[][] {
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		HierarchicalAgglomerative hac = 
			new HierarchicalAgglomerative(mat,
				new HierarchicalAgglomerative
					.HierarchicalPlanner()
						.setLinkage(Linkage.AVERAGE)
						.setVerbose(true));
		hac.fit();
		
		int[] labels = hac.getLabels();
		System.out.println(Arrays.toString(labels));
		assertTrue(labels[0] == labels[2]);
	}
}
