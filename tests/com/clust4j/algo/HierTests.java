package com.clust4j.algo;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

public class HierTests {
	private static Array2DRowRealMatrix matrix = ClustTests.getRandom(250, 10);

	@Test
	public void test1() {
		HierarchicalAgglomerativeClusterer hac = 
			new HierarchicalAgglomerativeClusterer(matrix,
				new HierarchicalAgglomerativeClusterer
					.HierarchicalPlanner().setVerbose(true));
		hac.fit();
		
		assertTrue(true);
	}

}
