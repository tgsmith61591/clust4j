package com.clust4j.algo;

import static org.junit.Assert.*;

import org.junit.Test;

public class BoruvkaTests {

	@Test
	public void testBallTreeMinDistDual() {
		double[][] d = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9}
		};
		
		double rad1 = 0.5, rad2 = 0.75;
		assertTrue(BoruvkaAlgorithm.ballTreeMinDistDual(rad1, rad2, 1, 2, d) == 4.75);
	}

}
