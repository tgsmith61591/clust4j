package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

public class VectorTests {

	@Test
	public void test() {
		final double[] a = new double[]{0, 1, 2, 3, 4};
		double sum = 0;
		double mean = 0;
		
		assertTrue((sum = VecUtils.sum(a)) == 10);
		assertTrue((mean = VecUtils.mean(a)) == 2);
		assertTrue(VecUtils.mean(a, sum) == 2);
		assertTrue(VecUtils.stdDev(a,mean) == VecUtils.stdDev(a));
	}
}
