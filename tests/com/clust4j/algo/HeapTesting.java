package com.clust4j.algo;

import static org.junit.Assert.*;

import org.junit.Test;

public class HeapTesting {
	private static void testMut(double[] a) {
		a = new double[]{1,2};
	}
	
	@Test
	public void testMutability() {
		double[] a = new double[]{1,2,3};
		testMut(a);
		assertTrue(a.length == 3);
	}

}
