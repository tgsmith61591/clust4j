package com.clust4j.kernel;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;

import com.clust4j.utils.VecUtils;

public class KernelTestCases {
	final static Random rand = new Random();
	
	private static double[] randomVector(int length) {
		final double[] a = new double[length];
		for(int i = 0; i < a.length; i++)
			a[i] = rand.nextDouble();
		return a;
	}

	@Test
	public void testSmall() {
		final double[] a = new double[]{0,1};
		final double[] b = new double[]{1,0};
		
		// Perfectly orthogonal
		assertTrue(new LinearKernel().distance(a, b) == 0);
		assertTrue(VecUtils.isOrthogonalTo(a, b));
	}

	@Test
	public void testBigger() {
		final double[] a = randomVector(10);
		final double[] b = randomVector(10);
		System.out.println(new LinearKernel().distance(a, b));
	}
}
