package com.clust4j.utils;

import static org.junit.Assert.*;

import org.apache.commons.math3.util.Precision;
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
	
	@Test
	public void testMutability() {
		final int[] i = new int[]{1,2,3,4};
		final int[] j = VecUtils.copy(i);
		
		i[0] = 0;
		assertTrue(j[0] != i[0]);
	}
	
	@Test
	public void testVecOps() {
		// Inner
		final double[] a = new double[]{1,1,1,1};
		final double[] b = new double[]{1,2,3,4};
		assertTrue(VecUtils.innerProduct(a, b) == 10d);
		
		// Scalar mult
		final double[] c = new double[]{2,4,6,8};
		assertTrue(VecUtils.equalsExactly(c, VecUtils.scalarMultiply(b, 2)));
		
		// Equals exactly and with tolerance
		assertTrue(VecUtils.equalsWithTolerance(c, VecUtils.scalarMultiply(b, 2)));
		assertTrue(VecUtils.equalsWithTolerance(c, VecUtils.scalarMultiply(b, 2), 0));
		
		// Mult
		final double[] d = new double[]{2,8,18,32};
		assertTrue(VecUtils.equalsExactly(d, VecUtils.multiply(b, c)));
		
		
		
		// Scalar add
		final double[] scadd = new double[]{3,4,5,6};
		assertTrue(VecUtils.equalsExactly(scadd, VecUtils.scalarAdd(b, 2)));
				
		// add
		final double[] add = new double[]{2,3,4,5};
		assertTrue(VecUtils.equalsExactly(add, VecUtils.add(a, b)));
		
		
		
		// Scalar div
		final double[] scdiv = new double[]{2,2,2,2};
		assertTrue(VecUtils.equalsExactly(b, VecUtils.scalarDivide(c, 2)));
				
		// div
		assertTrue(VecUtils.equalsExactly(scdiv, VecUtils.divide(c, b)));
		
		
		// Scalar sub
		final double[] scsub = new double[]{0,2,4,6};
		assertTrue(VecUtils.equalsExactly(scsub, VecUtils.scalarSubtract(c, 2)));
				
		// sub
		assertTrue(VecUtils.equalsExactly(scsub, VecUtils.subtract(c, scdiv)));
		
		
		
		// Outer prod
		final double[] by = new double[]{2,3,4};
		final double[][] ab = VecUtils.outerProduct(b, by);
		
		assertTrue(ab.length == 4);
		assertTrue( VecUtils.equalsExactly(ab[0], by) );
		assertTrue( VecUtils.equalsExactly(ab[1], new double[]{4,6,8}) );
		assertTrue( VecUtils.equalsExactly(ab[2], new double[]{6,9,12}) );
		assertTrue( VecUtils.equalsExactly(ab[3], new double[]{8,12,16}) );
		
		// Abs
		final double[] neg = new double[]{-2,3,-4};
		assertTrue( VecUtils.equalsExactly(by, VecUtils.abs(neg)) );
		assertTrue( VecUtils.l1Norm(neg) == 9 );
		assertTrue( VecUtils.lpNorm(neg, 1) == 9 );
		assertTrue( VecUtils.lpNorm(neg, 2) == VecUtils.l2Norm(neg) );
	}
	
	@Test
	public void testCosSim() {
		final double[] a = new double[]{1,1,1,1};
		final double[] b = new double[]{1,2,3,4};
		final double cosSim1 = VecUtils.cosSim(a, b);
		final double cosSim2 = new CosineSimilarity().distance(a, b);
		
		assertTrue(Precision.equals(cosSim1, cosSim2, Precision.EPSILON));
		assertTrue(Precision.equals(cosSim1, 0.9128709291752769));
	}
}
