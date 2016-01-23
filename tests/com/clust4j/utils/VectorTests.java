package com.clust4j.utils;

import static org.junit.Assert.*;

import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.utils.VecUtils.VecSeries;

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
	public void testMutability2() {
		double[] i = new double[]{1,2,3,4};
		double[] j = i;
		
		i = new double[]{4,3,2,1};
		assertTrue(j[0] != i[0]);
	}
	
	@Test
	public void testExtMathNorm() {
		double[] i = new double[]{0,1.0};
		double[] j = new double[]{0,0.9};
		assertTrue( VecUtils.l2Norm(VecUtils.subtract(i, j)) == 0.09999999999999998 );
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
		
		
		assertTrue( Precision.equals( VecUtils.lpNorm(neg, 2.0000000000000001), VecUtils.l2Norm(neg), Precision.EPSILON) );
	}
	
	@Test
	public void testCosSim() {
		final double[] a = new double[]{1,1,1,1};
		final double[] b = new double[]{1,2,3,4};
		final double cosSim1 = VecUtils.cosSim(a, b);
		
		assertTrue(Precision.equals(cosSim1, 0.9128709291752769));
	}
	
	@Test
	public void testHilbertSpace() {
		final double[] a = new double[]{0, 5};
		final double[] b = new double[]{0, 3};
		
		final double inner = VecUtils.innerProduct(a, b);
		final double length = VecUtils.l2Norm(a) * VecUtils.l2Norm(b) * VecUtils.cosSim(a, b);
		assertTrue(inner == length);
	}
	
	@Test
	public void testPMinMax() {
		final double[] a = new double[]{0, 5};
		final double[] b = new double[]{3, 0};
		assertTrue(VecUtils.equalsExactly(VecUtils.pmax(a, b), new double[]{3,5}));
		assertTrue(VecUtils.equalsExactly(VecUtils.pmin(a, b), new double[]{0,0}));
	}
	
	@Test
	public void testProd() {
		final double[] a = new double[]{1, 2, 3};
		assertTrue(VecUtils.prod(a) == 6);
	}
	
	@Test
	public void testMedian() {
		final double[] a = new double[]{2, 1, 3};
		assertTrue(VecUtils.median(a) == 2);
		
		final double[] b = new double[]{2, 1, 3, 5, 4, 9};
		assertTrue(VecUtils.median(b) == 3.5);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMedianExcept() {
		final double[] a = new double[]{};
		VecUtils.median(a);
	}
	
	@Test
	public void testArgs() {
		assertTrue(-5 > GlobalState.Mathematics.SIGNED_MIN);
		
		final double[] a = new double[]{0, 5};
		assertTrue(VecUtils.argMax(a) == 1);
		assertTrue(VecUtils.argMin(a) == 0);
		
		final double[] b = new double[]{0,0};
		assertTrue(VecUtils.argMax(b) == 0);
		assertTrue(VecUtils.argMin(b) == 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testArgs2() {
		final double[] a = new double[]{};
		VecUtils.argMax(a);
	}
	
	@Test
	public void testArange() {
		assertTrue(VecUtils.equalsExactly(VecUtils.arange(10),new int[]{0,1,2,3,4,5,6,7,8,9}));
		assertTrue(VecUtils.equalsExactly(VecUtils.arange(10,0),new int[]{10,9,8,7,6,5,4,3,2,1}));
		assertTrue(VecUtils.equalsExactly(VecUtils.arange(10,0,-2),new int[]{10,8,6,4,2}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testArange2() {
		VecUtils.arange(10,0,-3);
	}
	
	@Test
	public void testFloor() {
		final double[] d = new double[]{-1,0,2};
		final double[] b = VecUtils.floor(d, 0, 1);
		assertTrue(VecUtils.equalsExactly(b, new double[]{1,0,2}));
	}
	
	@Test
	public void testNanOps() {
		final double[] d = new double[]{-1,0,2,Double.NaN};
		assertTrue(VecUtils.containsNaN(d));
		assertTrue(VecUtils.nanCount(d) == 1);
		assertTrue(VecUtils.nanMean(d) == 1d/3d);
		assertTrue(VecUtils.nanSum(d) == 1);
	}
	
	@Test
	public void testNanMaxMinStdVar() {
		final double[] a = new double[]{Double.NaN};
		final double[] b = new double[]{Double.NaN, 1, -9, Double.NaN, 100};
		
		assertTrue( Double.isNaN(VecUtils.nanMax(a)) );
		assertTrue( Double.isNaN(VecUtils.nanMin(a)) );
		assertTrue( VecUtils.nanMax(b) == 100 );
		assertTrue( VecUtils.nanMin(b) == -9 );
		assertTrue( Double.isNaN(VecUtils.nanStdDev(a)) );
		assertFalse(Double.isNaN(VecUtils.nanStdDev(b)) );
		assertTrue( Double.isNaN(VecUtils.nanVar(a)) );
		assertFalse(Double.isNaN(VecUtils.nanVar(b)) );
		assertTrue( Double.isNaN(VecUtils.nanMean(a)) );
	}
	
	@Test
	public void testPartition() {
		final double[] a = new double[]{3, 4, 5, 1};
		final double[] b = new double[]{3, 4, 1, 5};
		final double[] c = VecUtils.copy(a);
		assertTrue( VecUtils.equalsExactly(VecUtils.partition(a, 2), b) );
		assertTrue( VecUtils.equalsExactly(a, c) );
	}
	
	@Test
	public void testWhere() {
		final double[] a = new double[]{1,7,3};
		final VecSeries series = new VecSeries(a, Inequality.GT, 5);
		
		final double[] b = VecUtils.rep(1, 3);
		final double[] c = VecUtils.rep(0, 3);
		final double[] res = new double[]{0,1,0};
		
		assertTrue(VecUtils.equalsExactly(res, VecUtils.where(series, b, c)));
	}
	
	@Test
	public void testCat() {
		final int[] a = new int[]{0,0,0};
		final int[] b = new int[]{1,1,1,1,1};
		final int[] c = new int[]{0,0,0,1,1,1,1,1};
		assertTrue(VecUtils.equalsExactly(c, VecUtils.cat(a, b)));
	}
	
	@Test
	public void testArgSort() {
		double[] a = new double[]{5,1,3,4};
		int[] exp = new int[]{1,2,3,0};
		assertTrue(VecUtils.equalsExactly(exp, VecUtils.argSort(a)));
	}
}
