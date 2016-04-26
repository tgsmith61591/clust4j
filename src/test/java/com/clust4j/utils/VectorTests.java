/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.utils;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.utils.Series.Inequality;
import com.clust4j.utils.VecUtils.DoubleSeries;
import com.clust4j.utils.VecUtils.VecSeries;

public class VectorTests {
	final static double[] empty = new double[]{};

	@Test
	public void test() {
		final double[] a = new double[]{0, 1, 2, 3, 4};
		double sum = 0;
		double mean = 0;
		
		assertTrue((sum = VecUtils.sum(a)) == 10);
		assertTrue((mean = VecUtils.mean(a)) == 2);
		assertTrue(VecUtils.mean(a, sum) == 2);
		assertTrue(VecUtils.stdDev(a,mean) == VecUtils.stdDev(a));
		assertTrue(Double.isNaN(VecUtils.mean(empty)));
	}
	
	@Test
	public void testMutability() {
		final int[] i = new int[]{1,2,3,4};
		final int[] j = VecUtils.copy(i);
		
		i[0] = 0;
		assertTrue(j[0] != i[0]);
		
		final String[] s= new String[]{"a","b"};
		String[] b = VecUtils.copy(s);
		b[0] = "b";
		assertFalse(s[0].equals("b"));
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
		final double[] a = empty;
		VecUtils.median(a);
	}
	
	@Test
	public void testArgs() {
		assertTrue(-5 > GlobalState.Mathematics.SIGNED_MIN);
		
		final double[] ad = new double[]{0, 5};
		assertTrue(VecUtils.argMax(ad) == 1);
		assertTrue(VecUtils.argMin(ad) == 0);
		
		final double[] bd = new double[]{0,0};
		assertTrue(VecUtils.argMax(bd) == 0);
		assertTrue(VecUtils.argMin(bd) == 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testArgs2() {
		final double[] a = empty;
		VecUtils.argMax(a);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testArangeInt() {
		assertTrue(VecUtils.equalsExactly(VecUtils.arange(10),new int[]{0,1,2,3,4,5,6,7,8,9}));
		assertTrue(VecUtils.equalsExactly(VecUtils.arange(10,0),new int[]{10,9,8,7,6,5,4,3,2,1}));
		assertTrue(VecUtils.equalsExactly(VecUtils.arange(10,0,-2),new int[]{10,8,6,4,2}));
		
		System.out.println(Arrays.toString(VecUtils.arange(10, 0, -3)));
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
		final DoubleSeries series = new DoubleSeries(a, Inequality.GREATER_THAN, 5);
		
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
		int[] b = new int[]{5,1,3,4};
		int[] exp = new int[]{1,2,3,0};
		assertTrue(VecUtils.equalsExactly(exp, VecUtils.argSort(a)));
		assertTrue(VecUtils.equalsExactly(exp, VecUtils.argSort(b)));
		
		// Test empty
		boolean p = false;
		try {p = false; VecUtils.argSort(empty);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.argSort(new int[]{ }); } catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	
		// test tie corner case
		b = new int[]{5,1,3,4,4,9};
		assertTrue(VecUtils.equalsExactly(VecUtils.argSort(b), new int[]{1,2,3,4,0,5}));
	}
	
	@Test
	public void testSlice() {
		final double[] a = new double[]{0,1,2,3};
		final double[] b = new double[]{1,2};
		assertTrue(VecUtils.equalsExactly(b, VecUtils.slice(a, 1, 3)));
	}
	
	@Test
	public void testReorder() {
		double[] a = new double[]{5,1,3,4};
		double[] ordered = VecUtils.reorder(a, new int[]{0,1,0,1});
		assertTrue(VecUtils.equalsExactly(ordered, new double[]{5,1,5,1}));
	}
	
	@Test
	public void testReverse() {
		final int[] a = new int[]{0,1,2,3};
		final int[] b = new int[]{3,2,1,0};
		assertTrue(VecUtils.equalsExactly(b, VecUtils.reverseSeries(a)));
	}
	
	@Test
	public void testOpsWithEmpty() {
		final double[] a = empty;
		final double[] b = empty;
		
		VecUtils.add(a, b);

		VecUtils.multiply(a, b);
		
		VecUtils.subtract(a, b);
		
		assertTrue(VecUtils.equalsExactly(a, b));
	}
	
	@Test
	public void testArgSortWithTie() {
		double[] a = new double[]{2,1,1};
		assertTrue(VecUtils.equalsExactly(new int[]{1,2,0}, VecUtils.argSort(a)));
		
		a = VecUtils.rep(0.8, 10);
		assertTrue(VecUtils.equalsExactly(new int[]{0,1,2,3,4,5,6,7,8,9}, VecUtils.argSort(a)));
	}
	
	@Test
	public void testAbs() {
		// test empty
		assertTrue(VecUtils.equalsExactly(empty, VecUtils.abs(empty)));

		// full tests
		assertTrue(VecUtils.equalsExactly(new double[]{1,2,3}, VecUtils.abs(new double[]{-1,-2,3})));
	}
	
	@Test
	public void testAdd() {
		// test empty
		assertTrue(VecUtils.equalsExactly(empty, VecUtils.add(empty, empty)));

		// full tests
		final double[] add = new double[]{1,2,3};
		final double[] tgt = new double[]{2,4,6};
		assertTrue(VecUtils.equalsExactly(tgt, VecUtils.add(add, add)));
		
		// Test DMEs
		boolean p = false;
		final double[] off = new double[]{1,2};
		try {p = false; VecUtils.add(add, off);}catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testArange() {
		boolean p = false;
		
		// test default
		assertTrue(VecUtils.equalsExactly(new int[]{0,1,2,3,4}, VecUtils.arange(5)));
		try {p = false; VecUtils.arange(0);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.arange(Integer.MAX_VALUE);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		
		// test second
		assertTrue(VecUtils.equalsExactly(new int[]{2,3,4}, VecUtils.arange(2,5)));
		assertTrue(VecUtils.equalsExactly(new int[]{5,4,3}, VecUtils.arange(5,2)));
		try {p = false; VecUtils.arange(2,2);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.arange(0,Integer.MAX_VALUE);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		
		// test third
		assertTrue(VecUtils.equalsExactly(new int[]{2,3,4}, VecUtils.arange(2,5,1)));
		assertTrue(VecUtils.equalsExactly(new int[]{5,4,3}, VecUtils.arange(5,2,-1)));
		assertTrue(VecUtils.equalsExactly(new int[]{0,2,4,6,8}, VecUtils.arange(0,10,2)));
		try {p = false; VecUtils.arange(2,2,0);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.arange(2,2,2);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.arange(0,Integer.MAX_VALUE,1);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.arange(0,5,3);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.arange(0,5,-1);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.arange(5,0, 1);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testArgMinMax() {
		final double[] v = new double[]{-1,4,2,5,-1,5};
		assertTrue(VecUtils.argMax(v) == 3);
		assertTrue(VecUtils.argMin(v) == 0);
		
		boolean p = false;
		try {p = false; VecUtils.argMax(empty);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.argMin(empty);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testAsDouble() {
		// Test populated and empty
		assertTrue(VecUtils.equalsExactly(empty, VecUtils.asDouble(new int[]{})));
		assertTrue(VecUtils.equalsExactly(new double[]{1.0,2.0,3.0}, VecUtils.asDouble(new int[]{1,2,3})));
	}
	
	@Test
	public void testCat2() {
		assertTrue(VecUtils.equalsExactly(new double[]{1,2,3,4}, VecUtils.cat(new double[]{1,2}, new double[]{3,4})));
		assertTrue(VecUtils.equalsExactly(new int[]{1,2,3,4}, VecUtils.cat(new int[]{1,2}, new int[]{3,4})));
		
		// test empties
		assertTrue(VecUtils.equalsExactly(new double[]{3,4}, VecUtils.cat(empty, new double[]{3,4})));
		assertTrue(VecUtils.equalsExactly(new double[]{1,2}, VecUtils.cat(new double[]{1,2}, empty)));
		assertTrue(VecUtils.equalsExactly(empty, 	 VecUtils.cat(empty, empty)));
		
		// test empties
		assertTrue(VecUtils.equalsExactly(new int[]{3,4}, VecUtils.cat(new int[]{}, new int[]{3,4})));
		assertTrue(VecUtils.equalsExactly(new int[]{1,2}, VecUtils.cat(new int[]{1,2}, new int[]{})));
		assertTrue(VecUtils.equalsExactly(new int[]{}, 	  VecUtils.cat(new int[]{}, new int[]{})));
	}
	
	@Test
	public void testCenter() {
		double[] v = new double[]{1,2,3};
		assertTrue(VecUtils.equalsExactly(new double[]{-1,0,1}, VecUtils.center(v)));
		assertTrue(VecUtils.equalsExactly(new double[]{-2,-1,0}, VecUtils.center(v, 3)));
		
		// Test empty
		boolean p = false;
		try {p = false; VecUtils.center(empty);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.center(empty, 2);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testCompleteCases() {
		assertTrue(VecUtils.equalsExactly(empty, VecUtils.completeCases(empty)));
		assertTrue(VecUtils.equalsExactly(empty, VecUtils.completeCases(new double[]{Double.NaN, Double.NaN})));
		assertTrue(VecUtils.equalsExactly(new double[]{0,0,1}, VecUtils.completeCases(new double[]{0,0,1})));
	}
	
	@Test
	public void testContainsNaN() {
		final double[] none = new double[]{1,2,3,4};
		final double[] some = new double[]{1, Double.NaN, 2};
		
		// default
		assertFalse(VecUtils.containsNaN(empty));
		assertFalse(VecUtils.containsNaN(none));
		assertTrue(VecUtils.containsNaN(some));
	}
	
	@Test
	public void testShallowClone() {
		ArrayList<double[]> og = new ArrayList<>();
		og.add(new double[]{0});
		
		ArrayList<double[]> co = VecUtils.copy(og);
		co.get(0)[0] = 1;
		assertTrue(og.get(0)[0] == 1);
	}
	
	@Test
	public void testDeepClone() {
		ArrayList<Double> og = new ArrayList<>();
		og.add(0.0);
		
		ArrayList<Double> co = VecUtils.copy(og);
		co.set(0, 1.0);
		assertFalse(og.get(0) == 1.0);
	}
	
	@Test
	public void testCosSim2() {
		boolean p = false;
		try {p = false; VecUtils.cosSim(empty,empty);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.cosSim(new double[]{1}, new double[]{1,2});}catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testDivide() {
		// test empty first
		assertTrue(VecUtils.equalsExactly(empty, VecUtils.divide(empty, empty)));
		assertTrue(VecUtils.equalsExactly(new double[]{0.5,0.5,1.0}, VecUtils.divide(new double[]{1.0,1.0,1.0}, new double[]{2.0,2.0,1.0})));
		
		// dme
		boolean p = false;
		try {p = false; VecUtils.divide(new double[]{1}, new double[]{1,2});}catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testExp() {
		double[] a = new double[]{1,2,3};
		double[] e = new double[]{2.718281828459045, 7.38905609893065, 20.085536923187668};
		assertTrue(VecUtils.equalsExactly(VecUtils.exp(a), e));
		assertTrue(VecUtils.equalsExactly(VecUtils.exp(empty), empty));
	}
	
	@Test
	public void testInnerProduct() {
		// test empties
		assertTrue(0.0 == VecUtils.innerProduct(empty, empty));
		
		// test populated
		double[] a = new double[]{1,2};
		double[] b = new double[]{1,2,3};
		final double res = 14;
		assertTrue(res == VecUtils.innerProduct(b,b));
		
		// test ortho
		assertFalse(VecUtils.isOrthogonalTo(b, b));
		
		// test DMEs
		boolean p = false;
		try {p = false; VecUtils.innerProduct(a,b);}catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.isOrthogonalTo(a,b);}catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testIqr() {
		assertTrue(0.0 == VecUtils.iqr(new double[]{1.0}));
		boolean p = false;
		try {p = false; VecUtils.iqr(empty);}catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testNorms() {
		assertTrue(VecUtils.l1Norm(new double[]{1,2,-3,4}) == 10.0);
		assertTrue(VecUtils.l1Norm(empty) == 0.0);
		assertTrue(VecUtils.l2Norm(new double[]{1,2,-3,4}) == FastMath.sqrt(30.0));
		assertTrue(VecUtils.l2Norm(empty) == 0.0);
	}
	
	@Test
	public void testVecSeries() {
		double[] va = new double[]{1,2,3};
		double[] vb = new double[]{3,2,1};
		DoubleSeries a = new DoubleSeries(va, Inequality.EQUAL_TO, vb);
		assertTrue(VecUtils.equalsExactly(a.get(), a.getRef()));
		assertTrue(VecUtils.equalsExactly(a.get(), new boolean[]{false, true, false}));
		
		a = new DoubleSeries(va, Inequality.GREATER_THAN, vb);
		assertTrue(VecUtils.equalsExactly(a.get(), new boolean[]{false, false, true}));
		
		boolean p = false;
		try {p = false; new DoubleSeries(va, Inequality.EQUAL_TO, new double[]{1,1}); }catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
		try {p = false; new DoubleSeries(va, Inequality.EQUAL_TO, new double[]{}); }catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
		try {p = false; new DoubleSeries(new double[]{}, Inequality.EQUAL_TO, vb); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void checkDims() {
		VecUtils.checkDims(new boolean[1]); // should pass
		VecUtils.checkDimsPermitEmpty(new boolean[]{});// should pass
		
		boolean p = false;
		boolean[] a = new boolean[]{true};
		boolean[] b = new boolean[]{true, false};
		boolean[] c = new boolean[]{};
		
		VecUtils.checkDims(a,a); // should pass
		VecUtils.checkDimsPermitEmpty(c, c); // should pass
		VecUtils.checkDimsPermitEmpty(a, a); // should pass
		try {p = false; VecUtils.checkDims(a,b); }catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.checkDims(c,c); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	
		int[] ia = new int[]{1};
		int[] ib = new int[]{1,2};
		int[] ic = new int[]{};
		VecUtils.checkDims(ia, ia); // should pass
		VecUtils.checkDimsPermitEmpty(ic, ic); // should pass
		try {p = false; VecUtils.checkDims(ia,ib); }catch(DimensionMismatchException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.checkDims(ic,ic); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testEquals() {
		int[] ia = new int[]{1,2,5};
		int[] ib = new int[]{1,2,3};
		assertFalse(VecUtils.equalsExactly(ia, ib));
		
		boolean[] ba = new boolean[]{true, false};
		boolean[] bb = new boolean[]{true, true};
		assertFalse(VecUtils.equalsExactly(ba, bb));
		
		double[] da = new double[]{1,2};
		double[] db = new double[]{2,2};
		assertFalse(VecUtils.equalsExactly(da, db));
	}
	
	@Test
	public void testLog() {
		final double[] a = new double[]{1, 2, 3};
		final double[] b = new double[]{0, 0.69314718055994529, 1.0986122886681098};
		final double[] c = new double[]{};
		assertTrue(VecUtils.equalsExactly(VecUtils.log(a), b));
		assertTrue(VecUtils.equalsExactly(VecUtils.log(c), c));
	}
	
	@Test
	public void testLpNorm() {
		double p = 3.0;
		double[] a = new double[]{1,2,3};
		assertTrue(VecUtils.lpNorm(a, p) == 3.3019272488946263);
		assertTrue(VecUtils.l2Norm(a) == VecUtils.magnitude(a));
	}
	
	@Test
	public void testMaxMinMedian() {
		double[] a = new double[]{1,2,3};
		double[] b = new double[]{1};
		assertTrue(VecUtils.max(a) == 3);
		assertTrue(VecUtils.min(a) == 1);
		assertTrue(VecUtils.median(a) == 2);
		assertTrue(VecUtils.median(b) == 1);
		
		boolean p = false;
		try {p = false; VecUtils.max(new double[]{}); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.min(new double[]{}); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.median(new double[]{}); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.nanMax(new double[]{}); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.nanMin(new double[]{}); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
		try {p = false; VecUtils.nanMedian(new double[]{}); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	
		double[] nan = new double[]{Double.NaN};
		assertTrue(Double.isNaN(VecUtils.nanMax(nan)));
		assertTrue(Double.isNaN(VecUtils.nanMin(nan)));
		assertTrue(VecUtils.nanMedian(new double[]{Double.NaN, 4}) == 4);
		try {p = false; VecUtils.nanMedian(nan); }catch(IllegalArgumentException e){p = true;}finally{if(!p)fail();}
	}
	
	@Test
	public void testNaNCountEmpty() {
		double[] a = new double[]{};
		assertTrue(VecUtils.nanCount(a) == 0);
	}
	
	@Test
	public void testPermutation() {
		final int[] a = new int[]{1,2,3,4,5,6};
		assertTrue(VecUtils.permutation(a).length == 6);
		assertTrue(VecUtils.permutation(a,new Random()).length == 6);
		assertTrue(VecUtils.equalsExactly(VecUtils.permutation(new int[]{}), new int[]{}));
	}
	
	@Test
	public void testNullOK() {
		int[] i = null;
		boolean[] b = null;
		String[] s = null;
		double[] d = null;
		
		assertNull(VecUtils.copy(i));
		assertNull(VecUtils.copy(b));
		assertNull(VecUtils.copy(s));
		assertNull(VecUtils.copy(d));
		
		assertTrue(VecUtils.equalsExactly(i, i));
		assertTrue(VecUtils.equalsExactly(b, b));
		assertTrue(VecUtils.equalsExactly(s, s));
		assertTrue(VecUtils.equalsExactly(d, d));
	}
	
	@Test
	public void testInequalityByNullOrLength() {
		String[] a = new String[]{};
		String[] b = new String[]{"a"};
		String[] c = null;
		String[] d = new String[]{"b"};
		
		assertTrue(VecUtils.equalsExactly(a, a));
		assertFalse(VecUtils.equalsExactly(a,b));
		assertTrue(VecUtils.equalsExactly(c, c));
		assertFalse(VecUtils.equalsExactly(c,b));
		assertFalse(VecUtils.equalsExactly(d,b)); // coverage love
		
		boolean[] ba = new boolean[]{true, false};
		boolean[] bb = new boolean[]{false};
		boolean[] bc = null;
		boolean[] bd = new boolean[]{true};
		
		assertTrue(VecUtils.equalsExactly(ba, ba));
		assertFalse(VecUtils.equalsExactly(ba,bb));
		assertTrue(VecUtils.equalsExactly(bc, bc));
		assertFalse(VecUtils.equalsExactly(bc,bb));
		assertFalse(VecUtils.equalsExactly(bd,bb)); // coverage love
		
		int[] ia = new int[]{};
		int[] ib = new int[]{0};
		int[] ic = null;
		
		assertTrue(VecUtils.equalsExactly(ia, ia));
		assertFalse(VecUtils.equalsExactly(ia,ib));
		assertTrue(VecUtils.equalsExactly(ic, ic));
		assertFalse(VecUtils.equalsExactly(ic,ib));
		
		double[] da = new double[]{};
		double[] db = new double[]{0.0};
		double[] dc = null;
		
		assertTrue(VecUtils.equalsExactly(da, da));
		assertFalse(VecUtils.equalsExactly(da,db));
		assertTrue(VecUtils.equalsExactly(dc, dc));
		assertFalse(VecUtils.equalsExactly(dc,db));
	}
	
	@Test
	public void testSomeMoreVecSeries() {
		final double[] d = {1.0,0.0,1.0};
		VecSeries v =new DoubleSeries(d, Inequality.EQUAL_TO, 1.0);
		assertFalse(v.all());
		
		v = new DoubleSeries(d, Inequality.EQUAL_TO, 2.0);
		assertFalse(v.any());
		
		/*
		 * Dim mismatch test
		 */
		boolean a = false;
		try {
			v = new VecUtils.IntSeries(new int[]{1,2,3}, Inequality.EQUAL_TO, new int[]{1,2});
		} catch(DimensionMismatchException dim) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void someVarianceTests() {
		double[] d = new double[]{1,2,3,4,5};
		double[] e = new double[]{1,2,3,4,Double.NaN};
		final double meanD = VecUtils.mean(d);
		assertTrue(VecUtils.stdDev(d,meanD) == VecUtils.nanStdDev(d,meanD));
		assertTrue(VecUtils.var(d) == VecUtils.nanVar(d));
		assertTrue(VecUtils.var(d,meanD) == VecUtils.nanVar(d,meanD));
		assertFalse(VecUtils.var(d) == VecUtils.nanVar(e));
	}
	
	@Test
	public void normalization() {
		// just for coverage...
		double[] d = new double[]{1,2,3,4,5};
		VecUtils.normalize(d);
	}
	
	@Test
	public void testOOB() {
		double[] d = new double[]{1,2,3,4,5};
		boolean a = false;
		
		try {
			VecUtils.partition(d, 100);
		} catch(IllegalArgumentException i) {
			a = true;	
		} finally {
			assertTrue(a);
		}
	}
	
	@Test
	public void coverage() {
		VecUtils.randomGaussian(5, 1.0);
		
		/*
		 * IAE on gaussian
		 */
		boolean a = false;
		try {
			VecUtils.randomGaussian(0);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * IAE on rep
		 */
		a = false;
		try {
			VecUtils.rep(0, -1);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * IAE on rep
		 */
		a = false;
		try {
			VecUtils.repInt(0, -1);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * IAE on rep
		 */
		a = false;
		try {
			VecUtils.repBool(false, -1);
		} catch(IllegalArgumentException i) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * reorder
		 */
		int[] i = new int[]{1,2,3};
		double[] d = new double[]{1,2,3};
		assertTrue(VecUtils.equalsExactly(VecUtils.reorder(i, new int[]{2,0,1}), new int[]{3,1,2}));
		assertTrue(VecUtils.equalsExactly(VecUtils.reverseSeries(d), new double[]{3,2,1}));
	
		/*
		 * AIOOB on slice
		 */
		a = false;
		try {
			VecUtils.slice(i, 0, 6);
		} catch(ArrayIndexOutOfBoundsException ai) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			VecUtils.slice(i, -1, 2);
		} catch(ArrayIndexOutOfBoundsException ai) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			VecUtils.slice(i, 3, 1);
		} catch(IllegalArgumentException ai) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		
		
		
		
		a = false;
		try {
			VecUtils.slice(d, 0, 6);
		} catch(ArrayIndexOutOfBoundsException ai) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			VecUtils.slice(d, -1, 2);
		} catch(ArrayIndexOutOfBoundsException ai) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		a = false;
		try {
			VecUtils.slice(d, 3, 1);
		} catch(IllegalArgumentException ai) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		assertTrue(VecUtils.equalsExactly(VecUtils.sortAsc(new double[]{}), new double[]{}));
		assertTrue(VecUtils.equalsExactly(VecUtils.sortAsc(new int[]{}), new int[]{}));
		assertTrue(VecUtils.equalsExactly(VecUtils.sortAsc(new int[]{3,2,1}), new int[]{1,2,3}));
		assertTrue(VecUtils.equalsExactly(VecUtils.sqrt(new double[]{9,16,25}), new double[]{3,4,5}));
		assertTrue(VecUtils.unique(new double[]{1,2,1}).size() == 2);
		
		assertTrue(VecUtils.vstack(new int[]{1,2}, new int[]{1,2}).length == 2);
	}
	
	@Test
	public void testCumSum() {
		double[] a = new double[]{};
		assertTrue(VecUtils.equalsExactly(a, VecUtils.cumsum(a)));
		
		a = new double[]{131,  15, 118, 100};
		assertTrue(VecUtils.equalsExactly(VecUtils.cumsum(a), new double[]{131, 146, 264, 364}));
	}
}
