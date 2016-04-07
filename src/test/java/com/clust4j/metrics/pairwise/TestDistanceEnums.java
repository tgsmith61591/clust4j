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
package com.clust4j.metrics.pairwise;

import static org.junit.Assert.*;

import org.junit.Test;
import org.apache.commons.math3.exception.DimensionMismatchException;

import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.MinkowskiDistance;

public class TestDistanceEnums {

	@Test
	public void testEuclidean() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		
		assertTrue(Distance.EUCLIDEAN.getDistance(a, b) == 5.0);
	}
	
	@Test
	public void testManhattan() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		assertTrue(Distance.MANHATTAN.getDistance(a, b) == 7.0);
	}
	
	@Test
	public void testMinkowski1() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		assertTrue(new MinkowskiDistance(2d).getDistance(a, b) == Distance.EUCLIDEAN.getDistance(a, b));
	}
	
	@Test
	public void testMinkowski2() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		assertTrue(new MinkowskiDistance(1d).getDistance(a, b) == Distance.MANHATTAN.getDistance(a, b));
		
	}

	@Test(expected=DimensionMismatchException.class)
	public void testEuclideanFail1() {
		final double[] a = new double[] {0d, 0d, 0d};
		final double[] b = new double[] {3d, 4d};
		Distance.EUCLIDEAN.getDistance(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testManhattanFail1() {
		final double[] a = new double[] {0d, 0d, 0d};
		final double[] b = new double[] {3d, 4d};
		Distance.MANHATTAN.getDistance(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testEuclideanFail2() {
		final double[] a = new double[0];
		final double[] b = new double[0];
		Distance.EUCLIDEAN.getDistance(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testManhattanFail2() {
		final double[] a = new double[]{};
		final double[] b = new double[]{};
		Distance.MANHATTAN.getDistance(a, b);
	}
	
	@Test
	public void testNaN() {
		final double[] a = new double[] {0d, 0d, 0d};
		final double[] b = new double[] {3d, 4d, Double.NaN};
		assertTrue( Double.isNaN(Distance.MANHATTAN.getDistance(a, b)) );
		assertTrue( Double.isNaN(Distance.EUCLIDEAN.getDistance(a, b)) );
	}
	
	@Test
	public void testHaversine1() {
		final double[] a = new double[]{47.6788206, -122.3271205};
		final double[] b = new double[]{47.6788206, -122.5271205};
		
		DistanceMetric haversine = Distance.HAVERSINE.KM;
		assertTrue(haversine.getDistance(a, b) == 14.97319048158622);
		haversine = Distance.HAVERSINE.MI;
		assertTrue(haversine.getDistance(a, b) == 9.304482988008138);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testHaversineFail() {
		final double[] a = new double[]{47.6788206, -122.3271205, 0d};
		final double[] b = new double[]{47.6788206, -122.5271205, 1d};
		Distance.HAVERSINE.KM.getDistance(a, b);
	}
	
	@Test
	public void testCheb() {
		final double[] a = new double[]{0,1,2,3};
		final double[] b = new double[]{10,9,8,7};
		assertTrue(Distance.CHEBYSHEV.getDistance(a, b) == 10);
	}
	
	@Test
	public void testHamming() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.HAMMING.getDistance(a, b) == 0.2);
	}
	
	@Test
	public void testKulsinski() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.KULSINSKI.getDistance(a, b) == 0.33333333333333331);
	}
	
	@Test
	public void testYule() {
		final double[] a = new double[]{1,0,1,0,3,4};
		final double[] b = new double[]{0,3,1,0,0,3};
		assertTrue(Distance.YULE.getDistance(a, b) == 1.0);
	}
	
	@Test
	public void testRogersTanimoto() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.ROGERS_TANIMOTO.getDistance(a, b) == 0.3333333333333333);
	}
	
	@Test
	public void testBrayCurtis() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.BRAY_CURTIS.getDistance(a, b) == 0.076923076923076927);
	}
	
	@Test
	public void testCanberra() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.CANBERRA.getDistance(a, b) == 1.0);
	}
	
	/*
	 * We added smoothing, so this should == 0
	 */
	@Test
	public void testCanberraNaN() {
		final double[] a = new double[]{1,0,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.CANBERRA.getDistance(a, b) == 0);
	}
	
	/*
	 * We added smoothing, so this should == 1
	 */
	@Test
	public void testCanberraNaN2() {
		final double[] a = new double[]{1,0,1,1,3};
		final double[] b = new double[]{1,0,0,1,3};
		assertTrue(Distance.CANBERRA.getDistance(a, b) == 1);
	}
	
	@Test
	public void testDice() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.DICE.getDistance(a, b) == 0.11111111111111111);
	}
	
	@Test
	public void testRussellRao() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.RUSSELL_RAO.getDistance(a, b) == 0.2);
	}
	
	@Test
	public void testSokalSneath() {
		final double[] a = new double[]{1,1,1,1,3};
		final double[] b = new double[]{1,0,1,1,3};
		assertTrue(Distance.SOKAL_SNEATH.getDistance(a, b) == 0.33333333333333331);
	}
	
	@Test
	public void testInfP() {
		assertTrue(Distance.CHEBYSHEV.getP() == Double.POSITIVE_INFINITY);
		assertTrue(Double.isInfinite(Distance.CHEBYSHEV.getP()));
		
		for(Distance d: Distance.values()) {
			if(d.equals(Distance.CHEBYSHEV))
				continue;
			
			assertFalse(d.getP() == Double.POSITIVE_INFINITY);
			assertFalse(Double.isInfinite(d.getP()));
		}
	}
	
	/**
	 * Metrics that would return nan but are smoothed to return zero...
	 * ...but then can also return INF
	 */
	@Test
	public void testNansToZero() {
		double[] a = new double[]{0,0,0,0};
		double[] b = new double[]{0,0,0,0};
		assertTrue(0.0 == Distance.DICE.getDistance(a, b));
		assertTrue(0.0 == Distance.ROGERS_TANIMOTO.getDistance(a, b));
		assertTrue(0.0 == Distance.SOKAL_SNEATH.getDistance(a, b));
		assertTrue(0.0 == Distance.YULE.getDistance(a, b));
	}
	
	@Test
	public void testDice1() {
		double[] a = new double[]{0,1,0,1};
		double[] b = new double[]{1,0,1,0};
		assertTrue(Distance.DICE.getDistance(a, b) == 1);
		
		a = new double[]{1,1,1,1};
		b = new double[]{1,1,1,1};
		assertTrue(Distance.DICE.getDistance(a, b) == 0);
	}
	
	/**
	 * Per Wolfram Alpha
	 */
	@Test
	public void testY1() {
		double[] a = new double[]{1,0,1,1,0};
		double[] b = new double[]{1,1,0,1,1};
		assertTrue(Distance.YULE.getDistance(a, b) == 2.0);
	}
	
	/**
	 * Per Wolfram Alpha
	 */
	@Test
	public void testY2() {
		double[] a = new double[]{1,0,1};
		double[] b = new double[]{1,1,0};
		assertTrue(Distance.YULE.getDistance(a, b) == 2.0);
	}
	
	@Test
	public void testBrayCurtisNaN() {
		/*
		 * this should be nan without smoothing..
		 */
		final double[] a = new double[]{0,0,0,0};
		final double[] b = new double[]{0,0,0,0};
		assertTrue(Distance.BRAY_CURTIS.getDistance(a, b) == 0);
	}
	
	@Test
	public void testMinkowskiShortcut() {
		final double[] a = new double[]{1,0,3,0};
		final double[] b = new double[]{0,2,5,1};
		assertTrue(Distance.MINKOWSKI(1.5).getDistance(a, b) == new MinkowskiDistance(1.5).getDistance(a, b));
	}
}
