package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;
import org.apache.commons.math3.exception.DimensionMismatchException;

import com.clust4j.utils.Distance;
import com.clust4j.utils.HaversineDistance.DistanceUnit;
import com.clust4j.utils.MinkowskiDistance;

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
		
		HaversineDistance km = new HaversineDistance(DistanceUnit.KM);
		assertTrue(km.getDistance(a, b) == 14.97319048158622);
		
		HaversineDistance mi = new HaversineDistance();
		assertTrue(mi.getDistance(a, b) == 9.304482988008138);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testHaversineFail() {
		final double[] a = new double[]{47.6788206, -122.3271205, 0d};
		final double[] b = new double[]{47.6788206, -122.5271205, 1d};
		
		HaversineDistance km = new HaversineDistance(DistanceUnit.KM);
		km.getDistance(a, b);
	}
	
	@Test
	public void testCheb() {
		final double[] a = new double[]{0,1,2,3};
		final double[] b = new double[]{10,9,8,7};
		assertTrue(Distance.CHEBYSHEV.getDistance(a, b) == 10);
	}
}
