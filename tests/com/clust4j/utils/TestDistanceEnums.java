package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;
import org.apache.commons.math3.exception.DimensionMismatchException;
import com.clust4j.utils.Distance;
import com.clust4j.utils.MinkowskiDistance;

public class TestDistanceEnums {

	@Test
	public void testEuclidean() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		assertTrue(Distance.EUCLIDEAN.distance(a, b) == 5.0);
	}
	
	@Test
	public void testManhattan() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		assertTrue(Distance.MANHATTAN.distance(a, b) == 7.0);
	}
	
	@Test
	public void testMinkowski1() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		assertTrue(new MinkowskiDistance(2d).distance(a, b) == Distance.EUCLIDEAN.distance(a, b));
	}
	
	@Test
	public void testMinkowski2() {
		final double[] a = new double[] {0d, 0d};
		final double[] b = new double[] {3d, 4d};
		assertTrue(new MinkowskiDistance(1d).distance(a, b) == Distance.MANHATTAN.distance(a, b));
	}

	@Test(expected=DimensionMismatchException.class)
	public void testEuclideanFail1() {
		final double[] a = new double[] {0d, 0d, 0d};
		final double[] b = new double[] {3d, 4d};
		Distance.EUCLIDEAN.distance(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testManhattanFail1() {
		final double[] a = new double[] {0d, 0d, 0d};
		final double[] b = new double[] {3d, 4d};
		Distance.MANHATTAN.distance(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testEuclideanFail2() {
		final double[] a = new double[0];
		final double[] b = new double[0];
		Distance.EUCLIDEAN.distance(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testManhattanFail2() {
		final double[] a = new double[]{};
		final double[] b = new double[]{};
		Distance.MANHATTAN.distance(a, b);
	}
	
	@Test
	public void testNaN() {
		final double[] a = new double[] {0d, 0d, 0d};
		final double[] b = new double[] {3d, 4d, Double.NaN};
		assertTrue( Double.isNaN(Distance.MANHATTAN.distance(a, b)) );
		assertTrue( Double.isNaN(Distance.EUCLIDEAN.distance(a, b)) );
	}
}
