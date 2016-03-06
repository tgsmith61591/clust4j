package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

public class SeriesTests {

	@Test
	public void test() {
		final double a = 3.0, b = 5.0;
		assertTrue(Series.eval(a, Inequality.LT, b));
		assertFalse(Series.eval(b, Inequality.LT, a));
		
		assertTrue(Series.eval(a, Inequality.LTOET, b));
		assertFalse(Series.eval(b, Inequality.LTOET, a));

		assertTrue(Series.eval(b, Inequality.GT, a));
		assertFalse(Series.eval(a, Inequality.GT, b));

		assertTrue(Series.eval(b, Inequality.GTOET, a));
		assertFalse(Series.eval(a, Inequality.GTOET, b));

		assertTrue(Series.eval(a, Inequality.ET, a));
		assertFalse(Series.eval(a, Inequality.ET, b));
	}

}
