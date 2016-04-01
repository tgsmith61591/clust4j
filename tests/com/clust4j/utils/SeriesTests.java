package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;
import com.clust4j.utils.Series.Inequality;

public class SeriesTests {

	@Test
	public void test() {
		final double a = 3.0, b = 5.0;
		assertTrue(Series.eval(a, Inequality.LESS_THAN, b));
		assertFalse(Series.eval(b, Inequality.LESS_THAN, a));
		
		assertTrue(Series.eval(a, Inequality.LESS_THAN_OR_EQUAL_TO, b));
		assertFalse(Series.eval(b, Inequality.LESS_THAN_OR_EQUAL_TO, a));

		assertTrue(Series.eval(b, Inequality.GREATER_THAN, a));
		assertFalse(Series.eval(a, Inequality.GREATER_THAN, b));

		assertTrue(Series.eval(b, Inequality.GREATER_THAN_OR_EQUAL_TO, a));
		assertFalse(Series.eval(a, Inequality.GREATER_THAN_OR_EQUAL_TO, b));

		assertTrue(Series.eval(a, Inequality.EQUAL_TO, a));
		assertFalse(Series.eval(a, Inequality.EQUAL_TO, b));
	}

}
