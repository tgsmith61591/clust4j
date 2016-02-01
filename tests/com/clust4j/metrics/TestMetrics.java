package com.clust4j.metrics;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestMetrics {

	@Test
	public void testAcc() {
		assertTrue(ClassificationScoring.ACCURACY.evaluate(
				new int[]{1,1,1,0}, 
				new int[]{1,1,1,1}) == 0.75);
	}

}
