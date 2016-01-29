package com.clust4j;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestGlobals {

	@Test
	public void testGammaFunctions() {
		// Test in the different ranges
		double val = 0.5;
		assertTrue(GlobalState.Mathematics.gamma(val) == 1.772453850905516);
		assertTrue(GlobalState.Mathematics.lgamma(val) == 0.5723649429247001);
		
		val = 6.0;
		assertTrue(GlobalState.Mathematics.gamma(val) == 120);
		assertTrue(GlobalState.Mathematics.lgamma(val) == 4.787491742782046);
		
		val = 12.1;
		assertTrue(GlobalState.Mathematics.gamma(val) == 5.098322784411637E7);
		assertTrue(GlobalState.Mathematics.lgamma(val) == 17.747007270798743);
	}

}
