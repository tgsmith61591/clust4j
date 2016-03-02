package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

public class FormatterTests {
	final MatrixFormatter formatter = new MatrixFormatter();

	@Test
	public void testMatrixUniform() {
		double[][] d = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6}
		};
		
		formatter.format(d);
		assertTrue(true); // get rid of import warning...
	}
	
	@Test
	public void testMatrixNonUniform() {
		double[][] d = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5}
		};
		
		formatter.format(d);
	}

	@Test
	public void testMatrixEmptyCols() {
		double[][] d = new double[][]{
			new double[]{},
			new double[]{}
		};
		
		formatter.format(d);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMatrixEmptyRows() {
		double[][] d = new double[][]{ };
		formatter.format(d);
	}
}
