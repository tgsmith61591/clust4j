package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

import com.clust4j.algo.NearestNeighbors;
import com.clust4j.utils.MatUtils.Axis;

public class MatTests {

	@Test
	public void test() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue(VecUtils.equalsExactly(data[1], MatUtils.meanRecord(data)));
		assertTrue(VecUtils.equalsExactly(data[1], MatUtils.medianRecord(data)));
	}

	@Test
	public void testTinyEps() {
		assertTrue(MatUtils.TINY > 0);
		assertTrue(MatUtils.EPS > 0);
		assertTrue(MatUtils.TINY*100 > 0);
	}
	
	@Test
	public void test2() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		final double[][] data2 = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue(MatUtils.equalsExactly(data, data2));
	}
	
	@Test
	public void test3() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		final double[][] data2 = new double[][] {
			new double[] {1.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		final double[][] data3 = new double[][] {
			new double[] {1.000, 	 0.000,     0.000},
			new double[] {3.000,     3.000,     3.000},
			new double[] {6.000,     6.000,     6.000}
		};
		
		assertTrue(MatUtils.equalsExactly(data3, MatUtils.add(data, data2)));
	}
	
	@Test
	public void testArgs() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue(VecUtils.equalsExactly(new int[]{2,2,2}, MatUtils.argMax(data, MatUtils.Axis.COL)));
		assertTrue(VecUtils.equalsExactly(new int[]{0,0,0}, MatUtils.argMin(data, MatUtils.Axis.COL)));
		assertTrue(VecUtils.equalsExactly(new int[]{0,0,0},   MatUtils.argMax(data, MatUtils.Axis.ROW)));
		assertTrue(VecUtils.equalsExactly(new int[]{0,0,0},   MatUtils.argMin(data, MatUtils.Axis.ROW)));
	}
	
	@Test
	public void testMinMaxes() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.500,     1.500,     1.500},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue(VecUtils.equalsExactly(new double[]{3,3,3}, MatUtils.max(data, MatUtils.Axis.COL)));
		assertTrue(VecUtils.equalsExactly(new double[]{0,0,0}, MatUtils.min(data, MatUtils.Axis.COL)));
		assertTrue(VecUtils.equalsExactly(new double[]{0,1.5,3}, MatUtils.max(data, MatUtils.Axis.ROW)));
		assertTrue(VecUtils.equalsExactly(new double[]{0,1.5,3}, MatUtils.min(data, MatUtils.Axis.ROW)));
	}
	
	@Test
	public void testFromVector() {
		final double[] a = new double[]{0,1,3};
		
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		final double[][] data2 = new double[][] {
			new double[] {0.000, 	 1.000,     3.000},
			new double[] {0.000, 	 1.000,     3.000},
			new double[] {0.000, 	 1.000,     3.000}
		};
		
		assertTrue(MatUtils.equalsExactly(MatUtils.fromVector(a, 3, Axis.ROW), data));
		assertTrue(MatUtils.equalsExactly(MatUtils.fromVector(a, 3, Axis.COL), data2));
	}
	
	@Test
	public void testRowColSums() {
		final double[] a = new double[]{4,4,4};
		final double[] b = new double[]{0,3,9};
		
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue(VecUtils.equalsExactly(a, MatUtils.colSums(data)));
		assertTrue(VecUtils.equalsExactly(b, MatUtils.rowSums(data)));
	}
	
	@Test
	public void testDiag() {
		final double[] a = new double[]{0,1,3};
		
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue(VecUtils.equalsExactly(MatUtils.diagFromSquare(data), a));
	}
	
	@Test
	public void testInPlace() {
		final double[] a = new double[]{0,1,3};
		
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		MatUtils.setRowInPlace(data, 0, a);
		assertTrue(VecUtils.equalsExactly(a, data[0]));
		
		MatUtils.setColumnInPlace(data, 0, a);
		assertTrue(VecUtils.equalsExactly(MatUtils.getColumn(data, 0), a));
	}
	
	@Test
	public void testFlatten() {
		final double[] a = new double[]{0,0,0,1,1,1,3,3,3};
		
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue( VecUtils.equalsExactly(MatUtils.flatten(data), a) );
	}
	
	@Test
	public void testCubing() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		int[] idcs = new int[]{0, 1};
		double[][] cube = MatUtils.getRows(MatUtils.getColumns(data, idcs), idcs);
		assertTrue(cube[0][0] == data[0][0]);
		assertTrue(cube[0][1] == data[0][1]);
		assertTrue(cube[1][0] == data[1][0]);
		assertTrue(cube[1][1] == data[1][1]);
		
		idcs = new int[]{1,2};
		cube = MatUtils.getRows(MatUtils.getColumns(data, idcs), idcs);
		assertTrue(cube[0][0] == data[1][1]);
		assertTrue(cube[0][1] == data[1][2]);
		assertTrue(cube[1][0] == data[2][1]);
		assertTrue(cube[1][1] == data[2][2]);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMatCheck() {
		MatUtils.checkDims(new double[5][]);
	}
	
	public void testMatVecScalarOperations() {
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		final double[] operator = new double[]{1,2,3};
		
		
		// Addition
		double[][] addedRowWise = MatUtils.scalarAdd(data, operator, Axis.ROW);
		assertTrue(MatUtils.equalsExactly(addedRowWise, new double[][]{
			new double[] {1.000, 	 1.000,     1.000},
			new double[] {3.000,     3.000,     3.000},
			new double[] {6.000,     6.000,     6.000}
		}));
		
		double[][] addedColWise = MatUtils.scalarAdd(data, operator, Axis.COL);
		assertTrue(MatUtils.equalsExactly(addedColWise, new double[][]{
			new double[] {1.000, 	 2.000,     3.000},
			new double[] {2.000,     3.000,     4.000},
			new double[] {4.000,     5.000,     6.000}
		}));
		
		
		// Subtraction
		double[][] subRowWise = MatUtils.scalarSubtract(data, operator, Axis.ROW);
		assertTrue(MatUtils.equalsExactly(subRowWise, new double[][]{
			new double[] {-1.000, 	 -1.000,     -1.000},
			new double[] {-1.000, 	 -1.000,     -1.000},
			new double[] { 0.000,     0.000,      0.000}
		}));
		
		double[][] subColWise = MatUtils.scalarSubtract(data, operator, Axis.COL);
		assertTrue(MatUtils.equalsExactly(subColWise, new double[][]{
			new double[] {-1.000, 	 -2.000,    -3.000},
			new double[] { 0.000,    -1.000,    -2.000},
			new double[] { 2.000,     1.000,     0.000}
		}));
		

		double[][] YM = MatUtils.fromVector(operator, 3, Axis.ROW);
		assertTrue(MatUtils.equalsExactly(subRowWise, MatUtils.subtract(data, YM)));
	}
	
	@Test
	public void testKNearestStatic() {
		final double[][] mat = new double[][]{
			new double[] {-1.000, 	 -1.000,     -1.000},
			new double[] {10.000, 	 10.000,     10.000},
			new double[] {90.000,    90.000,     90.000}
		};
		
		final double[] record = new double[]{0,0,0};
		assertTrue( NearestNeighbors.getKNearest(record, mat, 1, Distance.EUCLIDEAN)[0] == 0 );
	}
}
