package com.clust4j.utils;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.algo.NearestNeighbors;
import com.clust4j.utils.MatUtils.Axis;
import com.clust4j.utils.MatUtils.MatSeries;

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
		assertTrue(GlobalState.Mathematics.TINY > 0);
		assertTrue(GlobalState.Mathematics.EPS > 0);
		assertTrue(GlobalState.Mathematics.TINY*100 > 0);
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
	public void testRowColSumsMeans() {
		final double[] a = new double[]{4,4,4};
		final double[] b = new double[]{0,3,9};
		final double[] c = new double[]{4.0/3.0, 4.0/3.0, 4.0/3.0};
		final double[] d = new double[]{0.0, 1.0, 3.0};
		
		final double[][] data = new double[][] {
			new double[] {0.000, 	 0.000,     0.000},
			new double[] {1.000,     1.000,     1.000},
			new double[] {3.000,     3.000,     3.000}
		};
		
		assertTrue(VecUtils.equalsExactly(a, MatUtils.colSums(data)));
		assertTrue(VecUtils.equalsExactly(b, MatUtils.rowSums(data)));
		assertTrue(VecUtils.equalsExactly(c, MatUtils.colMeans(data)));
		assertTrue(VecUtils.equalsExactly(d, MatUtils.rowMeans(data)));
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
		NearestNeighbors nn = new NearestNeighbors(new Array2DRowRealMatrix(mat, false), 
			new NearestNeighbors.NearestNeighborsPlanner(1)).fit();
		assertTrue( nn.getNeighbors(new Array2DRowRealMatrix(new double[][]{record},
			false)).getIndices()[0][0] == 0 );
	}
	
	@Test
	public void testTrans() {
		final double[][] mat = new double[][]{
			new double[] {-1.000, 	 -1.000,     -1.000},
			new double[] {10.000, 	 10.000,     10.000},
			new double[] {90.000,    90.000,     90.000}
		};
		
		Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(mat);
		assertTrue(MatUtils.equalsExactly(MatUtils.transpose(mat), matrix.transpose().getData()));
	}
	
	@Test
	public void testPartition() {
		final double[][] a = new double[][]{
			new double[]{0,1,2},
			new double[]{0,0,1},
			new double[]{0,0,0}
		};
		
		final double[][] b = new double[][]{
			new double[]{0,0,0},
			new double[]{0,1,2},
			new double[]{0,0,1}
		};
		
		final double[][] c = MatUtils.copy(a);
		//System.out.println(TestSuite.formatter.format(new Array2DRowRealMatrix(MatUtils.partitionByRow(a, 2))));
		assertTrue( MatUtils.equalsExactly(MatUtils.partitionByRow(a, 2), b) );
		assertTrue( MatUtils.equalsExactly(a, c) );
	}
	
	@Test
	public void testWhere() {
		final double[][] a = new double[][]{
			new double[]{6, 0},
			new double[]{7, 8}
		};
		
		final MatSeries ser = new MatSeries(a, Inequality.GT, 5);
		final double[][] b = new double[][]{
			new double[]{1,2},
			new double[]{3,4}
		};
		
		final double[][] c = new double[][]{
			new double[]{9,8},
			new double[]{7,6}
		};
		
		final double[][] d = new double[][]{
			new double[]{1,8},
			new double[]{3,4}
		};
		
		assertTrue(MatUtils.equalsExactly(d, MatUtils.where(ser, b, c)));
	}
	
	@Test
	public void testTransposeVector() {
		final double[] a = new double[]{1,2,3};
		final double[][] b = new double[][]{
			new double[]{1},
			new double[]{2},
			new double[]{3}
		};
		
		assertTrue(MatUtils.equalsExactly(b, MatUtils.transpose(a)));
	}
	
	@Test
	public void testFromVec() {
		final double[] a = new double[]{1,2,3};
		final double[][] b = new double[][]{
			new double[]{1,2,3},
			new double[]{1,2,3},
			new double[]{1,2,3}
		};
		
		assertTrue(MatUtils.equalsExactly(b, MatUtils.rep(a,3)));
	}
	
	@Test
	public void testWhere2() {
		final double[][] a = new double[][]{
			new double[]{0,1,1},
			new double[]{1,0,1},
			new double[]{0,0,1}
		};
		
		MatSeries ser = new MatSeries(a, Inequality.ET, 1);
		final double[] b = new double[]{2,3,4};
		final double[][] c = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9}
		};
		
		final double[][] d = new double[][]{
			new double[]{1,3,4},
			new double[]{2,5,4},
			new double[]{7,8,4}
		};
		
		assertTrue(MatUtils.equalsExactly(d, MatUtils.where(ser, b, c)));
	}
	
	@Test
	public void testReshape() {
		final double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9},
			new double[]{10,11,12},
			new double[]{13,14,15}
		};
		
		final double[][] b = MatUtils.reshape(a, 3, 5);
		//System.out.println(TestSuite.formatter.format(b));
		assertTrue(b.length == 3);
		assertTrue(b[0].length == 5);
	}
	
	@Test
	public void testAbsJagged() {
		final double[][] a = new double[][]{
			new double[]{1,-2,3},
			new double[]{4,6},
			new double[]{},
			new double[]{10,-11,12},
			new double[]{13,-14,-15}
		};
		
		final double[][] b = new double[][]{
			new double[]{1,2,3},
			new double[]{4,6},
			new double[]{},
			new double[]{10,11,12},
			new double[]{13,14,15}
		};
		
		assertTrue(MatUtils.equalsExactly(MatUtils.abs(a), b));
	}
	
	
	@Test(expected=NonUniformMatrixException.class)
	public void testAddJagged() {
		final double[][] a = new double[][]{
			new double[]{1,-2,3},
			new double[]{4,6},
			new double[]{},
			new double[]{10,-11,12},
			new double[]{13,-14,-15}
		};
		
		final double[][] b = new double[][]{
			new double[]{1,2,3},
			new double[]{4,6},
			new double[]{},
			new double[]{10,11,12},
			new double[]{13,14,15}
		};
		
		MatUtils.add(a,b);
	}
	
	
	@Test
	public void testAddEmpty() {
		final double[][] a = new double[][]{
			new double[]{},
			new double[]{},
			new double[]{},
			new double[]{},
			new double[]{}
		};
		
		final double[][] b = new double[][]{
			new double[]{},
			new double[]{},
			new double[]{},
			new double[]{},
			new double[]{}
		};
		
		assertTrue(MatUtils.equalsExactly(MatUtils.add(a,b), a));
	}
	
	@Test
	public void testArgMaxMin() {
		final double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9},
			new double[]{10,11,12},
			new double[]{13,14,15}
		};
		
		final int[] argMax = MatUtils.argMax(a, Axis.ROW);
		assertTrue(VecUtils.equalsExactly(argMax, new int[]{2,2,2,2,2}));
		
		final int[] argMaxCol = MatUtils.argMax(a, Axis.COL);
		assertTrue(VecUtils.equalsExactly(argMaxCol, new int[]{4,4,4}));
		
		final int[] argMin = MatUtils.argMin(a, Axis.ROW);
		assertTrue(VecUtils.equalsExactly(argMin, new int[]{0,0,0,0,0}));
		
		final int[] argMinCol = MatUtils.argMin(a, Axis.COL);
		assertTrue(VecUtils.equalsExactly(argMinCol, new int[]{0,0,0}));
	}
	
	@Test
	public void testArgMinMaxEmpty() {
		final double[][] a = new double[][]{};
		assertTrue(VecUtils.equalsExactly(new int[]{}, MatUtils.argMax(a, Axis.ROW)));
		assertTrue(VecUtils.equalsExactly(new int[]{}, MatUtils.argMax(a, Axis.COL)));
		assertTrue(VecUtils.equalsExactly(new int[]{}, MatUtils.argMin(a, Axis.ROW)));
		assertTrue(VecUtils.equalsExactly(new int[]{}, MatUtils.argMin(a, Axis.COL)));
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testArgMaxMinNUME1() {
		final double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4},
			new double[]{7,9},
			new double[]{10,11,12,12},
			new double[]{}
		};
		
		MatUtils.argMax(a, Axis.ROW);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testArgMaxMinNUME2() {
		final double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4},
			new double[]{7,9},
			new double[]{10,11,12,12},
			new double[]{}
		};
		
		MatUtils.argMin(a, Axis.ROW);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testColMeanSumNUME1() {
		final double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4},
			new double[]{7,9},
			new double[]{10,11,12,12},
			new double[]{}
		};
		
		MatUtils.colMeans(a);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testColMeanSumNUME2() {
		final double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4},
			new double[]{7,9},
			new double[]{10,11,12,12},
			new double[]{}
		};
		
		MatUtils.colSums(a);
	}
	
	@Test
	public void testCompleteCases() {
		final double[][] a = new double[][]{new double[]{}};
		assertTrue(MatUtils.equalsExactly(MatUtils.completeCases(a), a));
		
		final double[][] b = new double[][]{
			new double[]{1,2,3},
			new double[]{Double.NaN, 2,3},
			new double[]{4,2,3}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(b);
		
		final double[][] c = new double[][]{
			new double[]{1,2,3},
			new double[]{4,2,3}
		};
		
		assertTrue(MatUtils.containsNaN(b));
		assertTrue(MatUtils.containsNaN(mat));
		assertFalse(MatUtils.containsNaN(c));
		assertTrue(MatUtils.containsNaNDistributed(b));
		assertTrue(MatUtils.containsNaNDistributed(mat));
		assertFalse(MatUtils.containsNaNDistributed(a));
		assertFalse(MatUtils.containsNaNDistributed(c));
		assertFalse(MatUtils.containsNaNDistributed(new Array2DRowRealMatrix(c)));
		assertTrue(MatUtils.equalsExactly(MatUtils.completeCases(mat), c));
	}
	
	@Test
	public void testCopyDouble() {
		double[][] a = new double[][]{new double[]{}};
		double[][] b = MatUtils.copy(a);
		assertTrue(MatUtils.equalsExactly(a, b));
		
		a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6}
		};
		
		b = MatUtils.copy(a);
		b[0][0] = 9;
		
		assertFalse(a[0][0] == b[0][0]);
	}
	
	@Test
	public void testCopyBoolean() {
		boolean[][] a = new boolean[][]{new boolean[]{}};
		boolean[][] b = MatUtils.copy(a);
		assertTrue(MatUtils.equalsExactly(a, b));
		
		a = new boolean[][]{
			new boolean[]{true, false, true},
			new boolean[]{false, true, false}
		};
		
		b = MatUtils.copy(a);
		b[0][0] = false;
		
		assertFalse(a[0][0] == b[0][0]);
	}
	
	@Test
	public void testCopyInt() {
		int[][] a = new int[][]{new int[]{}};
		int[][] b = MatUtils.copy(a);
		assertTrue(MatUtils.equalsExactly(a, b));
		
		a = new int[][]{
			new int[]{0,1,0},
			new int[]{1,0,1}
		};
		
		b = MatUtils.copy(a);
		b[0][0] = 1;
		
		assertFalse(a[0][0] == b[0][0]);
	}
	
	@Test
	public void testDiagonal() {
		double[][] a= new double[][]{
			new double[]{1,0,0},
			new double[]{0,1,0},
			new double[]{0,0,1}
		};
		
		assertTrue(VecUtils.equalsExactly(MatUtils.diagFromSquare(a), new double[]{1,1,1}));
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testDiagonalDME() {
		double[][] a= new double[][]{
			new double[]{1,0,0},
			new double[]{0,1,0}
		};
		
		MatUtils.diagFromSquare(a);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testDiagonalNUME() {
		double[][] a= new double[][]{
			new double[]{1,0,0},
			new double[]{0,1},
			new double[]{0,0,1}
		};
		
		MatUtils.diagFromSquare(a);
	}
	
	@Test
	public void testJaggedMultiArrayDims() {
		double[][] a = new double[][]{
			new double[]{1,2,4},
			new double[]{2}
		};
		
		double[][] b = new double[][]{
			new double[]{1,2,4},
			new double[]{2}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test
	public void testJaggedMultiIntArrayDims() {
		int[][] a = new int[][]{
			new int[]{1,2,4},
			new int[]{2}
		};
		
		int[][] b = new int[][]{
			new int[]{1,2,4},
			new int[]{2}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test
	public void testJaggedMultiBooleanArrayDims() {
		boolean[][] a = new boolean[][]{
			new boolean[]{true, false, true},
			new boolean[]{true}
		};
		
		boolean[][] b = new boolean[][]{
			new boolean[]{false, true, true},
			new boolean[]{false}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testJaggedMultiArrayDims2() {
		double[][] a = new double[][]{
			new double[]{1,2},
			new double[]{2}
		};
		
		double[][] b = new double[][]{
			new double[]{1,2,4},
			new double[]{2}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testJaggedMultiIntArrayDims2() {
		int[][] a = new int[][]{
			new int[]{1,2},
			new int[]{2}
		};
		
		int[][] b = new int[][]{
			new int[]{1,2,4},
			new int[]{2}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testJaggedMultiBooleanArrayDims2() {
		boolean[][] a = new boolean[][]{
			new boolean[]{true, false},
			new boolean[]{false}
		};
		
		boolean[][] b = new boolean[][]{
			new boolean[]{false, false, false},
			new boolean[]{true}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testEmptyMultiArrayDims() {
		double[][] a= new double[][]{};
		double[][] b= new double[][]{};
		MatUtils.checkDims(a,b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testEmptyMultiIntArrayDims() {
		int[][] a= new int[][]{};
		int[][] b= new int[][]{};
		MatUtils.checkDims(a,b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testEmptyMultiBooleanArrayDims() {
		boolean[][] a= new boolean[][]{};
		boolean[][] b= new boolean[][]{};
		MatUtils.checkDims(a,b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testJaggedMultiArrayDims3() {
		double[][] a = new double[][]{
			new double[]{1,2,3}
		};
		
		double[][] b = new double[][]{
			new double[]{1,2,4},
			new double[]{2}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testJaggedMultiIntArrayDims3() {
		int[][] a = new int[][]{
			new int[]{1,2,3}
		};
		
		int[][] b = new int[][]{
			new int[]{1,2,4},
			new int[]{2}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testJaggedMultiBooleanArrayDims3() {
		boolean[][] a = new boolean[][]{
			new boolean[]{false, false, true}
		};
		
		boolean[][] b = new boolean[][]{
			new boolean[]{true, true, true},
			new boolean[]{false}
		};
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testJaggedMultiArrayDims4() {
		double[][] a = new double[5][];
		double[][] b = new double[5][];
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testJaggedMultiIntArrayDims4() {
		int[][] a = new int[5][];
		int[][] b = new int[5][];
		
		MatUtils.checkDims(a, b);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testJaggedMultiBooleanArrayDims4() {
		boolean[][] a = new boolean[5][];
		boolean[][] b = new boolean[5][];
		
		MatUtils.checkDims(a, b);
	}
	
	@Test
	public void testMultiArrayUniformity1() {
		double[][] a = new double[][]{
			new double[]{1,2,7},
			new double[]{2,2,3}
		};
		
		double[][] b = new double[][]{
			new double[]{1,2,4},
			new double[]{2,1,9}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test
	public void testMultiArrayIntUniformity1() {
		int[][] a = new int[][]{
			new int[]{1,2,7},
			new int[]{2,2,3}
		};
		
		int[][] b = new int[][]{
			new int[]{1,2,4},
			new int[]{2,1,9}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test
	public void testMultiArrayBooleanUniformity1() {
		boolean[][] a = new boolean[][]{
			new boolean[]{true, true, true},
			new boolean[]{true, true, true}
		};
		
		boolean[][] b = new boolean[][]{
			new boolean[]{true, false, true},
			new boolean[]{false, true, true}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testMultiArrayUniformity2() {
		double[][] a = new double[][]{
			new double[]{1,2,7}
		};
		
		double[][] b = new double[][]{
			new double[]{1,2,4},
			new double[]{2,1,9}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testMultiArrayIntUniformity2() {
		int[][] a = new int[][]{
			new int[]{1,2,7}
		};
		
		int[][] b = new int[][]{
			new int[]{1,2,4},
			new int[]{2,1,9}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testMultiArrayBooleanUniformity2() {
		boolean[][] a = new boolean[][]{
			new boolean[]{false, true, false}
		};
		
		boolean[][] b = new boolean[][]{
			new boolean[]{true, true, true},
			new boolean[]{false, true, true}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testMultiArrayUniformity3() {
		double[][] a = new double[][]{
			new double[]{1,7},
			new double[]{9,8}
		};
		
		double[][] b = new double[][]{
			new double[]{1,2,4},
			new double[]{2,1,9}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testMultiArrayIntUniformity3() {
		int[][] a = new int[][]{
			new int[]{1,7},
			new int[]{9,8}
		};
		
		int[][] b = new int[][]{
			new int[]{1,2,4},
			new int[]{2,1,9}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testMultiArrayBooleanUniformity3() {
		boolean[][] a = new boolean[][]{
			new boolean[]{true, true},
			new boolean[]{true, true}
		};
		
		boolean[][] b = new boolean[][]{
			new boolean[]{false, false, false},
			new boolean[]{true, true, true}
		};
		
		MatUtils.checkDimsForUniformity(a, b);
	}
	
	@Test
	public void testEqualsWithTolerance() {
		double[][] a= new double[][]{
			new double[]{0.00000000000000000000000001, 0}
		};
		double[][] b= new double[][]{
			new double[]{0, 0}
		};
		assertTrue(MatUtils.equalsWithTolerance(a, b));
		assertTrue(MatUtils.equalsWithTolerance(a, b, 1e-5));
		assertFalse(MatUtils.equalsWithTolerance(a, b, 1e-35));
	}
	
	@Test
	public void testIntEqualsExactly() {
		int[][] a= new int[][]{new int[]{1,2,4}};
		int[][] b= new int[][]{new int[]{1,2,3}};
		assertFalse(MatUtils.equalsExactly(a, b));
	}
	
	@Test
	public void testBooleanEqualsExactly() {
		boolean[][] a= new boolean[][]{new boolean[]{false, false, false}};
		boolean[][] b= new boolean[][]{new boolean[]{false, false, true}};
		assertFalse(MatUtils.equalsExactly(a, b));
	}
	
	@Test
	public void testFlatten1() {
		double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6}
		};
		
		assertTrue(VecUtils.equalsExactly(new double[]{1,2,3,4,5,6}, MatUtils.flatten(a)));
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testFlatten2() {
		double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5}
		};
		
		MatUtils.flatten(a);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testFlatten3() {
		double[][] a = new double[][]{ };
		MatUtils.flatten(a);
	}
	
	@Test
	public void testIntFlatten1() {
		int[][] a = new int[][]{
			new int[]{1,2,3},
			new int[]{4,5,6}
		};
		
		assertTrue(VecUtils.equalsExactly(new int[]{1,2,3,4,5,6}, MatUtils.flatten(a)));
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testIntFlatten2() {
		int[][] a = new int[][]{
			new int[]{1,2,3},
			new int[]{4,5}
		};
		
		MatUtils.flatten(a);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIntFlatten3() {
		int[][] a = new int[][]{ };
		MatUtils.flatten(a);
	}
	
	@Test
	public void testFlattenUpperTriangular1() {
		double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6},
			new double[]{7,8,9}
		};
		
		assertTrue(VecUtils.equalsExactly(MatUtils
			.flattenUpperTriangularMatrix(a), new double[]{2,3,6}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testFlattenUpperTriangular2() {
		double[][] a = new double[][]{
		};
		
		MatUtils.flattenUpperTriangularMatrix(a);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testFlattenUpperTriangular3() {
		double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5},
			new double[]{7,8,9}
		};
		
		MatUtils.flattenUpperTriangularMatrix(a);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testFlattenUpperTriangular4() {
		double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6}
		};
		
		MatUtils.flattenUpperTriangularMatrix(a);
	}
	
	@Test
	public void testFlooring1() {
		double[][] a = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6}
		};
		
		assertTrue(MatUtils.equalsExactly(MatUtils.floor(a, 3, 0), 
			new double[][]{
				new double[]{0,0,3},
				new double[]{4,5,6}
		}));
	}
	
	@Test
	public void testFlooring2() {
		double[][] a = new double[][]{
			new double[]{},
			new double[]{}
		};
		
		assertTrue(MatUtils.equalsExactly(MatUtils.floor(a, 3, 0), 
			new double[][]{
				new double[]{},
				new double[]{}
		}));
	}
	
	@Test
	public void testFromVector2() {
		double[][] expected = new double[][]{
			new double[]{1,1}
		};
		
		double[][] expected2 = new double[][]{
			new double[]{1},
			new double[]{1}
		};
		
		assertTrue(MatUtils.equalsExactly(expected, 
			MatUtils.fromVector(new double[]{1}, 2, Axis.ROW)));
		assertTrue(MatUtils.equalsExactly(expected2, 
				MatUtils.fromVector(new double[]{1}, 2, Axis.COL)));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testFromVector3() {
		MatUtils.fromVector(new double[]{1}, 0, Axis.COL);
	}
	
	@Test
	public void testFromList() {
		final ArrayList<double[]> in = new ArrayList<>();
		in.add(new double[]{1,2,3,4});
		in.add(new double[]{});
		in.add(new double[]{1});
		
		final double[][] out = new double[][]{
			new double[]{1,2,3,4},
			new double[]{},
			new double[]{1}
		};
		
		assertTrue(MatUtils.equalsExactly(out, MatUtils.fromList(in)));
	}
	
	@Test
	public void testGetColumn() {
		double[][] ad = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6}
		};
		
		int[][] ai = new int[][]{
			new int[]{1,2,3},
			new int[]{4,5,6}
		};
		
		assertTrue(VecUtils.equalsExactly(new double[]{1,4}, MatUtils.getColumn(ad, 0)));
		assertTrue(VecUtils.equalsExactly(new int[]{1,4}, MatUtils.getColumn(ai, 0)));
		assertTrue(VecUtils.equalsExactly(new double[]{3,6}, MatUtils.getColumn(ad, 2)));
		assertTrue(VecUtils.equalsExactly(new int[]{3,6}, MatUtils.getColumn(ai, 2)));
	}
	
	@Test(expected=IndexOutOfBoundsException.class)
	public void testGetColumnException1() {
		MatUtils.getColumn(new double[][]{new double[]{}}, -1);
	}
	
	@Test(expected=IndexOutOfBoundsException.class)
	public void testGetColumnException2() {
		MatUtils.getColumn(new int[][]{new int[]{}}, -1);
	}
	
	@Test(expected=IndexOutOfBoundsException.class)
	public void testGetColumnException3() {
		MatUtils.getColumn(new double[][]{new double[]{1,2,3}}, 3);
	}
	
	@Test(expected=IndexOutOfBoundsException.class)
	public void testGetColumnException4() {
		MatUtils.getColumn(new int[][]{new int[]{1,2,3}}, 3);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testGetColumnException5() {
		MatUtils.getColumn(new double[][]{new double[]{1,2,3}, new double[]{1}}, 0);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testGetColumnException6() {
		MatUtils.getColumn(new int[][]{new int[]{1,2,3}, new int[]{1}}, 0);
	}
	
	@Test
	public void testGetColumns() {
		double[][] a = new double[][]{
			new double[]{0,1,2,3},
			new double[]{4,5,6,7},
			new double[]{8,9,10,11}
		};
		
		assertTrue(MatUtils.equalsExactly(
				MatUtils.getColumns(a, new int[]{0,0,2}),
				new double[][]{
					new double[]{0,0,2},
					new double[]{4,4,6},
					new double[]{8,8,10}
		}));
		
		assertTrue(MatUtils.equalsExactly(
				MatUtils.getColumns(a, new Integer[]{0,0,2}),
				new double[][]{
					new double[]{0,0,2},
					new double[]{4,4,6},
					new double[]{8,8,10}
		}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testGetColumnsEmpty() {
		MatUtils.getColumns(new double[][]{}, new Integer[]{0,0,2});
	}
	
	@Test
	public void testGetRows() {
		double[][] a = new double[][]{
			new double[]{0,1,2,3},
			new double[]{4,5,6,7},
			new double[]{8,9,10,11}
		};
		
		assertTrue(MatUtils.equalsExactly(
				MatUtils.getRows(a, new int[]{0,0,2}),
				new double[][]{
					new double[]{0,1,2,3},
					new double[]{0,1,2,3},
					new double[]{8,9,10,11}
		}));
		
		assertTrue(MatUtils.equalsExactly(
				MatUtils.getRows(a, new Integer[]{0,0,2}),
				new double[][]{
					new double[]{0,1,2,3},
					new double[]{0,1,2,3},
					new double[]{8,9,10,11}
		}));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testGetRowsEmpty() {
		MatUtils.getRows(new double[][]{}, new Integer[]{0,0,2});
	}
}
