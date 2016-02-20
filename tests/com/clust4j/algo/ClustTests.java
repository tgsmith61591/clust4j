package com.clust4j.algo;

import static org.junit.Assert.*;
import static com.clust4j.TestSuite.getRandom;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.DBSCAN;
import com.clust4j.utils.MatrixFormatter;
import com.clust4j.utils.NaNException;

public class ClustTests {
	{ // initializer
		com.clust4j.GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE = false;
	}
	
	private static boolean print = false;
	private static final MatrixFormatter formatter = new MatrixFormatter();
	

	@Test
	public void testFormatter() {
		final double[][] data = new double[][] {
			new double[] {0.0128275, 0.182751, 0.1284},
			new double[] {0.65816,   1.29518,  2.123316},
			new double[] {4.1234,    0.0001,   1.000002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		if(print) System.out.println(formatter.format(mat));
	}

	@Test
	public void mutabilityTest1() {
		final double eps = 0.3;
		final Array2DRowRealMatrix mat = getRandom(5,5);
		final double val11 = mat.getEntry(0, 0);
		
		DBSCAN db1 = new DBSCAN(mat, eps); // No scaling
		DBSCAN db2 = new DBSCAN(mat, new DBSCAN.DBSCANPlanner(eps).setScale(true));
		
		// Testing mutability of scaling
		assertTrue(db1.getData().getEntry(0, 0) == val11);
		assertFalse(db2.getData().getEntry(0, 0) == val11);
	}
	
	
	@Test(expected=NaNException.class)
	public void testNanException() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  Double.NaN,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		new NearestNeighbors(mat, 1);
	}
}
