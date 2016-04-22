package com.clust4j.load;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.TestSuite;
import com.clust4j.log.Log;
import com.clust4j.utils.MatUtils;

/**
 * A set of tests that are quite large. Not
 * to be included in the production test suite.
 * @author Taylor G Smith
 */
public class LoadTests {

	@Test
	public void testLargeMatrixMultiplicationTask() {
		// Force a massively distributed task... can take long time (if works...)!
		if(GlobalState.ParallelismConf.PARALLELISM_ALLOWED) {
			final int rows = GlobalState.ParallelismConf.MAX_SERIAL_VECTOR_LEN + 1;
			final int cols = 2;
			
			try {
				// 2 X 10,000,001
				Array2DRowRealMatrix A = TestSuite.getRandom(cols, rows);
				
				// 10,000,001 X 2
				Array2DRowRealMatrix B = TestSuite.getRandom(rows, cols);
				
				// Yield 2 X 2
				MatUtils.multiplyDistributed(A.getDataRef(), B.getDataRef());
			} catch(OutOfMemoryError e) {
				Log.info("could not complete large distributed multiplication due to heap space");
				fail();
			} finally { // don't want to fail tests just because of this...
				assertTrue(true);
			}
		}
	}
}
