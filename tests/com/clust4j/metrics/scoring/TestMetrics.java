package com.clust4j.metrics.scoring;

import static org.junit.Assert.*;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.data.DataSet;
import com.clust4j.metrics.scoring.BinomialClassificationScoring;
import com.clust4j.metrics.scoring.SilhouetteScore;
import com.clust4j.utils.VecUtils;

public class TestMetrics {
	final static DataSet IRIS = TestSuite.IRIS_DATASET.copy();
	
	@Test
	public void testAcc() {
		assertTrue(BinomialClassificationScoring.ACCURACY.evaluate(
				new int[]{1,1,1,0}, 
				new int[]{1,1,1,1}) == 0.75);
	}

	@Test
	public void testSilhouetteScore() {
		Array2DRowRealMatrix X = IRIS.getData();
		final int[] labels = IRIS.getLabels();
		
		double silhouette = SilhouetteScore
			.getInstance()
			.evaluate(X, labels);
		assertTrue(silhouette == 0.5032506980665507);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testSilhouetteScoreDME() {
		Array2DRowRealMatrix X = IRIS.getData();
		final int[] labels = new int[]{1,2,3};
		
		SilhouetteScore
			.getInstance()
			.evaluate(X, labels);
	}
	
	@Test
	public void testSilhouetteScoreNaN() {
		Array2DRowRealMatrix X = IRIS.getData();
		final int[] labels = VecUtils.repInt(1, X.getRowDimension());
		
		assertTrue(Double.isNaN(SilhouetteScore
			.getInstance()
			.evaluate(X, labels)));
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testDME() {
		BinomialClassificationScoring.ACCURACY.evaluate(new int[]{1,2}, new int[]{1,2,3});
	}
}
