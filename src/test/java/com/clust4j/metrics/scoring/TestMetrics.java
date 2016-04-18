/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.metrics.scoring;

import static org.junit.Assert.*;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.data.DataSet;
import com.clust4j.metrics.scoring.SupervisedMetric;
import com.clust4j.utils.VecUtils;

import static com.clust4j.metrics.scoring.UnsupervisedMetric.SILHOUETTE;

public class TestMetrics {
	final static DataSet IRIS = TestSuite.IRIS_DATASET.copy();
	
	@Test
	public void testAcc() {
		assertTrue(SupervisedMetric.BINOMIAL_ACCURACY.evaluate(
				new int[]{1,1,1,0}, 
				new int[]{1,1,1,1}) == 0.75);
	}

	@Test
	public void testSilhouetteScore() {
		Array2DRowRealMatrix X = IRIS.getData();
		final int[] labels = IRIS.getLabels();
		
		double silhouette = SILHOUETTE
			.evaluate(X, labels);
		assertTrue(silhouette == 0.5032506980665507);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testSilhouetteScoreDME() {
		Array2DRowRealMatrix X = IRIS.getData();
		final int[] labels = new int[]{1,2,3};
		
		SILHOUETTE.evaluate(X, labels);
	}
	
	@Test
	public void testSilhouetteScoreNaN() {
		Array2DRowRealMatrix X = IRIS.getData();
		final int[] labels = VecUtils.repInt(1, X.getRowDimension());
		
		assertTrue(Double.isNaN(SILHOUETTE
			.evaluate(X, labels)));
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testDME() {
		SupervisedMetric.BINOMIAL_ACCURACY.evaluate(new int[]{1,2}, new int[]{1,2,3});
	}
	
	@Test
	public void testIndexAffinityExceptionHandling() {
		final int[] a = new int[]{0,0,0,1,1};
		final int[] b = new int[]{0,0,0,1,2};
		
		boolean c = false;
		try {
			SupervisedMetric.INDEX_AFFINITY.evaluate(a, new int[]{0,0});
		} catch(DimensionMismatchException d) {
			c = true;
		} finally {
			assertTrue(c);
		}
		
		c = false;
		try {
			SupervisedMetric.INDEX_AFFINITY.evaluate(new int[]{}, new int[]{});
		} catch(IllegalArgumentException d) {
			c = true;
		} finally {
			assertTrue(c);
		}
		
		c = false;
		try {
			SupervisedMetric.INDEX_AFFINITY.evaluate(a,b);
		} catch(IllegalArgumentException d) {
			c = true;
		} finally {
			assertTrue(c);
		}
		
		assertTrue(SupervisedMetric.INDEX_AFFINITY.evaluate(new int[]{0}, new int[]{9}) == 1.0);
		assertTrue(SupervisedMetric.INDEX_AFFINITY.evaluate(new int[]{0,1,2}, new int[]{9,5,4}) == 1.0);
	}
}
