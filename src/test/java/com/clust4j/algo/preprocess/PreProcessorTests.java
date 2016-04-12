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
package com.clust4j.algo.preprocess;

import static org.junit.Assert.*;

import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.except.NonUniformMatrixException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class PreProcessorTests {

	@Test
	public void testMeanCenter() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final double[][] operated = FeatureNormalization.MEAN_CENTER.operate(data);
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 0)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 1)), 0, Precision.EPSILON));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 2)), 0, Precision.EPSILON));
	}
	
	@Test
	public void testCenterScale() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final double[][] operated = FeatureNormalization.STANDARD_SCALE.operate(data);
		
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 0)), 0, 1e-12));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 1)), 0, 1e-12));
		assertTrue(Precision.equals(VecUtils.mean(MatUtils.getColumn(operated, 2)), 0, 1e-12));
		
		assertTrue(Precision.equals(VecUtils.stdDev(MatUtils.getColumn(operated, 0)), 1, 1e-12));
		assertTrue(Precision.equals(VecUtils.stdDev(MatUtils.getColumn(operated, 1)), 1, 1e-12));
		assertTrue(Precision.equals(VecUtils.stdDev(MatUtils.getColumn(operated, 2)), 1, 1e-12));
	}
	
	@Test
	public void testMinMaxScale() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final double[][] operated = FeatureNormalization.MIN_MAX_SCALE.operate(data);
		for(int i = 0; i < operated[0].length; i++) {
			double[] col = MatUtils.getColumn(operated, i);
			assertTrue(VecUtils.min(col) >= 0);
			assertTrue(VecUtils.max(col) <= 1);
		}
	}

	@Test(expected=NonUniformMatrixException.class)
	public void testNUME1() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		FeatureNormalization.MEAN_CENTER.operate(data);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testNUME2() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		FeatureNormalization.MIN_MAX_SCALE.operate(data);
	}
	
	@Test(expected=NonUniformMatrixException.class)
	public void testNUME3() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		FeatureNormalization.STANDARD_SCALE.operate(data);
	}
	
	@Test
	public void testMinMaxScalerBadMinMax() {
		boolean a = false;
		final int orig_min = GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MIN;
		final int orig_max = GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MAX;
		
		try {
			GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MIN = 1;
			GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MAX = 1;
			
			double[][] d = new double[][]{
				new double[]{1,2,3},
				new double[]{1,2,3}
			};
			
			FeatureNormalization.MIN_MAX_SCALE.operate(d);
		} catch(IllegalStateException i) {
			a = true;
		} finally {
			GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MIN = orig_min;
			GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MAX = orig_max;
			assertTrue(a);
		}
	}
}
