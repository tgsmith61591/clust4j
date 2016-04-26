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

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.GlobalState;
import com.clust4j.algo.preprocess.impute.BootstrapImputation;
import com.clust4j.algo.preprocess.impute.MeanImputation;
import com.clust4j.algo.preprocess.impute.MedianImputation;
import com.clust4j.algo.preprocess.impute.NearestNeighborImputation;
import com.clust4j.algo.preprocess.impute.BootstrapImputation.BootstrapImputationPlanner;
import com.clust4j.algo.preprocess.impute.CentralTendencyMethod;
import com.clust4j.algo.preprocess.impute.MeanImputation.MeanImputationPlanner;
import com.clust4j.algo.preprocess.impute.MedianImputation.MedianImputationPlanner;
import com.clust4j.algo.preprocess.impute.NearestNeighborImputation.NNImputationPlanner;
import com.clust4j.except.NaNException;
import com.clust4j.sample.BootstrapTest;
import com.clust4j.sample.Bootstrapper;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.MatrixFormatter;

public class ImputationTests {
	final static MatrixFormatter formatter = new MatrixFormatter();

	@Test
	public void testMeanImputation() {
		final double[][] d = new double[][]{
			new double[]{Double.NaN, 1, 		 2},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{2,			 2,          1}
		};
		
		final MeanImputation mean = new MeanImputation(
			new MeanImputationPlanner()
				.setVerbose(true));
		
		final double[][] imputed = mean.transform(d);
		final double[][] res = new double[][]{
			new double[]{1.5,	1, 		2},
			new double[]{1, 	1.5, 	3},
			new double[]{2,		2,      1}
		};
		
		assertTrue(MatUtils.equalsExactly(res, imputed));
		System.out.println();
	}
	
	@Test
	public void testMedianImputation() {
		final double[][] d = new double[][]{
			new double[]{Double.NaN, 1, 		 2},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{2,			 2,          1}
		};
		
		final MedianImputation median = new MedianImputation(
			new MedianImputationPlanner().setSeed(GlobalState.DEFAULT_RANDOM_STATE));
		
		final double[][] imputed = median.transform(d);
		final double[][] res = new double[][]{
			new double[]{1.5,	1, 		2},
			new double[]{1, 	1.5, 	3},
			new double[]{2,		2,      1}
		};
		
		assertTrue(MatUtils.equalsExactly(res, imputed));
		assertTrue(MatUtils.equalsExactly(res, median.copy().transform(d)));
		assertTrue(MatUtils.equalsExactly(res, median.transform(new Array2DRowRealMatrix(d)).getData() ));
	}
	
	@Test
	public void testMedianImputation2() {
		final double[][] d = new double[][]{
			new double[]{Double.NaN, 1, 		 2},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{2,			 2,          1},
			new double[]{3,			 5,			 Double.NaN}
		};
		
		final MedianImputation median = new MedianImputation(
			new MedianImputationPlanner()
				.setVerbose(true));
		
		final double[][] imputed = median.transform(d);
		final double[][] res = new double[][]{
			new double[]{2,		1, 		2},
			new double[]{1, 	2, 		3},
			new double[]{2,		2,      1},
			new double[]{3,		5,		2}
		};
		
		assertTrue(MatUtils.equalsExactly(res, imputed));
		System.out.println();
	}
	
	@Test
	public void testBootstrapImputation() {
		final double[][] d = new double[][]{
			new double[]{Double.NaN, 1, 		 2},
			new double[]{1, 		 Double.NaN, 4},
			new double[]{2,			 2,          1},
			new double[]{5,			 7,			 12},
			new double[]{1.2,		 3,			 9},
			new double[]{2.8,		 0,			 0},
			new double[]{0,			 1.5,		 1},
			new double[]{3,			 5,			 Double.NaN}
		};
		
		final CentralTendencyMethod[] ctm = CentralTendencyMethod.values();
		final Bootstrapper[] strappers = Bootstrapper.values();
		
		double[][] res;
		for(CentralTendencyMethod method: ctm) {
			for(Bootstrapper strap: strappers) {
				res = new BootstrapImputation(
					new BootstrapImputationPlanner()
						.setBootstrapper(strap)
						.setVerbose(true)
						.setMethodOfCentralTendency(method))
					.transform(d);
				
				BootstrapTest.printMatrix(res);
			}
		}
		
		new BootstrapImputation().transform(new Array2DRowRealMatrix(d)); // ensure doesn't break;
	}
	
	@Test
	public void testKNNImputation() {
		final double[][] d = new double[][]{
			new double[]{1,	 		 1, 		 2},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{8.5,		 7.9,        6},
			new double[]{9,			 8,			 Double.NaN}
		};
		
		final NearestNeighborImputation nni = new NearestNeighborImputation(
			new NearestNeighborImputation.NNImputationPlanner()
				.setK(1).setVerbose(true));
		
		final double[][] imputed = nni.transform(d);
		final double[][] res = new double[][]{
			new double[]{1,	 		 1, 		 2},
			new double[]{1, 		 1, 		 3},
			new double[]{8.5,		 7.9,        6},
			new double[]{9,			 8,			 6}
		};
		
		assertTrue(MatUtils.equalsExactly(res, imputed));
		System.out.println();
	}

	@Test
	public void testFullyNaN() {
		boolean a = false;
		MeanImputation m = new MeanImputation();
		
		try {
			final double[][] d = new double[][]{
				new double[]{1,	 	 Double.NaN, 2},
				new double[]{1, 	 Double.NaN, 3},
				new double[]{8.5,	 Double.NaN, 6},
				new double[]{9,		 Double.NaN, 7}
			};
			
			m.transform(d);
		} catch(NaNException n) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * Coverage love...
		 */
		m.trace("coverage love");
		m.debug("coverage love");
		m.warn("blah");
		assertTrue(m.hasWarnings());
	}
	
	@Test
	public void testDefConst() {
		final double[][] d = new double[][]{
			new double[]{1,	 		 1, 		 2},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{8.5,		 7.9,        6},
			new double[]{9,			 8,			 Double.NaN},
			new double[]{3.5,		 2.9,        6.1},
			new double[]{3, 		 Double.NaN, 1},
			new double[]{0,	 		 0, 		 0},
			new double[]{2,	 		 4, 		 9},
			new double[]{1.4,	 	 5, 		 6},
		};
		
		new BootstrapImputation().transform(d);
		new MeanImputation().transform(d);
		new MedianImputation().transform(d);
		new NearestNeighborImputation().transform(d);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testNullBootstrapPlanner() {
		new BootstrapImputation(new BootstrapImputationPlanner().setBootstrapper(null));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testBadBootstrapPlannerRatio() {
		new BootstrapImputation(new BootstrapImputationPlanner().setRatio(0.0));
	}
	
	@Test
	public void testBootstrapFromCopyConst() {
		new BootstrapImputation().copy();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testNNWithNullCent() {
		new NearestNeighborImputation(new NNImputationPlanner().setMethodOfCentralTendency(null));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testNNWithBadK() {
		new NearestNeighborImputation(new NNImputationPlanner(0).setSeed(GlobalState.DEFAULT_RANDOM_STATE));
	}
	
	@Test
	public void testCopyNN() {
		NearestNeighborImputation nn = new NearestNeighborImputation(3);
		NearestNeighborImputation nn2= nn.copy();
		assertTrue(nn.getK() == nn2.getK());
		assertTrue(nn.getCentralTendency().equals(nn2.getCentralTendency()));
		
		final double[][] d = new double[][]{
			new double[]{1,	 		 1, 		 2},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{8.5,		 7.9,        6},
			new double[]{9,			 8,			 Double.NaN},
			new double[]{3.5,		 2.9,        6.1},
			new double[]{3, 		 Double.NaN, 1},
			new double[]{0,	 		 0, 		 0},
			new double[]{2,	 		 4, 		 9},
			new double[]{1.4,	 	 5, 		 6},
		};
		
		final Array2DRowRealMatrix a = new Array2DRowRealMatrix(d);
		assertTrue(MatUtils.equalsExactly(nn.transform(d), nn2.transform(d)));
		assertTrue(MatUtils.equalsExactly(nn.transform(a).getData(), nn2.transform(a).getData()));
	}
	
	@Test(expected=NaNException.class)
	public void testNoComplete() {
		final double[][] d = new double[][]{
			new double[]{1,	 		 1, 		 Double.NaN},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{Double.NaN,		 7.9,        6},
			new double[]{9,			 8,			 Double.NaN},
			new double[]{Double.NaN,		 2.9,        6.1},
			new double[]{3, 		 Double.NaN, 1},
			new double[]{Double.NaN,	 		 0, 		 0},
			new double[]{2,	 		 4, 		 Double.NaN},
			new double[]{1.4,	 	 Double.NaN, 		 6},
		};
		
		NearestNeighborImputation nn = new NearestNeighborImputation(3);
		nn.transform(d); // thrown here
	}
	
	@Test(expected=NaNException.class)
	public void testCompletelyNaNRow() {
		final double[][] d = new double[][]{
			new double[]{1,	 		 1, 		 1},
			new double[]{1, 		 2, 3},
			new double[]{5,		 7.9,        6},
			new double[]{9,			 8,			 7},
			new double[]{4,		 2.9,        6.1},
			new double[]{Double.NaN, 		 Double.NaN, Double.NaN},
			new double[]{1,	 		 0, 		 0},
			new double[]{2,	 		 4, 		 2},
			new double[]{1.4,	 	 2, 		 6},
		};
		
		NearestNeighborImputation nn = new NearestNeighborImputation(1);
		nn.transform(d); // thrown here
	}
	
	@Test
	public void testAdjustK() {
		final double[][] d = new double[][]{
			new double[]{1,	 		 1, 		 Double.NaN},
			new double[]{1, 		 Double.NaN, 3},
			new double[]{Double.NaN,		 7.9,        6},
			new double[]{9,			 8,			 Double.NaN},
			new double[]{Double.NaN,		 2.9,        6.1},
			new double[]{3, 		 2,    1},
			new double[]{Double.NaN,	 		 0, 		 0},
			new double[]{2,	 		 4, 		 Double.NaN},
			new double[]{1.4,	 	 1, 		 6},
		};
		
		NearestNeighborImputation nn = new NearestNeighborImputation(3);
		
		/*
		 * there are less complete records than k, so k should adjust
		 */
		nn.transform(d);
		assertTrue(nn.getK() < 3);
		
		// coverage love
		assertNotNull(nn.getName());
	}
	
	@Test
	public void testBootstrapNaN() {
		final double[][] d = new double[][]{
			new double[]{1,	 	 Double.NaN, 2},
			new double[]{1, 	 3, Double.NaN},
			new double[]{8.5,	 Double.NaN, 6},
			new double[]{9,		 Double.NaN, 7}
		};
		
		boolean a = false;
		try {
			new BootstrapImputation().transform(d);
		} catch(NaNException n) {
			a = true;
		} finally {
			assertTrue(a);
		}
	}
}
