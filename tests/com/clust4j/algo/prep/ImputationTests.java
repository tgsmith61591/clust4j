package com.clust4j.algo.prep;

import static org.junit.Assert.*;

import org.junit.Test;

import com.clust4j.algo.prep.BootstrapImputation.BootstrapImputationPlanner;
import com.clust4j.algo.prep.BootstrapImputation.CentralTendencyMethod;
import com.clust4j.algo.prep.MeanImputation.MeanImputationPlanner;
import com.clust4j.algo.prep.MedianImputation.MedianImputationPlanner;
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
		
		final double[][] imputed = mean.process(d);
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
			new MedianImputationPlanner());
		
		final double[][] imputed = median.process(d);
		final double[][] res = new double[][]{
			new double[]{1.5,	1, 		2},
			new double[]{1, 	1.5, 	3},
			new double[]{2,		2,      1}
		};
		
		assertTrue(MatUtils.equalsExactly(res, imputed));
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
		
		final double[][] imputed = median.process(d);
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
					.process(d);
				
				BootstrapTest.printMatrix(res);
			}
		}
	}

}
