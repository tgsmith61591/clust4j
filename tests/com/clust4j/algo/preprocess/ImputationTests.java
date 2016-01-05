package com.clust4j.algo.preprocess;

import static org.junit.Assert.*;

import org.junit.Test;

import com.clust4j.algo.preprocess.impute.BootstrapImputation;
import com.clust4j.algo.preprocess.impute.MeanImputation;
import com.clust4j.algo.preprocess.impute.MedianImputation;
import com.clust4j.algo.preprocess.impute.NearestNeighborImputation;
import com.clust4j.algo.preprocess.impute.BootstrapImputation.BootstrapImputationPlanner;
import com.clust4j.algo.preprocess.impute.MatrixImputation.CentralTendencyMethod;
import com.clust4j.algo.preprocess.impute.MeanImputation.MeanImputationPlanner;
import com.clust4j.algo.preprocess.impute.MedianImputation.MedianImputationPlanner;
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
		
		final double[][] imputed = mean.operate(d);
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
		
		final double[][] imputed = median.operate(d);
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
		
		final double[][] imputed = median.operate(d);
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
					.operate(d);
				
				BootstrapTest.printMatrix(res);
			}
		}
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
		
		final double[][] imputed = nni.operate(d);
		final double[][] res = new double[][]{
			new double[]{1,	 		 1, 		 2},
			new double[]{1, 		 1, 		 3},
			new double[]{8.5,		 7.9,        6},
			new double[]{9,			 8,			 6}
		};
		
		assertTrue(MatUtils.equalsExactly(res, imputed));
		System.out.println();
	}

}
