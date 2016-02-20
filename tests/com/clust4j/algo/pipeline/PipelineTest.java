package com.clust4j.algo.pipeline;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.KMedoids.KMedoidsPlanner;
import com.clust4j.algo.NearestCentroid;
import com.clust4j.algo.NearestCentroid.NearestCentroidPlanner;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.algo.preprocess.impute.MeanImputation;

public class PipelineTest {

	@Test
	public void testA() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final KMeans.KMeansPlanner planner = new KMeans.KMeansPlanner(2).setVerbose(true);
		
		// Build the pipeline
		final UnsupervisedPipeline pipe = new UnsupervisedPipeline(planner, 
			new PreProcessor[]{
				FeatureNormalization.STANDARD_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		final KMeans km = (KMeans) pipe.fit(mat);
		
		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
		System.out.println();
	}
	
	@Test
	public void testB() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final KMedoids.KMedoidsPlanner planner = new KMedoids.KMedoidsPlanner(2).setVerbose(true);
		
		// Build the pipeline
		final UnsupervisedPipeline pipe = new UnsupervisedPipeline(planner, 
			new PreProcessor[]{
				FeatureNormalization.STANDARD_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		
		@SuppressWarnings("unused")
		KMedoids km = (KMedoids)pipe.fit(mat); 
		System.out.println();
	}
	
	@Test
	public void testVarArgs() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final KMedoidsPlanner planner = new KMedoidsPlanner(2).setVerbose(true);
		
		// Build the pipeline
		final UnsupervisedPipeline pipe = new UnsupervisedPipeline(planner, FeatureNormalization.STANDARD_SCALE);
		
		@SuppressWarnings("unused")
		KMedoids km = (KMedoids)pipe.fit(mat);
		System.out.println();
	}
	
	
	
	
	
	@Test
	public void testSupervisedA() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final NearestCentroidPlanner planner = new NearestCentroidPlanner().setVerbose(true);
		
		// Build the pipeline
		final SupervisedPipeline pipe = new SupervisedPipeline(planner, 
			new PreProcessor[]{
				FeatureNormalization.STANDARD_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		final NearestCentroid nc = (NearestCentroid) pipe.fit(mat, new int[]{0,1,1});
		
		assertTrue(nc.getLabels()[0] == 0 && nc.getLabels()[1] == 1);
		assertTrue(nc.getLabels()[1] == nc.getLabels()[2]);
		System.out.println();
	}
	
	@Test
	public void testSupervisedVarArgs() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final NearestCentroidPlanner planner = new NearestCentroidPlanner().setVerbose(true);
		
		// Build the pipeline
		final SupervisedPipeline pipe = new SupervisedPipeline(planner, 
			FeatureNormalization.STANDARD_SCALE);
		final NearestCentroid nc = (NearestCentroid) pipe.fit(mat, new int[]{0,1,1});
		
		assertTrue(nc.getLabels()[0] == 0 && nc.getLabels()[1] == 1);
		assertTrue(nc.getLabels()[1] == nc.getLabels()[2]);
		System.out.println();
	}
}
