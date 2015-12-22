package com.clust4j.algo.pipeline;

import static org.junit.Assert.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.KMedoids.KMedoidsPlanner;
import com.clust4j.algo.prep.MeanImputation;
import com.clust4j.algo.prep.PreProcessor;
import com.clust4j.algo.prep.Normalize;

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
		final Pipeline pipe = new Pipeline(planner, 
			new PreProcessor[]{
				Normalize.CENTER_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		final KMeans km = (KMeans) pipe.fit(mat);
		
		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
		System.out.println();
	}
	
	// Should cause a class cast exception...
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
		final Pipeline pipe = new Pipeline(planner, 
			new PreProcessor[]{
				Normalize.CENTER_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		
		@SuppressWarnings("unused")
		KMedoids km = (KMedoids)pipe.fit(mat); // Thrown here...
		System.out.println();
	}
	
	// Should cause a class cast exception...
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
		final Pipeline pipe = new Pipeline(planner, Normalize.CENTER_SCALE);
		
		@SuppressWarnings("unused")
		KMedoids km = (KMedoids)pipe.fit(mat);
		System.out.println();
	}
}
