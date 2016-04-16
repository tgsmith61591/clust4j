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
package com.clust4j.algo.pipeline;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.BaseModelTest;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.KMedoidsParameters;
import com.clust4j.algo.NearestCentroid;
import com.clust4j.algo.NearestCentroidParameters;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.algo.preprocess.impute.MeanImputation;

public class PipelineTest implements BaseModelTest {

	@Test
	public void testA() throws FileNotFoundException, IOException, ClassNotFoundException {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final KMeansParameters planner = new KMeansParameters(2).setVerbose(true);
		
		// Build the pipeline
		final UnsupervisedPipeline<KMeans> pipe = new UnsupervisedPipeline<KMeans>(planner, 
			new PreProcessor[]{
				FeatureNormalization.STANDARD_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		
		final KMeans km = pipe.fit(mat);
		
		assertTrue(km.getLabels()[0] == 0 && km.getLabels()[1] == 1);
		assertTrue(km.getLabels()[1] == km.getLabels()[2]);
		assertTrue(km.didConverge());
		System.out.println();
		
		pipe.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		@SuppressWarnings("unchecked")
		UnsupervisedPipeline<KMeans> pipe2 = (UnsupervisedPipeline<KMeans>)UnsupervisedPipeline
			.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		
		final KMeans km2 = pipe2.fit(mat);
		
		assertTrue(km2.getLabels()[0] == 0 && km2.getLabels()[1] == 1);
		assertTrue(km2.getLabels()[1] == km2.getLabels()[2]);
		assertTrue(km2.didConverge());
		
		Files.delete(TestSuite.path);
	}
	
	@Test
	public void testB() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final KMedoidsParameters planner = new KMedoidsParameters(2).setVerbose(true);
		
		// Build the pipeline
		final UnsupervisedPipeline<KMedoids> pipe = new UnsupervisedPipeline<KMedoids>(planner, 
			new PreProcessor[]{
				FeatureNormalization.STANDARD_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		
		@SuppressWarnings("unused")
		KMedoids km = pipe.fit(mat); 
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
		final KMedoidsParameters planner = new KMedoidsParameters(2).setVerbose(true);
		
		// Build the pipeline
		final UnsupervisedPipeline<KMedoids> pipe = new UnsupervisedPipeline<KMedoids>(planner, FeatureNormalization.STANDARD_SCALE);
		
		@SuppressWarnings("unused")
		KMedoids km = pipe.fit(mat);
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
		final NearestCentroidParameters planner = new NearestCentroidParameters().setVerbose(true);
		
		// Build the pipeline
		final SupervisedPipeline<NearestCentroid> pipe = new SupervisedPipeline<NearestCentroid>(planner, 
			new PreProcessor[]{
				FeatureNormalization.STANDARD_SCALE, 
				new MeanImputation(new MeanImputation.MeanImputationPlanner().setVerbose(true)) // Will create a warning
			});
		final NearestCentroid nc = pipe.fit(mat, new int[]{0,1,1});
		
		assertTrue(nc.getLabels()[0] == 0 && nc.getLabels()[1] == 1);
		assertTrue(nc.getLabels()[1] == nc.getLabels()[2]);
		System.out.println();
		
		
	}
	
	@Test
	public void testSupervisedVarArgs() throws ClassNotFoundException, FileNotFoundException, IOException {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		final NearestCentroidParameters planner = new NearestCentroidParameters().setVerbose(true);
		
		// Build the pipeline
		final SupervisedPipeline<NearestCentroid> pipe = new SupervisedPipeline<NearestCentroid>(planner, 
			FeatureNormalization.STANDARD_SCALE);
		final NearestCentroid nc = pipe.fit(mat, new int[]{0,1,1});
		
		assertTrue(nc.getLabels()[0] == 0 && nc.getLabels()[1] == 1);
		assertTrue(nc.getLabels()[1] == nc.getLabels()[2]);
		System.out.println();

		
		pipe.saveObject(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		@SuppressWarnings("unchecked")
		SupervisedPipeline<NearestCentroid> pipe2 = (SupervisedPipeline<NearestCentroid>)SupervisedPipeline
			.loadObject(new FileInputStream(TestSuite.tmpSerPath));
		
		final NearestCentroid nc2 = (NearestCentroid) pipe2.fit(mat, new int[]{0,1,1});
		
		assertTrue(nc2.getLabels()[0] == 0 && nc2.getLabels()[1] == 1);
		assertTrue(nc2.getLabels()[1] == nc2.getLabels()[2]);
		
		Files.delete(TestSuite.path);
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		assertTrue(true); // This gets tested above^^
	}
}
