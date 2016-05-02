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
import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.BaseModelTest;
import com.clust4j.algo.KMeans;
import com.clust4j.algo.KMeansParameters;
import com.clust4j.algo.KMedoids;
import com.clust4j.algo.KMedoidsParameters;
import com.clust4j.algo.NearestCentroid;
import com.clust4j.algo.NearestCentroidParameters;
import com.clust4j.algo.Neighborhood;
import com.clust4j.algo.NearestNeighbors;
import com.clust4j.algo.NearestNeighborsParameters;
import com.clust4j.algo.preprocess.BoxCoxTransformer;
import com.clust4j.algo.preprocess.MinMaxScaler;
import com.clust4j.algo.preprocess.PCA;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.algo.preprocess.StandardScaler;
import com.clust4j.algo.preprocess.WeightTransformer;
import com.clust4j.algo.preprocess.impute.MeanImputation;
import com.clust4j.algo.preprocess.impute.MedianImputation;
import com.clust4j.data.DataSet;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.data.TrainTestSplit;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.metrics.scoring.SupervisedMetric;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

import static com.clust4j.metrics.scoring.SupervisedMetric.*;

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
				new StandardScaler(), 
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
				new StandardScaler(), 
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
		final UnsupervisedPipeline<KMedoids> pipe = new UnsupervisedPipeline<KMedoids>(planner, new StandardScaler());
		
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
				new StandardScaler(), 
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
			new StandardScaler());
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
	
	@Test
	public void testNeighborsPipe() {
		final double[][] data = new double[][] {
			new double[] {0.005, 	 0.182751,  0.1284},
			new double[] {3.65816,   0.29518,   2.123316},
			new double[] {4.1234,    0.27395,   1.8900002}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(data);
		NearestNeighborsParameters planner = new NearestNeighborsParameters(1).setVerbose(true);
		
		PreProcessor[] pipe = new PreProcessor[]{
			new MedianImputation(),
			new StandardScaler()
		};
		
		NeighborsPipeline<NearestNeighbors> pipeline = new NeighborsPipeline<NearestNeighbors>(planner, pipe);
		Neighborhood hood = pipeline.fit(mat).getNeighbors();
		
		int[][] neighbors = hood.getIndices();
		assertTrue(MatUtils.equalsExactly(neighbors, new int[][]{
			new int[]{2},
			new int[]{2},
			new int[]{1}
		}));
		
		// coverage love
		pipeline.getName();
	}
	
	@Test
	public void testUnsupervisedFitToPredict() {
		DataSet data = TestSuite.IRIS_DATASET.shuffle();
		
		Array2DRowRealMatrix training = data.getData();	// all 150
		Array2DRowRealMatrix holdout  = new Array2DRowRealMatrix(
			MatUtils.slice(training.getData(), 0, 50), false);	// just take the first 50
		
		/*
		 * Initialize pipe
		 */
		UnsupervisedPipeline<KMeans> pipeline = new UnsupervisedPipeline<KMeans>(
			new KMeansParameters(3)
				.setVerbose(true)
				.setMetric(new GaussianKernel()),
			new StandardScaler(),
			new MinMaxScaler()
		);
		
		/*
		 * Pre-fit, test that we throw exceptions if called too early
		 */
		boolean a = false;
		try {
			pipeline.getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * Fit the pipe
		 */
		pipeline.fit(training);
		System.out.println("Silhouette: " + pipeline.silhouetteScore());
		System.out.println("Affinity:   " + pipeline.indexAffinityScore(data.getLabels()));
		
		// let's get predictions...
		int[] fit_labels = VecUtils.slice(pipeline.getLabels(),0,holdout.getRowDimension()); // only first 50!!
		int[] predicted_labels = pipeline.predict(holdout);
		
		// let's examine the affinity of the fit, and the predicted:
		double affinity = SupervisedMetric.INDEX_AFFINITY.evaluate(fit_labels, predicted_labels);
		System.out.println("Predicted affinity: " + affinity);
	}
	
	@Test
	public void testSupervisedFitToPredict() {
		DataSet data = TestSuite.BC_DATASET.shuffle();
		
		Array2DRowRealMatrix training = data.getData();	// all 300+
		Array2DRowRealMatrix holdout  = new Array2DRowRealMatrix(
			MatUtils.slice(training.getData(), 0, 50), false);	// just take the first 50
		
		/*
		 * Initialize pipe
		 */
		SupervisedPipeline<NearestCentroid> pipeline = new SupervisedPipeline<NearestCentroid>(
			new NearestCentroidParameters()
				.setVerbose(true)
				.setMetric(new GaussianKernel()),
			new StandardScaler(),
			new MinMaxScaler()
		);
		
		/*
		 * Pre-fit, test that we throw exceptions if called too early
		 */
		boolean a = false;
		try {
			pipeline.getLabels();
		} catch(ModelNotFitException m) {
			a = true;
		} finally {
			assertTrue(a);
		}
		
		/*
		 * Fit the pipe
		 */
		pipeline.fit(training, data.getLabels());
		System.out.println("Default score: " + pipeline.score());
		System.out.println("Metric score:  " + pipeline.score(BINOMIAL_ACCURACY));
		assertNotNull(pipeline.getTrainingLabels());
		assertNotNull(pipeline.getLabels());
		
		// let's get predictions...
		int[] fit_labels = VecUtils.slice(pipeline.getTrainingLabels(),0,holdout.getRowDimension()); // only first 50!!
		int[] predicted_labels = pipeline.predict(holdout);
		
		// let's examine the accuracy of the fit, and the predicted:
		System.out.println("Predicted accuracy: " + BINOMIAL_ACCURACY.evaluate(fit_labels, predicted_labels));
	}
	
	@Test
	public void testSupervisedFitToPredictWithSplit() {
		DataSet data = TestSuite.BC_DATASET.shuffle();
		TrainTestSplit split = new TrainTestSplit(data, 0.7);
		
		DataSet training = split.getTrain();
		DataSet holdout  = split.getTest();
		
		/*
		 * Initialize pipe
		 */
		SupervisedPipeline<NearestCentroid> pipeline = new SupervisedPipeline<NearestCentroid>(
			new NearestCentroidParameters()
				.setVerbose(true)
				.setMetric(new GaussianKernel()),
			new StandardScaler(),
			new MinMaxScaler()
		);
		
		/*
		 * Fit the pipe and make predictions
		 */
		pipeline.fit(training.getData(), training.getLabels());
		
		// let's get predictions...
		int[] predicted_labels = pipeline.predict(holdout.getData());
		
		// let's examine the accuracy of the holdout, and the predicted:
		System.out.println("Predicted accuracy: " + BINOMIAL_ACCURACY.evaluate(holdout.getLabels(), predicted_labels));
	}
	
	@Test
	public void testPCAPipeline() {
		DataSet data = TestSuite.BC_DATASET.shuffle();
		TrainTestSplit split = new TrainTestSplit(data, 0.7);
		
		DataSet training = split.getTrain();
		DataSet holdout  = split.getTest();
		
		/*
		 * Initialize pipe
		 */
		SupervisedPipeline<NearestCentroid> pipeline = new SupervisedPipeline<NearestCentroid>(
			new NearestCentroidParameters()
				.setVerbose(true)
				.setMetric(new GaussianKernel()),
			new StandardScaler(),
			new MinMaxScaler(),
			new PCA(0.85)
		);
		
		/*
		 * Fit the pipe and make predictions
		 */
		pipeline.fit(training.getData(), training.getLabels());
		
		// let's get predictions...
		int[] predicted_labels = pipeline.predict(holdout.getData());
		
		// let's examine the accuracy of the holdout, and the predicted:
		System.out.println("Predicted accuracy: " + BINOMIAL_ACCURACY.evaluate(holdout.getLabels(), predicted_labels));
	}
	
	@Test
	public void testBoxCoxPipeline() {
		DataSet data = TestSuite.BC_DATASET.shuffle();
		TrainTestSplit split = new TrainTestSplit(data, 0.7);
		
		DataSet training = split.getTrain();
		DataSet holdout  = split.getTest();
		
		/*
		 * Initialize pipe
		 */
		SupervisedPipeline<NearestCentroid> pipeline = new SupervisedPipeline<NearestCentroid>(
			new NearestCentroidParameters()
				.setVerbose(true)
				.setMetric(new GaussianKernel()),
			new StandardScaler(),
			new MinMaxScaler(),
			new PCA(0.85),
			new BoxCoxTransformer()
		);
		
		/*
		 * Fit the pipe and make predictions
		 */
		pipeline.fit(training.getData(), training.getLabels());
		
		// let's get predictions...
		int[] predicted_labels = pipeline.predict(holdout.getData());
		
		// let's examine the accuracy of the holdout, and the predicted:
		System.out.println("Predicted accuracy: " + BINOMIAL_ACCURACY.evaluate(holdout.getLabels(), predicted_labels));
	}
	
	@Test
	public void testWeightingPipeline() {
		DataSet data = TestSuite.BC_DATASET.shuffle();
		TrainTestSplit split = new TrainTestSplit(data, 0.7);
		
		DataSet training = split.getTrain();
		DataSet holdout  = split.getTest();
		
		
		// Initialize the weights...
		double[] weights = new double[]{
			/*
			 * There are 30 features, and each 3 are related.
			 * Maybe we just want 10 {1.0, 0.0, 0.0} to see how it works?
			 */
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0,
			1.0,0.0,0.0
		};
		
		/*
		 * Initialize pipe
		 */
		SupervisedPipeline<NearestCentroid> pipeline = new SupervisedPipeline<NearestCentroid>(
			new NearestCentroidParameters()
				.setVerbose(true)
				.setMetric(new GaussianKernel()),
			new StandardScaler(),
			new MinMaxScaler(),
			new WeightTransformer(weights)
		);
		
		/*
		 * Fit the pipe and make predictions
		 */
		pipeline.fit(training.getData(), training.getLabels());
		
		// let's get predictions...
		int[] predicted_labels = pipeline.predict(holdout.getData());
		
		// let's examine the accuracy of the holdout, and the predicted:
		System.out.println("Predicted accuracy: " + BINOMIAL_ACCURACY.evaluate(holdout.getLabels(), predicted_labels));
	
		/*
		 * This performs better than all of the other pipelines! This indicates we may not need all the features.
		 */
	}
	
	
	
	@Test
	public void testMoonSet() {
		DataSet moons = ExampleDataSets.loadToyMoons();
		
		final int[] actuals = moons.getLabels();
		RealMatrix data = moons.getData();
		
		// LE MODEL PARAMS
		KMeansParameters params = new KMeansParameters(2)
			;
		
		// Without any preprocessing:
		int[] predicted_labels = params.fitNewModel(data).getLabels();
		System.out.println(Arrays.toString(predicted_labels));
		System.out.println("Accuracy sans pre-processing: " + INDEX_AFFINITY.evaluate(actuals, predicted_labels));
	
		// With weighting
		UnsupervisedPipeline<KMeans> pipe = new UnsupervisedPipeline<KMeans>(
			params,
			new WeightTransformer(new double[]{0.5, 0.0, 2.0})
		);
		
		predicted_labels = pipe.fit(data).getLabels();
		System.out.println(Arrays.toString(predicted_labels));
		System.out.println("Accuracy with weighting: " + INDEX_AFFINITY.evaluate(actuals, predicted_labels));
	}
}
