package com.clust4j.algo;

import static org.junit.Assert.*;
import static com.clust4j.TestSuite.getRandom;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.AffinityPropagation.AffinityPropagationPlanner;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.utils.MatUtils;

public class AffinityPropagationTests implements ClusterTest, ClassifierTest, ConvergeableTest {
	final Array2DRowRealMatrix data = ExampleDataSets.IRIS.getData();
	
	@Test
	@Override
	public void testItersElapsed() {
		assertTrue(new AffinityPropagation(data).fit().itersElapsed() > 0);
	}

	@Test
	@Override
	public void testConverged() {
		assertTrue(new AffinityPropagation(data).fit().didConverge());
	}

	@Test
	@Override
	public void testScoring() {
		new AffinityPropagation(data).fit().silhouetteScore();
	}

	@Test
	@Override
	public void testDefConst() {
		new AffinityPropagation(data);
	}

	@Test
	@Override
	public void testArgConst() {
		// Pass -- no such constructor
		return;
	}

	@Test
	@Override
	public void testPlannerConst() {
		new AffinityPropagation(data, new AffinityPropagationPlanner());
	}

	@Test
	@Override
	public void testFit() {
		new AffinityPropagation(data).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new AffinityPropagationPlanner().buildNewModelInstance(data);
	}

	@Test
	public void AffinityPropTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		final Random seed = new Random(5);
		final boolean[] b = new boolean[]{true, false};
		
		for(boolean bool: b) {
			AffinityPropagation a = 
					new AffinityPropagation(mat, new AffinityPropagation
						.AffinityPropagationPlanner()
							.useGaussianSmoothing(bool)
							.setVerbose(true)
							.setSeed(seed)).fit();
					
					final int[] labels = a.getLabels();
					assertTrue(labels.length == 5);
					assertTrue(labels[0] == labels[1]);
					assertTrue(labels[2] == labels[3]);
					if(bool) assertTrue(a.getNumberOfIdentifiedClusters() == 3);
					assertTrue(a.didConverge());
					assertTrue(labels[0] == 0);
					assertTrue(labels[2] == 1);
					assertTrue(labels[4] == 2);
		}
	}
	
	@Test
	public void AffinityPropLoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1000, 10);
		new AffinityPropagation(mat, new AffinityPropagation
			.AffinityPropagationPlanner()
				.setVerbose(true)).fit();
	}
	
	@Test
	public void NNTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  2.0,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		Neighbors nn = new NearestNeighbors(mat, 
			new NearestNeighbors.NearestNeighborsPlanner(1)
				.setVerbose(true)).fit();
		
		int[][] ne = nn.getNeighbors().getIndices();
		assertTrue(ne[0].length == 1);
		assertTrue(ne[0].length == 1);
		System.out.println();
		
		nn = new RadiusNeighbors(mat, 
			new RadiusNeighbors.RadiusNeighborsPlanner(3.0)
				.setVerbose(true)).fit();
		
		ne = nn.getNeighbors().getIndices();
		assertTrue(ne[0].length == 1);
		assertTrue(ne[1].length == 1);
		assertTrue(ne[2].length == 0);
	}
	
	@Test
	public void AffinityPropKernelTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.001,  1.002,   0.481,   3.029,  2.019},
			new double[] {0.426,  1.291,   0.615,   2.997,  3.018},
			new double[] {6.019,  5.069,   3.681,   5.998,  5.182},
			new double[] {5.928,  4.972,   4.013,   6.123,  5.004},
			new double[] {12.091, 153.001, 859.013, 74.852, 3.091}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		final Random seed = new Random(5);
		final boolean[] b = new boolean[]{true, false};
		
		for(boolean bool: b) {
			AffinityPropagation a = 
					new AffinityPropagation(mat, new AffinityPropagation
						.AffinityPropagationPlanner()
							.useGaussianSmoothing(bool)
							.setVerbose(true)
							.setSep(new GaussianKernel())
							.setSeed(seed)).fit();
					
					final int[] labels = a.getLabels();
					assertTrue(labels.length == 5);
					assertTrue(labels[0] == labels[1]);
					assertTrue(labels[2] == labels[3]);
					if(bool) assertTrue(a.getNumberOfIdentifiedClusters() == 3);
					assertTrue(a.didConverge());
		}
	}
	
	@Test
	public void AffinityPropKernelLoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1000, 10);
		new AffinityPropagation(mat, new AffinityPropagation
			.AffinityPropagationPlanner()
				.setSep(new GaussianKernel())
				.setVerbose(true)).fit();
	}

	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		AffinityPropagation ap = new AffinityPropagation(data, 
			new AffinityPropagation
				.AffinityPropagationPlanner()
					.setVerbose(true)).fit();
		
		double[][] a = ap.getAvailabilityMatrix();
		ap.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		AffinityPropagation ap2 = (AffinityPropagation)AffinityPropagation.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(MatUtils.equalsExactly(a, ap2.getAvailabilityMatrix()));
		assertTrue(ap2.equals(ap));
		Files.delete(TestSuite.path);
	}
}
