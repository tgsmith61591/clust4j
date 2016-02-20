package com.clust4j.algo;

import static com.clust4j.TestSuite.getRandom;
import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;

public class NearestNeighborsTests implements ClusterTest {
	final Array2DRowRealMatrix data = ExampleDataSets.IRIS.getData();
	
	@Test
	@Override
	public void testDefConst() {
		new NearestNeighbors(data);
	}

	@Test
	@Override
	public void testArgConst() {
		new NearestNeighbors(data, 3);
	}

	@Test
	@Override
	public void testPlannerConst() {
		new NearestNeighbors(data, new NearestNeighborsPlanner());
	}

	@Test
	@Override
	public void testFit() {
		new NearestNeighbors(data).fit();
		new NearestNeighbors(data).fit().fit(); // test the extra fit for any exceptions
		new NearestNeighbors(data, 3).fit();
		new NearestNeighbors(data, new NearestNeighborsPlanner()).fit();
		new NearestNeighbors(data, new NearestNeighborsPlanner(5)).fit();
	}

	@Test
	@Override
	public void testFromPlanner() {
		new NearestNeighborsPlanner().buildNewModelInstance(data);
		new NearestNeighborsPlanner(4).buildNewModelInstance(data);
	}
	
	@Test
	@Override
	public void testSerialization() throws IOException, ClassNotFoundException {
		NearestNeighbors nn = new NearestNeighbors(data, 
			new NearestNeighbors.NearestNeighborsPlanner(5)
				.setVerbose(true)
				.setScale(true)).fit();
		
		final int[][] c = nn.getNeighbors().getIndices();
		nn.saveModel(new FileOutputStream(TestSuite.tmpSerPath));
		assertTrue(TestSuite.file.exists());
		
		NearestNeighbors nn2 = (NearestNeighbors)NearestNeighbors.loadModel(new FileInputStream(TestSuite.tmpSerPath));
		assertTrue(MatUtils.equalsExactly(nn2.getNeighbors().getIndices(), c));
		assertTrue(nn2.equals(nn));
		assertTrue(nn.equals(nn)); // test the ref return
		assertFalse(nn.equals(new Object()));
		
		Files.delete(TestSuite.path);
	}

	@Test
	public void NN_KNEAREST_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		
		final int[] ks = new int[]{1, 5, 10};
		for(int k: ks) {
			new NearestNeighbors(mat, 
				new NearestNeighbors.NearestNeighborsPlanner(k)
					.setVerbose(true)).fit();
		}
	}
	
	@Test
	public void NN_RADIUS_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		
		final double[] radii = new double[]{0.5, 5.0, 10.0};
		for(double radius: radii) {
			new RadiusNeighbors(mat, 
				new RadiusNeighbors.RadiusNeighborsPlanner(radius)
					.setVerbose(true)).fit();
			System.out.println();
		}
	}
	
	@Test
	public void NNTest1() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  2.0,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		NearestNeighbors nn = new NearestNeighbors(mat, 
			new NearestNeighbors.NearestNeighborsPlanner(1)
				.setVerbose(true)
				.setSep(new GaussianKernel())).fit();
		
		int[][] ne = nn.getNeighbors().getIndices();
		assertTrue(ne[0].length == 1);
		assertTrue(ne[0].length == 1);
		System.out.println();
		
		new RadiusNeighbors(mat, 
			new RadiusNeighbors.RadiusNeighborsPlanner(3.0)
				.setVerbose(true)
				.setSep(new GaussianKernel()) ).fit();
	}
	
	@Test
	public void NN_kernel_KNEAREST_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		
		final int[] ks = new int[]{1, 5, 10};
		for(int k: ks) {
			new NearestNeighbors(mat, 
				new NearestNeighbors.NearestNeighborsPlanner(k)
					.setVerbose(true)
					.setSep(new GaussianKernel()) ).fit();
		}
	}
	
	@Test
	public void NN_kernel_RADIUS_LoadTest() {
		final Array2DRowRealMatrix mat = getRandom(1500, 10);
		
		final double[] radii = new double[]{0.5, 5.0, 10.0};
		for(double radius: radii) {
			new RadiusNeighbors(mat, 
				new RadiusNeighbors.RadiusNeighborsPlanner(radius)
					.setVerbose(true)
					.setSep(new GaussianKernel()) ).fit();
			System.out.println();
		}
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE1() {
		new NearestNeighbors(data, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE2() {
		new NearestNeighbors(data, 151);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE3() {
		new NearestNeighbors(data, new NearestNeighborsPlanner(0));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAE4() {
		new NearestNeighbors(data, new NearestNeighborsPlanner(151));
	}
	
	@Test
	public void testMiscellany() {
		assertTrue(new NearestNeighborsPlanner().getNormalizer().equals(AbstractClusterer.DEF_NORMALIZER));
	}
	
	@Test
	public void testK() {
		assertTrue(new NearestNeighbors(data, 3).getK() == 3);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFNeigb1() {
		new NearestNeighbors(data).getNeighbors(data);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFNeigb2() {
		new NearestNeighbors(data).getNeighbors(data, 2);
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFNeigb3() {
		new NearestNeighbors(data).getNeighbors();
	}
	
	@Test
	public void testGetNeighb() {
		new NearestNeighbors(data).fit().getNeighbors().getDistances();
		new NearestNeighbors(data).fit().getNeighbors(data).getDistances();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testGetNeighbIAE1() {
		new NearestNeighbors(data).fit().getNeighbors(data, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testGetNeighbIAE2() {
		new NearestNeighbors(data).fit().getNeighbors(data, 151);
	}
}
