package com.clust4j.algo;

import static com.clust4j.TestSuite.getRandom;
import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.TestSuite;
import com.clust4j.algo.BaseNeighborsModel.Algorithm;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class NearestNeighborsTests implements ClusterTest, BaseModelTest {
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
	
	final static Array2DRowRealMatrix DATA=
			new Array2DRowRealMatrix(new double[][]{
				new double[]{0.0,0.1,0.2},
				new double[]{2.3,2.5,3.1},
				new double[]{2.0,2.6,3.0},
				new double[]{0.3,0.2,0.1}
			}, false);

	@Test
	public void testWithVerbose() {
		Algorithm[] algs = new Algorithm[]{Algorithm.BALL_TREE, Algorithm.KD_TREE};
		
		for(Algorithm alg: algs) {
			new NearestNeighbors(DATA, 
				new NearestNeighbors.NearestNeighborsPlanner(1)
					.setVerbose(true)
					.setAlgorithm(alg)
					.setLeafSize(3)
					.setNormalizer(FeatureNormalization.MIN_MAX_SCALE)
					.setSeed(new Random())
					.setSep(Distance.RUSSELL_RAO) ).fit();
		}
	}
	
	@Test
	public void testWarning() {
		Neighbors n = new NearestNeighbors(DATA, 
			new NearestNeighbors.NearestNeighborsPlanner(1)
				.setSep(new GaussianKernel()));
		assertTrue(n.hasWarnings());
	}
	
	@Test
	public void testMasking() {
		int[][] indices = new int[][]{
			new int[]{1,2,3},
			new int[]{4,5,6},
			new int[]{7,8,9}
		};
		
		// Set up sample range
		int[] sampleRange = new int[]{1,2,8};
		int m = indices.length, ni = indices[0].length;
		
		
		boolean allInRow, bval;
		boolean[] dupGroups = new boolean[m];
		boolean[][] sampleMask= new boolean[m][ni];
		for(int i = 0; i < m; i++) {
			allInRow = true;
			
			for(int j = 0; j < ni; j++) {
				bval = indices[i][j] != sampleRange[i];
				sampleMask[i][j] = bval;
				allInRow &= bval;
			}
			
			dupGroups[i] = allInRow;
		}
		
		assertTrue(MatUtils.equalsExactly(sampleMask, new boolean[][]{
			new boolean[]{false, true,  true},
			new boolean[]{true,  true,  true},
			new boolean[]{true,  false, true},
		}));
		
		assertTrue(VecUtils.equalsExactly(dupGroups, 
			new boolean[]{false, true, false}));
		
		// Test de-dup cornercase
		for(int i = 0; i < m; i++)
			if(dupGroups[i])
				sampleMask[i][0] = false;
		
		assertTrue(MatUtils.equalsExactly(sampleMask, new boolean[][]{
			new boolean[]{false, true,  true},
			new boolean[]{false, true,  true},
			new boolean[]{true,  false, true},
		}));
	}

	@Test
	public void testFitResults() {
		Algorithm[] algos = new Algorithm[]{Algorithm.KD_TREE, Algorithm.BALL_TREE};
		
		for(Algorithm algo: algos) {
			double[][] expected = new double[][]{
				new double[]{1,2,3},
				new double[]{4,5,6},
				new double[]{7,8,9}
			};
			
			Array2DRowRealMatrix x = new Array2DRowRealMatrix(expected);
			
			int k= 2;
			NearestNeighbors nn = new NearestNeighbors(x, new NearestNeighbors.NearestNeighborsPlanner(k).setAlgorithm(algo)).fit();
			
			assertTrue(MatUtils.equalsExactly(expected, nn.fit_X));
			
			Neighborhood n1 = nn.getNeighbors();
			double[][] d1 = new double[][]{
				new double[]{5.196152422706632, 10.392304845413264},
				new double[]{5.196152422706632, 5.196152422706632 },
				new double[]{5.196152422706632, 10.392304845413264}
			};
			
			int[][] i1 = new int[][]{
				new int[]{1,2},
				new int[]{2,0},
				new int[]{1,0}
			};
			assertTrue(MatUtils.equalsExactly(d1, n1.getDistances()));
			assertTrue(MatUtils.equalsExactly(i1, n1.getIndices()));
			
			Neighborhood n2 = nn.getNeighbors(x);
			double[][] d2 = new double[][]{
				new double[]{0.0, 5.196152422706632},
				new double[]{0.0, 5.196152422706632},
				new double[]{0.0, 5.196152422706632}
			};
			
			// Test the toString() method for total coverage:
			String n2s = n2.toString();
			assertTrue(n2s.startsWith("Distances"));
			
			int[][] i2 = new int[][]{
				new int[]{0,1},
				new int[]{1,0},
				new int[]{2,1}
			};
			assertTrue(MatUtils.equalsExactly(d2, n2.getDistances()));
			assertTrue(MatUtils.equalsExactly(i2, n2.getIndices()));
			
			assertTrue(nn.getK() == 2);
			
			Neighborhood n3 = nn.getNeighbors(x, 1);
			double[][] d3 = new double[][]{
				new double[]{0.0},
				new double[]{0.0},
				new double[]{0.0}
			};
			
			int[][] i3 = new int[][]{
				new int[]{0},
				new int[]{1},
				new int[]{2}
			};
			
			assertTrue(MatUtils.equalsExactly(d3, n3.getDistances()));
			assertTrue(MatUtils.equalsExactly(i3, n3.getIndices()));
		}
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit1() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 1);
		nn.getNeighbors();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit2() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 1);
		nn.getNeighbors(DATA);
	}
	
	@Test
	public void testFromPlanner2() {
		NearestNeighbors nn = new NearestNeighbors.NearestNeighborsPlanner(1)
			.setAlgorithm(BaseNeighborsModel.Algorithm.BALL_TREE)
			.setLeafSize(40)
			.setScale(true)
			.setNormalizer(FeatureNormalization.MEAN_CENTER)
			.setSeed(new Random())
			.setSep(new GaussianKernel())
			.setVerbose(false).copy().buildNewModelInstance(data);
		
		assertTrue(nn.hasWarnings()); // Sep method
		assertTrue(nn.getK() == 1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor1() {
		new NearestNeighbors(DATA, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor2() {
		new NearestNeighbors(DATA, 8);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor3() {
		new NearestNeighbors(DATA, -1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor4() {
		new NearestNeighbors(DATA, new NearestNeighbors.NearestNeighborsPlanner(2).setLeafSize(-1));
	}
	
	@Test(expected=NullPointerException.class)
	public void testNPEConstructor1() {
		new NearestNeighbors(DATA, new NearestNeighbors.NearestNeighborsPlanner(2).setAlgorithm(null));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod1() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 2).fit();
		nn.getNeighbors(DATA, 9);
		// test refit
		nn.fit();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod2() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 2).fit();
		nn.getNeighbors(DATA, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod3() {
		NearestNeighbors nn = new NearestNeighbors(DATA, 2).fit();
		nn.getNeighbors(DATA, -1);
	}
	
	@Test
	public void NNTest1_ap() {
		final double[][] train_array = new double[][] {
			new double[] {0.0,  1.0,  2.0,  3.0},
			new double[] {1.0,  2.3,  2.0,  4.0},
			new double[] {9.06, 12.6, 6.5,  9.0}
		};
		
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(train_array);
		
		Neighbors nn = new NearestNeighbors(mat, 
			new NearestNeighbors.NearestNeighborsPlanner(1)
				.setVerbose(false)).fit();
		
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
}
