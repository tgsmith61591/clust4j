package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.BaseNeighborsModel.Algorithm;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.utils.Distance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

public class TestNearestNeighbors {
	final static Array2DRowRealMatrix data=
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
			new NearestNeighbors(data, 
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
		Neighbors n = new NearestNeighbors(data, 
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
		NearestNeighbors nn = new NearestNeighbors(data, 1);
		nn.getNeighbors();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testNotFit2() {
		NearestNeighbors nn = new NearestNeighbors(data, 1);
		nn.getNeighbors(data);
	}
	
	@Test
	public void testFromPlanner() {
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
		new NearestNeighbors(data, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor2() {
		new NearestNeighbors(data, 8);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor3() {
		new NearestNeighbors(data, -1);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEConstructor4() {
		new NearestNeighbors(data, new NearestNeighbors.NearestNeighborsPlanner(2).setLeafSize(-1));
	}
	
	@Test(expected=NullPointerException.class)
	public void testNPEConstructor1() {
		new NearestNeighbors(data, new NearestNeighbors.NearestNeighborsPlanner(2).setAlgorithm(null));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod1() {
		NearestNeighbors nn = new NearestNeighbors(data, 2).fit();
		nn.getNeighbors(data, 9);
		// test refit
		nn.fit();
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod2() {
		NearestNeighbors nn = new NearestNeighbors(data, 2).fit();
		nn.getNeighbors(data, 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testIAEMethod3() {
		NearestNeighbors nn = new NearestNeighbors(data, 2).fit();
		nn.getNeighbors(data, -1);
	}
}
